import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

_str_to_activation = {
    'relu': nn.ReLU(),
    'elu' : nn.ELU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class RSSM(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size,  hidden_size, obs_embed_size, activation):

        super().__init__()

        self.action_size = action_size
        self.stoch_size  = stoch_size   
        self.deter_size  = deter_size   # GRU hidden units
        self.hidden_size = hidden_size  # intermediate fc_layers hidden units 
        self.embedding_size = obs_embed_size

        self.act_fn = _str_to_activation[activation]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.fc_state_action = nn.Linear(self.stoch_size + self.action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior  = nn.Linear(self.hidden_size, 2*self.stoch_size)
        self.fc_embed_posterior = nn.Linear(self.embedding_size + self.deter_size, self.hidden_size)
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2*self.stoch_size)


    def init_state(self, batch_size, device):

        return dict(
            mean = torch.zeros(batch_size, self.stoch_size).to(device),
            std  = torch.zeros(batch_size, self.stoch_size).to(device),
            stoch = torch.zeros(batch_size, self.stoch_size).to(device),
            deter = torch.zeros(batch_size, self.deter_size).to(device))

    def get_dist(self, mean, std):

        distribution = distributions.Normal(mean, std)
        distribution = distributions.independent.Independent(distribution, 1)
        return distribution

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm)
        posterior_embed = self.act_fn(self.fc_embed_posterior(torch.cat([obs_embed, prior['deter']], dim=-1)))
        posterior = self.fc_state_posterior(posterior_embed)
        mean, std = torch.chunk(posterior, 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        posterior = {'mean': mean, 'std': std, 'stoch': sample, 'deter': prior['deter']}
        return prior, posterior

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):

        state_action = self.act_fn(self.fc_state_action(torch.cat([prev_state['stoch']*nonterm, prev_action], dim=-1)))
        deter = self.rnn(state_action, prev_state['deter']*nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        prior = {'mean': mean, 'std': std, 'stoch': sample, 'deter': deter}
        return prior

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):

        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t]* nonterms[t]
            prior_state, posterior_state = self.observe_step(prev_state, prev_action, obs_embed[t], nonterms[t])
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actor, prev_state, horizon):

        rssm_state = prev_state
        next_states = []

        for t in range(horizon):
            action = actor(torch.cat([rssm_state['stoch'], rssm_state['deter']], dim=-1).detach())
            rssm_state = self.imagine_step(rssm_state, action)
            next_states.append(rssm_state)

        next_states = self.stack_states(next_states)
        return next_states

    def stack_states(self, states, dim=0):

        return dict(
            mean = torch.stack([state['mean'] for state in states], dim=dim),
            std  = torch.stack([state['std'] for state in states], dim=dim),
            stoch = torch.stack([state['stoch'] for state in states], dim=dim),
            deter = torch.stack([state['deter'] for state in states], dim=dim))

    def detach_state(self, state):

        return dict(
            mean = state['mean'].detach(),
            std  = state['std'].detach(),
            stoch = state['stoch'].detach(),
            deter = state['deter'].detach())

    def seq_to_batch(self, state):

        return dict(
            mean = torch.reshape(state['mean'], (state['mean'].shape[0]* state['mean'].shape[1], *state['mean'].shape[2:])),
            std = torch.reshape(state['std'], (state['std'].shape[0]* state['std'].shape[1], *state['std'].shape[2:])),
            stoch = torch.reshape(state['stoch'], (state['stoch'].shape[0]* state['stoch'].shape[1], *state['stoch'].shape[2:])),
            deter = torch.reshape(state['deter'], (state['deter'].shape[0]* state['deter'].shape[1], *state['deter'].shape[2:])))


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, embed_size, activation, depth=32):
        """
        Beginner-friendly version of ConvEncoder.
        Args:
            input_shape: tuple, e.g., (3, 64, 64)
            embed_size: desired output embedding size
            activation: string key (e.g. 'relu', 'elu')
            depth: base number of channels
        """
        super().__init__()
        self.input_shape = input_shape
        self.embed_size = embed_size
        self.depth = depth
        self.act_fn = _str_to_activation[activation]
        self.kernels = [4,4,4,4]

        # Define layers explicitly (no nn.Sequential)
        self.h1 = nn.Conv2d(input_shape[0], 1 * self.depth, self.kernels[0], stride=2)
        self.h2 = nn.Conv2d(1 * self.depth, 2 * self.depth, self.kernels[1], stride=2)
        self.h3 = nn.Conv2d(2 * self.depth, 4 * self.depth, self.kernels[2], stride=2)
        self.h4 = nn.Conv2d(4 * self.depth, 8 * self.depth, self.kernels[3], stride=2)

        # Fully connected layer
        self.fc = nn.Identity() if self.embed_size == 1024 else nn.Linear(1024, self.embed_size)

    def forward(self, inputs):
        x = inputs.reshape(-1, *self.input_shape) 

        x = self.act_fn(self.h1(x))
        x = self.act_fn(self.h2(x))
        x = self.act_fn(self.h3(x))
        x = self.act_fn(self.h4(x))

        x = torch.reshape(x, (*inputs.shape[:-3], -1))
        x = self.fc(x)

        return x

class ConvDecoder(nn.Module):
    def __init__(self, stoch_size, deter_size, output_shape, activation, depth=32):
        """         
        Args:
            stoch_size: Size of stochastic state vector
            deter_size: Size of deterministic state vector
            output_shape: Shape of the reconstructed image (C, H, W)
            activation: Activation function as a string (e.g., 'relu', 'elu')
            depth: Base number of channels
        """
        super().__init__()
        self.output_shape = output_shape
        self.depth = depth
        self.act_fn = _str_to_activation[activation]
        self.kernels = [5, 5, 6, 6]  

        # Fully connected layer to reshape feature vector into initial conv shape
        self.dense = nn.Linear(stoch_size + deter_size, 32 * depth)

        
        self.h1 = nn.ConvTranspose2d(32 * depth, 4 * depth, self.kernels[0], stride=2)
        self.h2 = nn.ConvTranspose2d(4 * depth, 2 * depth, self.kernels[1], stride=2)
        self.h3 = nn.ConvTranspose2d(2 * depth, 1 * depth, self.kernels[2], stride=2)
        self.h4 = nn.ConvTranspose2d(1 * depth, output_shape[0], self.kernels[3], stride=2)

    def forward(self, features):
        """
        Args:
            features: Latent feature vector of shape [B, stoch_size + deter_size]

        Returns:
            Normal distribution over reconstructed image.
        """
        batch_shape = features.shape[:-1]  # Preserve batch dimensions
        x = self.dense(features)
        x = torch.reshape(x, [-1, 32*self.depth, 1, 1])  # Reshape for transposed convs

        
        x = self.act_fn(self.h1(x))
        x = self.act_fn(self.h2(x))
        x = self.act_fn(self.h3(x))
        mean = self.h4(x)  

        
        mean = mean.view(*batch_shape, *self.output_shape)

        # Return as a normal distribution with fixed std dev
        out_dist = distributions.Independent(distributions.Normal(mean, 1.0), len(self.output_shape))
        return out_dist

# used for reward and value models
class DenseDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, n_layers, units, activation, dist):

        super().__init__()

        self.input_size = stoch_size + deter_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        layers=[]

        for i in range(self.n_layers):
            in_ch = self.input_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn) 

        layers.append(nn.Linear(self.units, int(np.prod(self.output_shape))))

        self.model = nn.Sequential(*layers)

    def forward(self, features):

        out = self.model(features)

        if self.dist == 'normal':
            return distributions.independent.Independent(
                distributions.Normal(out, 1), len(self.output_shape))
        if self.dist == 'binary':
            return distributions.independent.Independent(
                distributions.Bernoulli(logits =out), len(self.output_shape))
        if self.dist == 'none':
            return out

        raise NotImplementedError(self.dist)

class ActionDecoder(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, n_layers, units, 
                        activation, min_std=1e-4, init_std=5, mean_scale=5):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.units = units  
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        layers = []
        for i in range(self.n_layers):
            in_ch = self.stoch_size + self.deter_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2*self.action_size))
        self.action_model = nn.Sequential(*layers)

    def forward(self, features, deter=False):

        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1) 

        raw_init_std = np.log(np.exp(self._init_std)-1)
        action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        action_std = F.softplus(std + raw_init_std) + self._min_std

        dist = distributions.Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = distributions.independent.Independent(dist, 1)
        dist = SampleDist(dist)

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)


class TanhBijector(distributions.Transform):

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))

class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
