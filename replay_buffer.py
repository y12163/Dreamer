import collections
import numpy as np
import torch
import random

class ReplayBuffer:
    """
    Replay Buffer for Dreamer.
    Stores full episodes and samples contiguous sequences.
    """
    def __init__(self, size, obs_shape, proprio_shape, action_size, seq_len, batch_size):
        self.capacity = size                  # Maximum number of episodes
        self.obs_shape = obs_shape            # Observation shape (e.g. (3, 64, 64))
        self.proprio_shape= proprio_shape  
        self.action_size = action_size        # Action dimension
        self.seq_len = seq_len                # Sequence length for sampling
        self.batch_size = batch_size          # Number of sequences per batch
        self.buffer = collections.deque(maxlen=size)  # Buffer to store episodes
        self.ongoing_episode = []             # Temporary storage for current episode
        self.steps, self.episodes = 0, 0

    def add(self, obs, ac, rew, done):
        """
        Adds a single transition to the current episode.
        Expects obs to be a dict with an 'image' key.
        """
        # We store only the image (so that memory is lower)
        self.ongoing_episode.append((obs["image"], obs['proprio'],ac, rew, done))
        self.steps += 1 
        if done:
            self._finalize_episode()
            self.episodes += 1

    def _finalize_episode(self):
        """
        Converts the ongoing episode to a dict of NumPy arrays and saves it.
        Discards episodes that are too short.
        """
        if len(self.ongoing_episode) < self.seq_len:
            self.ongoing_episode = []  # Discard short episodes
            return
        episode = {
            'obs_image': np.array([step[0] for step in self.ongoing_episode], dtype=np.uint8),
            'obs_proprio': np.array([step[1] for step in self.ongoing_episode], dtype=np.float32),
            'actions': np.array([step[2] for step in self.ongoing_episode], dtype=np.float32),
            'rewards': np.array([step[3] for step in self.ongoing_episode], dtype=np.float32),
            'terminals': np.array([step[4] for step in self.ongoing_episode], dtype=np.float32),
        }
        self.buffer.append(episode)
        self.ongoing_episode = []  # Reset after storing

    def sample(self):
        """
        Samples full contiguous sequences from stored episodes.
        Returns tensors in the shape:
          obs:      [seq_len, batch_size, *obs_shape]
          actions:  [seq_len, batch_size, action_size]
          rewards:  [seq_len, batch_size]
          terminals:[seq_len, batch_size]
        """
        # Preallocate NumPy arrays for the batch.
        batch_obs_image = np.empty((self.batch_size, self.seq_len, *self.obs_shape), dtype=np.uint8)
        batch_obs_proprio = np.empty((self.batch_size, self.seq_len,self.proprio_shape), dtype=np.float32)
        batch_actions = np.empty((self.batch_size, self.seq_len, self.action_size), dtype=np.float32)
        batch_rewards = np.empty((self.batch_size, self.seq_len), dtype=np.float32)
        batch_terminals = np.empty((self.batch_size, self.seq_len), dtype=np.float32)

        # To avoid repeatedly converting the deque to a list, do it once.
        episodes = list(self.buffer)
        for i in range(self.batch_size):
            # Randomly select an episode
            episode = random.choice(episodes)
            max_start = len(episode['obs_image']) - self.seq_len
            start_idx = np.random.randint(0, max_start + 1)
            batch_obs_image[i] = episode['obs_image'][start_idx:start_idx + self.seq_len]
            batch_obs_proprio[i] = episode['obs_proprio'][start_idx:start_idx + self.seq_len]
            batch_actions[i] = episode['actions'][start_idx:start_idx + self.seq_len]
            batch_rewards[i] = episode['rewards'][start_idx:start_idx + self.seq_len]
            batch_terminals[i] = episode['terminals'][start_idx:start_idx + self.seq_len]
        
        # Convert to tensors using torch.from_numpy (faster than torch.tensor(np.array(...)))
        batch_obs_image = torch.from_numpy(batch_obs_image).float()  # [batch_size, seq_len, *obs_shape]
        batch_obs_proprio = torch.from_numpy(batch_obs_proprio).float()  # [batch_size, seq_len, *obs_shape]
        batch_actions = torch.from_numpy(batch_actions).float()
        batch_rewards = torch.from_numpy(batch_rewards).float()
        batch_terminals = torch.from_numpy(batch_terminals).float()

        # Permute to [seq_len, batch_size, ...]
        # For observations and actions, we need to move the sequence dimension (axis 1) to axis 0.
        batch_obs_image = batch_obs_image.permute(1, 0, *range(2, batch_obs_image.ndim)).contiguous()
        batch_obs_proprio = batch_obs_proprio.permute(1, 0, *range(2, batch_obs_proprio.ndim)).contiguous()
        batch_actions = batch_actions.permute(1, 0, *range(2, batch_actions.ndim)).contiguous()
        batch_rewards = batch_rewards.permute(1, 0).contiguous()
        batch_terminals = batch_terminals.permute(1, 0).contiguous()

        return batch_obs_image,batch_obs_proprio, batch_actions, batch_rewards, batch_terminals
