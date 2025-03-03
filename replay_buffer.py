import collections
import numpy as np
import torch

class ReplayBuffer:
    """
    Replay Buffer for Dreamer.
    This version is made fully swappable with the target code.
    It stores full episodes and samples contiguous sequences.
    """
    def __init__(self, size, obs_shape, action_size, seq_len, batch_size):
        self.capacity = size                  # Maximum number of episodes
        self.obs_shape = obs_shape            # Observation shape (e.g. (3, 64, 64))
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
        self.ongoing_episode.append((obs["image"], ac, rew, done))
        self.steps += 1 
        
        if done:
            self._finalize_episode()
            self.episodes = self.episodes + 1

    def _finalize_episode(self):
        """
        Converts the ongoing episode to a dict of numpy arrays and saves it.
        Discards episodes that are too short.
        """
        if len(self.ongoing_episode) < self.seq_len:
            self.ongoing_episode = []  # Discard short episodes
            return
        episode = {
            'obs': np.array([step[0] for step in self.ongoing_episode], dtype=np.uint8),
            'actions': np.array([step[1] for step in self.ongoing_episode], dtype=np.float32),
            'rewards': np.array([step[2] for step in self.ongoing_episode], dtype=np.float32),
            'terminals': np.array([step[3] for step in self.ongoing_episode], dtype=np.float32),
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
        batch_obs, batch_actions, batch_rewards, batch_terminals = [], [], [], []
        for _ in range(self.batch_size):
            episode = np.random.choice(list(self.buffer))  # Select a full episode randomly
            max_start = len(episode['obs']) - self.seq_len
            start_idx = np.random.randint(0, max_start + 1)
            obs_seq = episode['obs'][start_idx:start_idx + self.seq_len]
            action_seq = episode['actions'][start_idx:start_idx + self.seq_len]
            reward_seq = episode['rewards'][start_idx:start_idx + self.seq_len]
            terminal_seq = episode['terminals'][start_idx:start_idx + self.seq_len]
            batch_obs.append(obs_seq)
            batch_actions.append(action_seq)
            batch_rewards.append(reward_seq)
            batch_terminals.append(terminal_seq)
        
        # Convert lists to tensors (initial shape: [batch_size, seq_len, ...])
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        batch_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)
        batch_terminals = torch.tensor(np.array(batch_terminals), dtype=torch.float32)
        
        # Transpose to shape [seq_len, batch_size, ...]
        batch_obs = batch_obs.permute(1, 0, *range(2, len(batch_obs.shape)))
        batch_actions = batch_actions.permute(1, 0, *range(2, len(batch_actions.shape)))
        batch_rewards = batch_rewards.permute(1, 0)
        batch_terminals = batch_terminals.permute(1, 0)
        
        return batch_obs, batch_actions, batch_rewards, batch_terminals
