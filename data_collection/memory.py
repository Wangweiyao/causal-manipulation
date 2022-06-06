import numpy as np
import torch


class ExperienceReplay():
  def __init__(self, size, view_nbr, observation_size, action_size, device):
    self.device = device
    self.size = size
    self.view_nbr = view_nbr
    self.observations = np.zeros((size, 6, 128, 128), np.uint8)
    self.observations2 = np.zeros((size, view_nbr, 3, 128, 128), np.uint8)
    self.observations2cam = np.zeros((size, view_nbr), np.uint8)
    self.actions = np.zeros((size, action_size), dtype=np.float32)
    self.rewards = np.zeros((size, ), dtype=np.float32) 
    self.nonterminals = np.zeros((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total

  def append(self, observation, ob2, ob2cam, action, reward, done):
    self.observations[self.idx] = observation#.numpy()
    self.observations2[self.idx] = ob2#.numpy()
    self.observations2cam[self.idx] = ob2cam
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
      if self.nonterminals[idxs].min() == 0:
        valid_idx = False
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    observations2 = torch.as_tensor(self.observations2[vec_idxs].astype(np.float32))
    return observations.reshape(L, n, *observations.shape[1:]), \
          observations2.reshape(L, n, *observations2.shape[1:]), \
          self.observations2cam[vec_idxs].reshape(L, n, *self.observations2cam.shape[1:]), \
          self.actions[vec_idxs].reshape(L, n, -1), \
          self.rewards[vec_idxs].reshape(L, n), \
          self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    return [torch.as_tensor(item).to(device=self.device) for item in batch]
