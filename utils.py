import os

import torch
from torch.utils import tensorboard


def pad(episodes):
  """Pads episodes to all be the same length by repeating the last exp.

  Args:
    episodes (list[list[Experience]]): episodes to pad.

  Returns:
    padded_episodes (list[list[Experience]]): now of shape
      (batch_size, max_len)
    mask (torch.BoolTensor): of shape (batch_size, max_len) with value 0 for
      padded experiences.
  """
  max_len = max(len(episode) for episode in episodes)
  mask = torch.zeros((len(episodes), max_len), dtype=torch.bool)
  padded_episodes = []
  for i, episode in enumerate(episodes):
    padded = episode + [episode[-1]] * (max_len - len(episode))
    padded_episodes.append(padded)
    mask[i, :len(episode)] = True
  return padded_episodes, mask


class EpisodeAndStepWriter(object):
  """Logs to tensorboard against both episode and number of steps."""

  def __init__(self, log_dir):
    self._episode_writer = tensorboard.SummaryWriter(
        os.path.join(log_dir, "episode"))
    self._step_writer = tensorboard.SummaryWriter(
        os.path.join(log_dir, "step"))

  def add_scalar(self, key, value, episode, step):
    self._episode_writer.add_scalar(key, value, episode)
    self._step_writer.add_scalar(key, value, step)
