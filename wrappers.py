import gym
from gym import spaces
import numpy as np
import torch
import meta_exploration


class MultiEpisodeWrapper(gym.Wrapper):
  """Allows for outer episodes (trials in RL^2) consisting of multiple inner
  episodes by subsuming the intermediate dones.

  Dones are already labeled by the InstructionState.
  """

  def __init__(self, env, episodes_per_trial=2):
    super().__init__(env)
    assert isinstance(env, meta_exploration.InstructionWrapper)

    self._episodes_so_far = 0
    self._episodes_per_trial = episodes_per_trial

  def step(self, action):
    next_state, reward, done, info = super().step(action)

    if done:
      self._episodes_so_far += 1
      # Need to copy reward from previous state
      next_state = self.env.reset()._replace(
          prev_reward=next_state.prev_reward, done=done)

    trial_done = self._episodes_so_far == self._episodes_per_trial
    return next_state, reward, trial_done, info

  def reset(self):
    self._episodes_so_far = 0
    state = super().reset()
    return state

  def render(self):
    return self.env.render()
