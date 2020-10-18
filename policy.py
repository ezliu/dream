import abc
import numpy as np


class Policy(abc.ABC):
  """Implements a policy \pi(a | s)."""

  @abc.abstractmethod
  def act(self, state):
    """Returns an action given the current state.

    Args:
      state (object): current state.

    Returns:
      action (int): action to take.
    """
    raise NotImplementedError()

  @property
  def stats(self):
    """Returns a dict of relevant statistics about the policy."""
    return {}


class RandomPolicy(Policy):
  """Acts uniformly at random on discrete actions."""

  def __init__(self, action_space):
    """Constructs on a discrete action space.

    Args:
      action_space (spaces.Discrete): action space of the environment.
    """
    self._action_space = action_space

  def act(self, state, hidden_state, test=False):
    del state, hidden_state, test

    return np.random.randint(self._action_space.n), None

  def update(self, experience):
    pass


class ConstantActionPolicy(Policy):
  """Always returns the same action."""

  def __init__(self, action):
    self._action = action

  def act(self, state, hidden_state, test=False):
    del state, hidden_state, test

    return self._action, None

  def update(self, experience):
    pass
