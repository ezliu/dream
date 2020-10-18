import abc

import rl


class TrajectoryExperience(object):
  """An experience that holds a reference to the trajectory it came from.
  This should be substitutable wherever Experience is used.
  
  In particular, it holds:
    - trajectory (list[Experience]): the in-order trajectory that this
        experience is part of
    - index (int): the index inside of this trajectory that this experience
        is.
  """
  def __init__(self, experience, trajectory, index):
    self._experience = experience
    self._trajectory = trajectory
    self._index = index

  def __getattr__(self, attr):
    if attr[0] == "_":
      raise AttributeError("accessing private attribute '{}'".format(attr))
    return getattr(self._experience, attr)

  @property
  def trajectory(self):
    return self._trajectory

  @property
  def index(self):
    return self._index


class RewardLabeler(abc.ABC):
  """Computes rewards for trajectories on the fly."""

  @abc.abstractmethod
  def label_rewards(self, trajectories):
    """Computes rewards for each experience in the trajectory.

    Args:
      trajectories (list[list[TrajectoryExperience]]): batch of
          trajectories.

    Returns:
      rewards (torch.FloatTensor): of shape (batch_size, max_seq_len) where
        rewards[i][j] is the rewards for the experience trajectories[i][j].
        This is padded with zeros and is detached from the graph.
      distances (torch.FloatTensor): of shape (batch_size, max_seq_len + 1)
        equal to ||f(e) - g(\tau^e_{:t})|| for each t.
    """
