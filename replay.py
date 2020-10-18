# Adapted from OpenAI Gym Baselines
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np


class ReplayBuffer(object):
  def __init__(self, size):
    """Create Replay buffer.
    Parameters
    ----------
    size: int
      Max number of transitions to store in the buffer. When the buffer
      overflows the old memories are dropped.
    """
    # use list instead of deque for better random-access performance
    self._storage = []
    self._maxsize = size
    self._next_idx = 0

  @classmethod
  def from_config(cls, config):
    buffer_type = config.get("type")
    if buffer_type == "vanilla":
      return cls(config.get("max_buffer_size"))
    elif buffer_type == "sequential":
      return SequentialReplayBuffer.from_config(config)
    else:
      raise ValueError("Unsupported buffer type: {}".format(buffer_type))

  def __len__(self):
    return len(self._storage)

  def add(self, experience):
    if self._next_idx >= len(self._storage):
      self._storage.append(experience)
    else:
      self._storage[self._next_idx] = experience
    self._next_idx = (self._next_idx + 1) % self._maxsize

  def sample(self, batch_size):
    """Sample a batch of experiences.

    Args:
      batch_size (int): How many transitions to sample.

    Returns:
      list[Experience]: sampled experiences, not necessarily unique
    """
    indices = np.random.randint(len(self._storage), size=batch_size)
    return [self._storage[i] for i in indices]


class SequentialReplayBuffer(ReplayBuffer):
  """Replay buffer that samples length N contiguous sequences.

  Calls to add are assumed to be contiguous experiences.
  """
  def __init__(self, size, sequence_length=10):
    super().__init__(size)

    self._sequence_length = sequence_length
    # True if the previous experience completed the sequence, i.e.,
    # returned done.
    self._first_experience_of_sequence = True

  def add(self, experience):
    if self._first_experience_of_sequence:
      self._first_experience_of_sequence = False
      if self._next_idx >= len(self._storage):
        self._storage.append([])
      self._storage[self._next_idx] = []

    self._storage[self._next_idx].append(experience)
    if experience.done:
      self._first_experience_of_sequence = True
      self._next_idx = (self._next_idx + 1) % self._maxsize

  def sample(self, batch_size):
    """Returns a batch of up-to length N continguous experiences.

    Args:
      batch_size (int): Number of sequences to sample.

    Returns:
      list[list[Experience]]: Sampled sequences, not necessarily unique. The
      outer list is length batch_size, and the inner lists are length <= N,
      where inner sequences are truncated early, if the last experience.done is
      True.
    """
    indices = np.random.randint(len(self._storage), size=batch_size)
    sequences = []
    for index in indices:
      start = 0
      finish = start + self._sequence_length
      sequences.append(self._storage[index][start: finish + 1])
    return sequences

  @classmethod
  def from_config(cls, config):
    return cls(config.get("max_buffer_size"), config.get("sequence_length"))
