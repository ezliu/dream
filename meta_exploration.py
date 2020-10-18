import abc
import collections

import gym
import numpy as np
import torch

import rl


class MetaExplorationState(collections.namedtuple(
    "MetaExplorationState",
    ("observation", "prev_reward", "prev_action", "env_id"))):
  """Consists of:

    - observation (object): the state s.
    - prev_action (int): the action that was played in the previous timestep or
          None on the first timestep.
    - env_id (int): the dynamics e.
  """
  pass


class MetaExplorationEnv(abc.ABC, gym.Env):
  """Defines an environment with varying dynamics identified by an env_id e.

  Subclass observation spaces are expected to be gym.spaces.Dict with two keys:
  observation and env_id.
  """

  def __init__(self, env_id, wrapper):
    self._env_id = env_id
    self._wrapper = wrapper

  @classmethod
  def create_env(cls, seed, test=False, wrapper=None):
    """Randomly creates an environment instance.

    Args:
      seed (int): used to randomly select.
      test (bool): determines whether to make an environment from the train or
        test split.
      wrapper (function): gets called on observations. Defaults to converting
        observations to torch.tensor. Pass lambda state: state to receive numpy
        observations.

    Returns:
      MetaExplorationEnv
    """
    if wrapper is None:
      wrapper = lambda state: torch.tensor(state)

    random = np.random.RandomState(seed)
    train_ids, test_ids = cls.env_ids()
    split = test_ids if test else train_ids
    env_id = split[random.randint(len(split))]
    return cls(env_id, wrapper)

  @abc.abstractmethod
  def env_ids(cls):
    """Returns the list of valid task IDs split into train / test.

    Returns:
      train_ids (list[int]): ids used in create_env(test=False).
      test_ids (list[int]): ids used in create_env(test=True).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def instruction_wrapper(cls):
    """Class method returning the typical InstructionWrapper to use.

    Returns:
      InstructionWrapper type: InstructionWrapper subclass to wrap instances of
        this class with.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _step(self, action):
    raise NotImplementedError()

  def step(self, action):
    state, reward, done, info = self._step(action)
    state = MetaExplorationState(
        self._wrapper(state), reward, action, self._env_id)
    return state, reward, done, info

  @abc.abstractmethod
  def _reset(self):
    raise NotImplementedError()

  def reset(self):
    state = self._reset()
    state = MetaExplorationState(self._wrapper(state), 0, None, self._env_id)
    return state

  @property
  def env_id(self):
    return self._env_id


class InstructionState(collections.namedtuple(
    "InstructionState",
    ("observation", "instructions", "prev_action", "prev_reward", "done",
     "trajectory", "env_id"))):
  """Consists of:

    - observation (object): see MetaExplorationState.
    - instructions (object): instructions that define the reward function i.
    - prev_action (int | None): see MetaExplorationState
    - prev_reward (float | None): the reward in the previous timestep, or None
        on the first timestep.
    - done (bool): True if the episode ended.
    - trajectory (list[rl.Experience]): the exploration trajectory \tau^e.
    - env_id (int): see MetaExplorationState.
  """
  pass


class InstructionWrapper(abc.ABC, gym.Wrapper):
  """Environment for generating instructions and trajectories."""

  def __init__(self, env, exploration_trajectory, seed=None,
               test=False, first_episode_no_instruction=False,
               first_episode_no_optimization=False,
               fixed_instructions=False):
    """Sets all states to contain the given exploration_trajectory.

    Args:
      env (MetaExplorationEnv): environment to wrap.
      exploration_trajectory (list[rl.Experience]): the exploration trajectory
        \tau^e to wrap states with.
      seed (int | None): used to randomly generate instructions.
      test (bool): returns test split (if exists) of instructions if True
      first_episode_no_instruction (bool): if True, the first episode with this
        wrapper returns a sentinel instruction with no rewards. First episode
        is equivalent to wrapping with no rewards.
      first_episode_no_optimization (bool): if True, rewards from the first
        episode are added to the state, but not returned in step, for the use
        case where the first episode rewards are observed but not optimized.
      fixed_instructions (bool): if True, the same instructions are used
        throughout all episodes.
    """
    super().__init__(env)
    self._trajectory = exploration_trajectory
    self._current_instructions = None
    self.observation_space = gym.spaces.Dict({
      "observation": env.observation_space["observation"],
      "env_id": env.observation_space["env_id"],
      "instructions": self._instruction_observation_space(),
      # Trajectory currently not included in observation space
    })
    self._random = np.random.RandomState(seed)
    self._test = test
    self._first_episode_no_instruction = first_episode_no_instruction
    self._first_episode_no_optimization = first_episode_no_optimization
    self._fixed_instructions = fixed_instructions
    self._num_episodes = 0

  @abc.abstractmethod
  def _generate_instructions(self, test=False):
    """Generates a new set of instructions.

    Returns:
      instructions (object): instructions to use.
      test (bool): True during test time.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _instruction_observation_space(cls):
    """Returns the observation space of _generate_instructions (spaces.Space)"""
    raise NotImplementedError()

  @abc.abstractmethod
  def _reward(self, instruction_state, action, original_reward):
    """Defines the reward function r_i(s, a) and episode termination.

    Args:
      instruction_state (InstructionState): state at this timestep.
      action (object): action taken during this timestep.
      original_reward (float): the reward in the wrapped env.

    Returns:
      reward (float): r_i(s, a).
      done (bool): True if the instructions have been completed.
    """
    raise NotImplementedError()

  @property
  def current_instructions(self):
    """Returns the instructions being used on the current episode."""
    return self._current_instructions

  @property
  def random(self):
    """Random state for generating instructions."""
    return self._random

  def _sentinel_instructions(self):
    """The instructions to return on the first episode, if the
    first_episode_no_instruction flag is set."""
    return np.array(self.observation_space["instructions"].low)

  def reset(self):
    self._num_episodes += 1
    state = super().reset()

    if self._num_episodes == 1 and self._first_episode_no_instruction:
      self._current_instructions = self._sentinel_instructions()
    else:
      if not (self._num_episodes > 1 and self._fixed_instructions):
        self._current_instructions = self._generate_instructions(
            test=self._test)

    return InstructionState(
        state.observation, self._current_instructions, None, 0, False,
        self._trajectory, state.env_id)

  def step(self, action):
    state, original_reward, done, info = super().step(action)
    state = InstructionState(
        state.observation, self._current_instructions, action, None, False,
        self._trajectory, state.env_id)

    if self._num_episodes == 1 and self._first_episode_no_instruction:
      reward, instructions_complete = 0, False
    else:
      reward, instructions_complete = self._reward(
          state, action, original_reward)

    done = instructions_complete or done
    state = state._replace(prev_reward=reward, done=done)

    if self._num_episodes == 1 and self._first_episode_no_optimization:
      reward = 0
    return state, reward, done, info
