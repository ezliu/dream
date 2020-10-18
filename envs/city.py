import itertools

import gym
import numpy as np

from envs import grid
import meta_exploration


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """Instruction wrapper for CityGridEnv.

  Provides instructions (goal locations) and their corresponding rewards.

  Reward function for a given goal is:
      R(s, a) = -0.1 if s != goal
              = 1    otherwise
  """

  def _instruction_observation_space(self):
    return gym.spaces.Box(
        np.array([0, 0]), np.array([self.width, self.height]), dtype=np.int)

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    done = False
    reward = -0.1
    if np.array_equal(self.agent_pos, instruction_state.instructions):
      reward = 1
      done = True
    elif action == grid.Action.end_episode:
      reward -= self.steps_remaining * 0.1  # penalize ending the episode
    return reward, done

  def _generate_instructions(self, test=False):
    del test

    goals = [np.array((0, 0)), np.array((8, 8)),
             np.array((0, 8)), np.array((8, 0))]
    goal = goals[self._random.randint(len(goals))]
    return goal

  def render(self, mode="human"):
    image = super().render(mode)
    image.draw_rectangle(self.current_instructions, 0.5, "green")
    image.write_text("Instructions: {}".format(self.current_instructions))
    return image

  def __str__(self):
    s = super().__str__()
    s += "\nInstructions: {}".format(self.current_instructions)
    return s


class CityGridEnv(grid.GridEnv):
  """Defines a city grid with bus stops at fixed locations.

  Upon toggling a bus stop, the agent is teleported to the next bus stop.
  - The environment defines no reward function (see InstructionWrapper for
  rewards).
  - The episode ends after a fixed number of steps.
  - Different env_ids correspond to different bus destination permutations.
  """

  # Location of the bus stops and the color to render them
  _bus_sources = [
      (np.array((4, 5)), "rgb(0,0,255)"),
      (np.array((5, 4)), "rgb(255,0,255)"),
      (np.array((3, 4)), "rgb(255,255,0)"),
      (np.array((4, 3)), "rgb(0,255,255)"),
  ]

  _destinations = [
      np.array((0, 1)), np.array((0, 7)), np.array((8, 1)), np.array((8, 7)),
  ]

  _bus_permutations = list(itertools.permutations(_destinations))

  _height = 9
  _width = 9

  # Optimization: Full set of train / test IDs is large, so only compute it
  # once. Even though individual IDs are small, the whole ID matrix cannot be
  # freed if we have a reference to a single ID.
  _train_ids = None
  _test_ids = None

  def __init__(self, env_id, wrapper, max_steps=20):
    super().__init__(env_id, wrapper, max_steps=max_steps, width=self._width,
                     height=self._height)

  @classmethod
  def instruction_wrapper(cls):
    return InstructionWrapper

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([len(self._bus_permutations)])
    dtype = np.int
    return low, high, dtype

  @classmethod
  def env_ids(cls):
    ids = np.expand_dims(np.array(range(len(cls._bus_permutations))), 1)
    return np.array(ids), np.array(ids)

  def text_description(self):
    return "bus grid"

  def _place_objects(self):
    super()._place_objects()
    self._agent_pos = np.array([4, 4])

    destinations = self._bus_permutations[
        self.env_id[0] % len(self._bus_permutations)]
    for (bus_stop, color), dest in zip(self._bus_sources, destinations):
      self.place(grid.Bus(color, dest), bus_stop)
      self.place(grid.Bus(color, bus_stop), dest)


# TODO(evzliu): Could make the map an object
class DistractionGridEnv(CityGridEnv):
  """Has additional bus stops that go to random places, but are not useful
  for any intructions.
  """

  # Useless bus stops that randomly go to one of the useless destinations
  _useless_bus_sources = [
      np.array((5, 5)), np.array((3, 3)), np.array((3, 5)), np.array((5, 3)),
  ]

  # Color for useless buses
  _useless_color = "rgb(128, 128, 0)"

  _useless_destinations = [
      np.array((0, 4)), np.array((4, 0)), np.array((8, 4)), np.array((4, 8)),
  ]

  _useless_permutations = list(itertools.permutations(_useless_destinations))

  # Optimization: Full set of train / test IDs is large, so only compute it
  # once. Even though individual IDs are small, the whole ID matrix cannot be
  # freed if we have a reference to a single ID.
  _train_ids = None
  _test_ids = None

  def _env_id_space(self):
    low, high, dtype = super()._env_id_space()
    high = high * len(self._useless_permutations)
    return low, high, dtype

  @classmethod
  def env_ids(cls):
    if cls._train_ids is None:
      ids = np.expand_dims(np.array(
          range(len(cls._bus_permutations) * len(cls._useless_permutations))), 1)
      train_indices = np.array(range(0, ids.shape[0], 11))
      test_indices = np.array(
              list(set(range(ids.shape[0])) - set(train_indices)))
      cls._train_ids = ids[train_indices]
      cls._test_ids = ids[test_indices]
    return cls._train_ids, cls._test_ids

  def text_description(self):
    return "distraction grid"

  def _place_objects(self):
    super()._place_objects()

    env_id = self.env_id
    destinations = self._useless_permutations[
        self.env_id[0] // len(self._bus_permutations)]
    for bus_stop, dest in zip(self._useless_bus_sources, destinations):
      self.place(grid.Bus(self._useless_color, dest), bus_stop)
      self.place(grid.Bus(self._useless_color, bus_stop), dest)


class MapGridEnv(CityGridEnv):
  """Includes a map that tells the bus orientations."""

  def _observation_space(self):
    low, high, dtype = super()._observation_space()
    # add dim for map
    env_id_low, env_id_high, _ = self._env_id_space()

    low = np.concatenate((low, [env_id_low[0]]))
    high = np.concatenate((high, [env_id_high[0] + 1]))
    return low, high, dtype

  def text_description(self):
    return "map grid"

  def _place_objects(self):
    super()._place_objects()
    self._map_pos = np.array([7, 3])

  def _gen_obs(self):
    obs = super()._gen_obs()
    map_info = [0]
    if np.array_equal(self.agent_pos, self._map_pos):
      map_info = [self.env_id[0] + 1]
    return np.concatenate((obs, map_info), 0)

  def render(self, mode="human"):
    image = super().render(mode=mode)
    image.draw_rectangle(self._map_pos, 0.4, "black")
    return image
