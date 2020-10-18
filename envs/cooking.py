import gym
import numpy as np

from envs import grid
import meta_exploration


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """Instruction wrapper for CookingGridEnv.

  Provides instructions (goal bowl contents) and their corresponding rewards.

  Reward function for a given goal is:
      R(s, a) = -0.1 if s != goal
              = 1    otherwise
  """

  def __init__(self, cooking_env, exploration_trajectory, **kwargs):
    super().__init__(cooking_env, exploration_trajectory, **kwargs)
    self._step = 0

  def _instruction_observation_space(self):
    return gym.spaces.Box(
        np.array([0] * 2), np.array([self.num_ingredients + 1] * 2), dtype=np.int)

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    done = False
    reward = -0.1
    if self._step == 0:
      if self.inventory == instruction_state.instructions[0]:
        self._step += 1
        reward = 0.25
      elif action == grid.Action.pickup and self.inventory != instruction_state.instructions[0]:
        reward -= 0.25
    elif self._step == 1:
      if np.array_equal(
          self.bowl_contents[:1], instruction_state.instructions[:1]):
        self._step += 1
        reward = 0.25
      elif self.inventory != instruction_state.instructions[0]:
        self._step -= 1
        reward -= 0.25
    elif self._step == 2:
      if self.inventory == instruction_state.instructions[1]:
        self._step += 1
        reward = 0.25
      elif action == grid.Action.pickup and self.inventory != instruction_state.instructions[1]:
        reward -= 0.25
    elif self._step == 3:
      if np.array_equal(
          self.bowl_contents[:2], instruction_state.instructions[:2]):
        reward = 0.25
        done = True
      elif self.inventory != instruction_state.instructions[1]:
        self._step -= 1
        reward -= 0.25

    if action == grid.Action.end_episode:
      reward -= self.steps_remaining * 0.1  # penalize ending the episode
    return reward, done

  def reset(self):
    state = super().reset()
    self._step = 0
    return state

  def _generate_instructions(self, test=False):
    return list(self.random.choice(self.drawer_contents, size=(2,)))

  def render(self, mode="human"):
    image = super().render(mode)
    image.write_text("Instructions: {}".format(self.current_instructions))

    for i, contents in enumerate(self.current_instructions):
      if contents < self.num_ingredients:
        color = self.ingredient_colors[contents]
        image.draw_inventory(-i - 1, color)
    return image

  def __str__(self):
    s = super().__str__()
    s += "\nInstructions: {}".format(self.current_instructions)
    return s


class CookingGridEnv(grid.GridEnv):
  """Defines a kitchen grid with drawers and a bowl at fixed locations.

  Ingredients can be picked up out of drawers and dropped.
  The env_id representation is: [ingredient 1, ingredient 2, ingredient 3]
    - where ingredient_i is a number [0, num_ingredients] denoting the
      ingredient inside of the i-th drawer.

  Different env_ids correspond to different ingredients in the drawers.

  The state representation is:
    [x, y, inventory, status, bowl_ingredient_1, bowl_ingredient_2,
        bowl_ingredient_3]
    - x is the agent's x coordinate
    - y is the agent's y coordinate
    - status is a number [0, 3], where:
        - 1 means the agent is at a bus stop
        - 2 means that the agent is at a drawer
        - 3 means that the the agent is at the bowl
        - 0 means that the agent is at none of the above
    - inventory is a number [0, num_ingredients] of the object current held.
        num_ingredients denotes that the inventory is empty.
    - bowl_ingredient_i is the i-th ingredient in the bowl [0, num_ingredients].
        num_ingredients denotes that the i-th ingredient is empty.
  """
  _num_ingredients = 4
  _drawers = [np.array((2, 0)), np.array((2, 1)), np.array((2, 2))]
  _bowl_pos = np.array((1, 1))
  _train_ids = None
  _test_ids = None
  _ingredient_colors = [
      "rgb(0, 0, 255)", "rgb(128, 128, 0)", "rgb(255, 0, 128)",
      "rgb(128, 128, 255)", "rgb(128, 255, 0)", "rgb(255, 0, 128)",
      "rgb(0, 128, 0)", "rgb(128, 255, 255)", "rgb(255, 128, 128)",
      "rgb(255, 255, 0)",
  ]

  def __init__(self, env_id, wrapper, max_steps=20):
    super().__init__(env_id, wrapper, max_steps=max_steps, width=3, height=3)
    self._drawer_contents = None

  @classmethod
  def instruction_wrapper(cls):
    return InstructionWrapper

  def _observation_space(self):
    low, high, dtype = super()._observation_space()
    # inventory + drawers
    low = np.concatenate((low, [0] + [0] * len(self._drawers)))
    high = np.concatenate(
        (high, [self.num_ingredients + 1] * (len(self._drawers) + 1)))
    return low, high, dtype

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([self.num_ingredients ** len(self._drawers)])
    dtype = np.int
    return low, high, dtype

  @property
  def drawer_contents(self):
    if self._drawer_contents is None:
      self._drawer_contents = []
      env_id = self.env_id[0]
      for i in range(len(self._drawers)):
        self._drawer_contents.append(env_id % self.num_ingredients)
        env_id = env_id // self.num_ingredients
    return self._drawer_contents

  @property
  def num_ingredients(self):
    return self._num_ingredients

  @property
  def ingredient_colors(self):
    return self._ingredient_colors

  @classmethod
  def env_ids(cls):
    if cls._train_ids is None:
      # (num_ids, id_dim)
      ids = np.arange(cls._num_ingredients ** len(cls._drawers)).astype(np.int64).reshape(-1, 1)
      test_ids = np.array([10])
      train_ids = np.array(list(set(range(ids.shape[0])) - set(test_ids)))
      cls._test_ids = ids[test_ids]
      cls._train_ids = ids[train_ids]
    return cls._train_ids, cls._test_ids

  @property
  def bowl_contents(self):
    bowl = self.get(self._bowl_pos)
    return bowl.contents(empty=self.num_ingredients)

  def text_description(self):
    return "kitchen grid"

  def _gen_obs(self):
    obs = super()._gen_obs()
    inventory = (self._inventory if self._inventory is not None
                 else self.num_ingredients)
    return np.concatenate((obs, [inventory] + self.bowl_contents))

  def _place_objects(self):
    """After grid size is determined, place any objects into the grid."""
    super()._place_objects()
    self.place(grid.Bowl("black"), self._bowl_pos)
    for i, drawer_pos in enumerate(self._drawers):
      ingredient = self.drawer_contents[i]
      color = self._ingredient_colors[ingredient]
      self.place(grid.Drawer(color, ingredient), drawer_pos)

  def render(self, mode="human"):
    image = super().render(mode=mode)

    for i, contents in enumerate(self.bowl_contents):
      if contents < self._num_ingredients:
        color = self._ingredient_colors[contents]
        image.draw_inventory(-i - 4, color)  # reserve last bits for instruction

    if self._inventory is not None:
      image.draw_inventory(0, self._ingredient_colors[self._inventory])
    return image
