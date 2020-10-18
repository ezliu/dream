import collections
import math
import gym
import gym_miniworld
from gym_miniworld import entity
from gym_miniworld import miniworld
from gym_miniworld import random
import torch
import numpy as np
from PIL import Image

import meta_exploration
import render


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """InstructionWrapper for SignEnv."""

  def __init__(self, env, exploration_trajectory, **kwargs):
    super().__init__(env, exploration_trajectory, **kwargs)
    self._step = 0

  def _instruction_observation_space(self):
    return gym.spaces.Box(np.array([0]), np.array([2]), dtype=np.int)

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    # TODO(evzliu): Probably should more gracefully grab the base_env
    reward = 0
    done = False

    for obj_index, object_pair in enumerate(self.env._base_env._objects):
      for color_index, obj in enumerate(object_pair):
        if self.env._base_env.near(obj):
          done = True
          reward = float(color_index == self.env.env_id and
                         obj_index == instruction_state.instructions[0]) * 2 - 1
    return reward, done

  def _generate_instructions(self, test=False):
    return self.random.randint(2, size=(1,))

  def render(self):
    image = self.env.render()
    image.write_text("Instructions: {}".format(self._current_instructions))
    return image


class BigKey(entity.Key):
  """A key with a bigger size."""

  def __init__(self, color, size=0.6):
    assert color in entity.COLOR_NAMES
    entity.MeshEnt.__init__(
        self,
        mesh_name='key_{}'.format(color),
        height=size,
        static=False
    )


class SignEnv(miniworld.MiniWorldEnv):
  """The sign environment from IMPORT.

  Touching either the red or blue box ends the episode.
  If the box corresponding to the color_index is touched, reward = +1
  If the wrong box is touched, reward = -1.

  The sign behind the wall either says "blue" or "red"
  """

  def __init__(self, size=10, max_episode_steps=20, color_index=0):
    params = gym_miniworld.params.DEFAULT_PARAMS.no_random()
    params.set('forward_step', 0.7)
    params.set('turn_step', 45)  # 45 degree rotation

    self._size = size
    self._color_index = color_index

    super().__init__(
        params=params, max_episode_steps=max_episode_steps, domain_rand=False)

    # Allow for left / right / forward + custom end episode
    self.action_space = gym.spaces.Discrete(self.actions.move_forward + 2)

  def set_color_index(self, color_index):
    self._color_index = color_index

  def _gen_world(self):
    gap_size = 0.25
    top_room = self.add_rect_room(
        min_x=0, max_x=self._size, min_z=0, max_z=self._size * 0.65)
    left_room = self.add_rect_room(
        min_x=0, max_x=self._size * 3 / 5, min_z=self._size * 0.65 + gap_size,
        max_z=self._size * 1.3)
    right_room = self.add_rect_room(
        min_x=self._size * 3 / 5, max_x=self._size,
        min_z=self._size * 0.65 + gap_size, max_z=self._size * 1.3)
    self.connect_rooms(top_room, left_room, min_x=0, max_x=self._size * 3 / 5)
    self.connect_rooms(
        left_room, right_room, min_z=self._size * 0.65 + gap_size,
        max_z=self._size * 1.3)

    self._objects = [
        # Boxes
        (self.place_entity(
            gym_miniworld.entity.Box(color="blue"), pos=(1, 0, 1)),
         self.place_entity(
            gym_miniworld.entity.Box(color="red"), pos=(9, 0, 1)),
         self.place_entity(
            gym_miniworld.entity.Box(color="green"), pos=(9, 0, 5)),
         ),

        # Keys
        (self.place_entity(BigKey(color="blue"), pos=(5, 0, 1)),
         self.place_entity(BigKey(color="red"), pos=(1, 0, 5)),
         self.place_entity(BigKey(color="green"), pos=(1, 0, 9))),
    ]

    text = ["BLUE", "RED", "GREEN"][self._color_index]
    sign = gym_miniworld.entity.TextFrame(
        pos=[self._size, 1.35, self._size + gap_size],
        dir=math.pi,
        str=text,
        height=1,
    )
    self.entities.append(sign)
    self.place_agent(min_x=4, max_x=5, min_z=4, max_z=6)

  def step(self, action):
    obs, reward, done, info = super().step(action)
    if action == self.actions.move_forward + 1:  # custom end episode action
      done = True

    for object_pair in self._objects:
      for obj in object_pair:
        if self.near(obj):
          done = True
    return obs, reward, done, info


# From:
# https://github.com/maximecb/gym-miniworld/blob/master/pytorch-a2c-ppo-acktr/envs.py
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class MiniWorldSign(meta_exploration.MetaExplorationEnv):
  """Wrapper around the gym-miniworld Maze conforming to the MetaExplorationEnv
  interface.
  """
  def __init__(self, env_id, wrapper):
    super().__init__(env_id, wrapper)
    self._base_env = SignEnv()
    self._env = TransposeImage(self._base_env)
    self.observation_space = gym.spaces.Dict({
      "observation": self._env.observation_space,
      "env_id": gym.spaces.Box(np.array([0]), np.array([3]), dtype=np.int)
    })
    self.action_space = self._env.action_space

  @classmethod
  def instruction_wrapper(cls):
    return InstructionWrapper

  # Grab instance of env and modify it, to prevent creating many envs, which
  # causes memory issues.
  @classmethod
  def create_env(cls, seed, test=False, wrapper=None):
    if wrapper is None:
      wrapper = lambda state: torch.tensor(state)

    random = np.random.RandomState(seed)
    train_ids, test_ids = cls.env_ids()
    to_sample = test_ids if test else train_ids
    env_id = to_sample[random.randint(len(to_sample))]
    sign_instance._env_id = env_id
    sign_instance._wrapper = wrapper
    return sign_instance

  def _step(self, action):
    return self._env.step(action)

  def _reset(self):
    # Don't set the seed, otherwise can cheat from initial camera angle position!
    self._env.set_color_index(self.env_id)
    return self._env.reset()

  @classmethod
  def env_ids(cls):
    return list(range(3)), list(range(3))

  def render(self):
    first_person_render = self._base_env.render(mode="rgb_array")
    top_render = self._base_env.render(mode="rgb_array", view="top")
    image = render.concatenate(
        [Image.fromarray(first_person_render), Image.fromarray(top_render)],
        "horizontal")
    image.thumbnail((320, 240))
    image = render.Render(image)
    image.write_text("Env ID: {}".format(self.env_id))
    return image


# Prevents from opening too many windows.
sign_instance = MiniWorldSign(0, None)
