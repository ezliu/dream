import abc
import enum

import gym
import numpy as np
from PIL import Image, ImageDraw

import meta_exploration

# Unfortunately, this light-weight grid world turns out to duplicate MiniGrid a
# lot.
class Action(enum.IntEnum):
  left = 0
  up = 1
  right = 2
  down = 3
  noop = 4
  ride_bus = 5
  end_episode = 6
  pickup = 7
  drop = 8


class GridObject(abc.ABC):
  """An object that can be placed in the GridEnv.

  Subclasses should register themselves in the list of GridObject above."""

  def __init__(self, color, size=0.4):
    """Constructs.

    Args:
      color (str): valid PIL color.
    """
    self._color = color
    self._size = size

  @property
  def status(self):
    """Returns a unique int for each GridObject subclass."""
    return all_objects.index(self.__class__) + 1

  @abc.abstractmethod
  def toggle(self, agent_pos):
    """Changes the grid based on what toggling the object does.

    Args:
      agent_pos (np.array): agent position while toggling (the position of the
        object).

    Returns:
      agent_pos (np.array): the new agent position after toggling.
    """

  def pickup(self):
    """Returns the agent's new inventory after the pickup action.

    Returns:
      object | None: None if there's nothing to pickup.
    """
    return None

  # Currently allows you to drop anything.
  def drop(self, inventory):
    """Updates the state after dropping the inventory on this object.

    Args:
      inventory (object | None): the agent's current inventory

    Returns:
      success (bool): if False, the drop failed, and the agent's inventory
        should not change. Otherwise, the agent's inventory becomes None.
    """
    return True

  @property
  def color(self):
    return self._color

  @property
  def size(self):
    return self._size


class Bus(GridObject):
  """Toggling a bus changes the agent's position to the destination."""

  def __init__(self, color, destination):
    """Constructs.

    Args:
      color (str): see parent class.
      destination (np.array): where toggling this bus stop leads to.
    """
    super().__init__(color)
    self._destination = destination

  def toggle(self, agent_pos):
    return np.array(self._destination)


class Bowl(GridObject):
  """Holds several ingredients (int)."""

  def __init__(self, color, capacity=3):
    super().__init__(color)
    self._contents = [None] * capacity

  def toggle(self, agent_pos):
    return agent_pos

  def drop(self, inventory):
    try:
      index = self._contents.index(None)
      self._contents[index] = inventory
    except ValueError:  # bowl is full
      pass

    # Drop currently always succeeds
    return True

  def contents(self, empty=None):
    """Returns the contents of the bowl.

    Args:
      empty (object | None): empty parts of the bowl are represented with this.

    Returns:
      list[object | None]: the contents of the bowl.
    """
    return [content if content is not None else empty
            for content in self._contents]


class Drawer(GridObject):
  """Holds ingredients (int)."""

  def __init__(self, color, ingredient):
    """Constructs.

    Args:
      color (str): see parent class.
      ingredient (int): the ingredient returned from the pickup action.
    """
    super().__init__(color)
    self._ingredient = ingredient

  def toggle(self, agent_pos):
    return agent_pos

  def pickup(self):
    return self._ingredient


# Subclasses of GridObject should register themselves here.
all_objects = [Bus, Bowl, Drawer]


class GridEnv(meta_exploration.MetaExplorationEnv, abc.ABC):
  """A grid world to move around in.

  Default observations are: (x, y, status)
    - (x, y) are the coordinates of the agent
    - status is a unique int for each type of object that the agent may be
        standing on.
  """

  def __init__(self, env_id, wrapper, max_steps=20, width=10, height=10):
    """Constructs the environment with dynamics according to env_id.

    Args:
      env_id (int): a valid env_id in TransportationGridEnv.env_ids()
      wrapper (function): see create_env.
      max_steps (int): maximum horizon of a single episode.
    """
    super().__init__(env_id, wrapper)
    self._max_steps = max_steps
    self._grid = [[None for _ in range(height)] for _ in range(width)]
    self._width = width
    self._height = height
    self._inventory = None

  @abc.abstractmethod
  def instruction_wrapper(cls):
    """Returns the instruction wrapper to use with this env.

    Returns:
      instruction_wrapper (type): type of the instruction wrapper.
    """

  def _observation_space(self):
    """Returns high, low and dtype of observation component of observation
    space.

    Returns:
      low (np.array): lowerbound (inclusive) of observations.
      high (np.array): of same shape as low, upperbound (exclusive) of
        observations.
      dtype (np.dtype): dtype of observations.
    """
    low = np.array([0, 0, 0])
    # 1 for each type of object + 1 for no objects.
    high = np.array([self._width, self._height, len(all_objects) + 1])
    return low, high, np.int

  @abc.abstractmethod
  def _env_id_space(self):
    """Returns high, low and dtype of env_id component of observation space.

    Returns:
      low (np.array): lowerbound (inclusive) of env_id.
      high (np.array): of same shape as low, upperbound (exclusive) of
        env_id.
      dtype (np.dtype): dtype of env_id.
    """

  @property
  def observation_space(self):
    observation_low, observation_high, dtype = self._observation_space()
    env_id_low, env_id_high, dtype = self._env_id_space()
    return gym.spaces.Dict({
        "observation": gym.spaces.Box(
            observation_low, observation_high, dtype=dtype),
        "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=dtype)
    })

  @property
  def action_space(self):
    return gym.spaces.Discrete(len(Action))

  @property
  def width(self):
    """Returns number of columns in the grid (int)."""
    return self._width

  @property
  def height(self):
    """Returns number of rows in the grid (int)."""
    return self._height

  @property
  def inventory(self):
    """Returns the contents of the agent's inventory (object | None)."""
    return self._inventory

  @property
  def agent_pos(self):
    """Returns location of the agent (np.array)."""
    return self._agent_pos

  @property
  def steps_remaining(self):
    """Returns the number of timesteps remaining in the episode (int)."""
    return self._max_steps - self._steps

  def text_description(self):
    return "grid"

  def get(self, position):
    """Returns the object in the grid at the given position.

    Args:
      position (np.array): (x, y) coordinates.

    Returns:
      object | None
    """
    return self._grid[position[0]][position[1]]

  def place(self, obj, position):
    """Places an object in the grid at the given position.

    Args:
      obj (GridObj): object to place into the grid.
      position (np.array): (x, y) coordinates.
    """
    existing_obj = self.get(position)
    if existing_obj is not None:
      raise ValueError(
          "Object {} already exists at {}.".format(existing_obj, position))
    self._grid[position[0]][position[1]] = obj

  def _place_objects(self):
    """After grid size is determined, place any objects into the grid."""
    self._agent_pos = np.array([1, 1])
    self._inventory = None

  def _gen_obs(self):
    """Returns an observation (np.array)."""
    try:
      status = all_objects.index(type(self.get(self.agent_pos))) + 1
    except ValueError:
      status = 0
    return np.concatenate((np.array(self.agent_pos), [status]))

  def _reset(self):
    self._steps = 0
    self._grid = [[None for _ in range(self.height)]
                  for _ in range(self.width)]
    self._place_objects()
    self._history = [np.array(self._agent_pos)]
    return self._gen_obs()

  def _step(self, action):
    self._steps += 1

    obj = self.get(self.agent_pos)
    if action == Action.left:
      self._agent_pos[0] -= 1
    elif action == Action.up:
      self._agent_pos[1] += 1
    elif action == Action.right:
      self._agent_pos[0] += 1
    elif action == Action.down:
      self._agent_pos[1] -= 1
    elif action == Action.noop:
      pass
    elif action == Action.end_episode:
      return self._gen_obs(), 0, True, {}
    elif action == Action.ride_bus:
      if obj is not None:
        self._agent_pos = obj.toggle(self.agent_pos)
    elif action == Action.pickup:
      if obj is not None:
        self._inventory = obj.pickup()
    elif action == Action.drop:
      success = True
      if obj is not None:
        success = obj.drop(self._inventory)

      if success:
        self._inventory = None

    self._agent_pos[0] = max(min(self._agent_pos[0], self.width - 1), 0)
    self._agent_pos[1] = max(min(self._agent_pos[1], self.height - 1), 0)
    done = self._steps == self._max_steps
    self._history.append(np.array(self._agent_pos))
    return self._gen_obs(), 0, done, {}

  def render(self, mode="human"):
    image = GridRender(self.width, self.height)

    image.draw_rectangle(self.agent_pos, 0.6, "red")
    for x, col in enumerate(self._grid):
      for y, obj in enumerate(col):
        if obj is not None:
          image.draw_rectangle(np.array((x, y)), obj.size, obj.color)

    for pos in self._history:
      image.draw_rectangle(pos, 0.2, "orange")

    image.write_text(self.text_description())
    image.write_text("Current state: {}".format(self._gen_obs()))
    image.write_text("Env ID: {}".format(self.env_id))
    return image


class GridRender(object):
  """Human-readable rendering of a GridEnv state."""

  def __init__(self, width, height):
    """Creates a grid visualization with a banner with the text.

    Args:
      width (int): number of rows in the SimpleGridEnv state space.
      height (int): number of columns in the SimpleGridEnv state space.
    """
    self._PIXELS_PER_GRID = 100
    self._width = width
    self._height = height

    self._banner = Image.new(
        mode="RGBA",
        size=(width * self._PIXELS_PER_GRID,
              int(self._PIXELS_PER_GRID * 1.5)),
        color="white")
    self._text = []
    self._inventory = Image.new(
        mode="RGBA",
        size=(width * self._PIXELS_PER_GRID, self._PIXELS_PER_GRID),
        color="white")
    self._inventory_draw = ImageDraw.Draw(self._inventory)
    self._image = Image.new(
        mode="RGBA",
        size=(width * self._PIXELS_PER_GRID,
              height * self._PIXELS_PER_GRID),
        color="white")
    self._draw = ImageDraw.Draw(self._image)
    for col in range(width):
      x = col * self._PIXELS_PER_GRID
      line = ((x, 0), (x, height * self._PIXELS_PER_GRID))
      self._draw.line(line, fill="black")

    for row in range(height):
      y = row * self._PIXELS_PER_GRID
      line = ((0, y), (width * self._PIXELS_PER_GRID, y))
      self._draw.line(line, fill="black")

  def write_text(self, text):
    """Adds a banner with the given text. Appends to any previous text.

    Args:
      text (str): text to display on top of rendering.
    """
    self._text.append(text)

  def draw_rectangle(self, position, size, color):
    """Draws a rectangle at the specified position with the specified size.

    Args:
      position (np.array): (x, y) position corresponding to a valid state.
      size (float): between 0 and 1 corresponding to how large the rectangle
        should be.
      color (Object): valid PIL color for the rectangle.
    """
    start = position * self._PIXELS_PER_GRID + (0.5 - size / 2) * np.array(
        [self._PIXELS_PER_GRID, self._PIXELS_PER_GRID])
    end = position * self._PIXELS_PER_GRID + (0.5 + size / 2) * np.array(
        [self._PIXELS_PER_GRID, self._PIXELS_PER_GRID])
    self._draw.rectangle((tuple(start), tuple(end)), fill=color)

  def draw_inventory(self, position, color):
    """Draws a rectangle at the given position in the inventory.

    Args:
      position (int): numeric position. Draws from the right if negative.
      color (object): valid PIL color.
    """
    rect_width = self._PIXELS_PER_GRID // 6
    if position < 0:
      position = self._inventory.width // (rect_width * 2 + 1) + position

    start = np.array(
        [(position * 2 + 1) * rect_width,
         (self._inventory.height - rect_width) // 2], dtype=np.int)
    end = np.array(
        [((position + 1) * 2) * rect_width,
         (self._inventory.height + rect_width) // 2], dtype=np.int)
    self._inventory_draw.rectangle((tuple(start), tuple(end)), fill=color)

  def __deepcopy__(self, memo):
    cls = self.__class__
    deepcopy = cls.__new__(cls)
    memo[id(self)] = deepcopy

    # PIL doesn't support deepcopy directly
    image_copy = self.__dict__["_image"].copy()
    draw_copy = ImageDraw.Draw(image_copy)
    banner_copy = self.__dict__["_banner"].copy()
    setattr(deepcopy, "_image", image_copy)
    setattr(deepcopy, "_draw", draw_copy)
    setattr(deepcopy, "_banner", banner_copy)

    for k, v in self.__dict__.items():
      if k not in ["_image", "_draw", "_banner"]:
        setattr(deepcopy, k, copy.deepcopy(v, memo))
    return deepcopy

  def image(self):
    """Returns PIL Image representing this render."""
    image = Image.new(
        mode="RGBA",
        size=(
            self._image.width, self._image.height + self._banner.height +
            self._inventory.height))
    draw = ImageDraw.Draw(self._banner)
    draw.text((0, 0), "\n".join(self._text), (0, 0, 0))
    image.paste(self._banner, (0, 0))
    image.paste(self._inventory, (0, self._banner.height))
    image.paste(self._image, (0, self._banner.height + self._inventory.height))
    return image
