import abc
from PIL import Image, ImageDraw


class Render(abc.ABC):
  """Convenience object to return from env.render.

  Allows for annotated text on a banner on top and exporting to PIL.
  """

  def __init__(self, main_image):
    """Constructs from the main PIL image to render.

    Args:
      main_image (PIL.Image): the main thing to render.
    """
    self._main_image = main_image
    self._banner = Image.new(
        mode="RGBA", size=(main_image.width, 150), color="white")
    self._text = []

  def write_text(self, text):
    """Appends new line of text to any previous text.

    Args:
      text (str): text to display on top of rendering.
    """
    self._text.append(text)

  def image(self):
    """Returns a PIL.Image representation of this rendering."""
    draw = ImageDraw.Draw(self._banner)
    draw.text((0, 0), "\n".join(self._text), (0, 0, 0))
    return concatenate([self._banner, self._main_image], "vertical")

  def __deepcopy__(self, memo):
    cls = self.__class__
    deepcopy = cls.__new__(cls)
    memo[id(self)] = deepcopy

    # PIL doesn't support deepcopy directly
    image_copy = self.__dict__["_main_image"].copy()
    banner_copy = self.__dict__["_banner"].copy()
    setattr(deepcopy, "_main_image", image_copy)
    setattr(deepcopy, "_banner", banner_copy)

    for k, v in self.__dict__.items():
      if k not in ["_main_image", "_banner"]:
        setattr(deepcopy, k, copy.deepcopy(v, memo))
    return deepcopy


def concatenate(images, mode="horizontal"):
  assert mode in ["horizontal", "vertical"]

  if mode == "horizontal":
    new_width = sum(img.width for img in images)
    new_height = max(img.height for img in images)
  else:
    new_width = max(img.width for img in images)
    new_height = sum(img.height for img in images)

  final_image = Image.new(mode="RGBA", size=(new_width, new_height))
  curr_width, curr_height = (0, 0)
  for img in images:
    final_image.paste(img, (curr_width, curr_height))

    if mode == "horizontal":
      curr_width += img.width
    else:
      curr_height += img.height
  return final_image
