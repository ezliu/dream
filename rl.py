import collections
import numpy as np


class Experience(collections.namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done", "info",
                   "agent_state", "next_agent_state"))):
  """Defines a single (s, a, r, s')-tuple.

  Includes the agent state, as well for any agents with hidden state.
  """
