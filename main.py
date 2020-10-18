import argparse
import collections
import os
import shutil

import git
import numpy as np
import torch
import tqdm

import config as cfg
import dqn
from envs import grid
from envs import cooking
from envs import city
import policy
import relabel
import rl
import utils


def run_episode(env, policy, experience_observers=None, test=False):
  """Runs a single episode on the environment following the policy.

  Args:
    env (gym.Environment): environment to run on.
    policy (Policy): policy to follow.
    experience_observers (list[Callable] | None): each observer is called with
      with each experience at each timestep.

  Returns:
    episode (list[Experience]): experiences from the episode.
    renders (list[object | None]): renderings of the episode, only rendered if
      test=True. Otherwise, returns list of Nones.
  """
  # Optimization: rendering takes a lot of time.
  def maybe_render(env, action, reward, timestep):
    if test:
      render = env.render()
      render.write_text("Action: {}".format(str(action)))
      render.write_text("Reward: {}".format(reward))
      render.write_text("Timestep: {}".format(timestep))
      return render
    return None

  if experience_observers is None:
    experience_observers = []

  episode = []
  state = env.reset()
  timestep = 0
  renders = [maybe_render(env, None, 0, timestep)]
  hidden_state = None
  while True:
    action, next_hidden_state = policy.act(
        state, hidden_state, test=test)
    next_state, reward, done, info = env.step(action)
    timestep += 1
    renders.append(
        maybe_render(env, grid.Action(action), reward, timestep))
    experience = rl.Experience(
        state, action, reward, next_state, done, info, hidden_state,
        next_hidden_state)
    episode.append(experience)
    for observer in experience_observers:
      observer(experience)

    state = next_state
    hidden_state = next_hidden_state
    if done:
      return episode, renders


def get_env_class(environment_type):
  """Returns the environment class specified by the type.

  Args:
    environment_type (str): a valid environment type.

  Returns:
    environment_class (type): type specified.
  """
  if environment_type == "vanilla":
    return city.CityGridEnv
  elif environment_type == "distraction":
    return city.DistractionGridEnv
  elif environment_type == "map":
    return city.MapGridEnv
  elif environment_type == "cooking":
    return cooking.CookingGridEnv
  elif environment_type == "miniworld_sign":
    # Dependencies on OpenGL, so only load if absolutely necessary
    from envs.miniworld import sign
    return sign.MiniWorldSign
  else:
    raise ValueError(
        "Unsupported environment type: {}".format(environment_type))


def get_instruction_agent(instruction_config, instruction_env):
  if instruction_config.get("type") == "learned":
    return dqn.DQNAgent.from_config(instruction_config, instruction_env)
  else:
    raise ValueError(
        "Invalid instruction agent: {}".format(instruction_config.get("type")))


def get_exploration_agent(exploration_config, exploration_env):
  if exploration_config.get("type") == "learned":
    return dqn.DQNAgent.from_config(exploration_config, exploration_env)
  elif exploration_config.get("type") == "random":
    return policy.RandomPolicy(exploration_env.action_space)
  elif exploration_config.get("type") == "none":
    return policy.ConstantActionPolicy(grid.Action.end_episode)
  else:
    raise ValueError("Invalid exploration agent: {}".format(
      exploration_config.get("type")))


def log_episode(exploration_episode, exploration_rewards, distances, path):
  with open(path, "w+") as f:
    f.write("Env ID: {}\n".format(exploration_episode[0].state.env_id))
    for t, (exp, exploration_reward, distance) in enumerate(
        zip(exploration_episode, exploration_rewards, distances)):
      f.write("=" * 80 + "\n")
      f.write("Timestep: {}\n".format(t))
      f.write("State: {}\n".format(exp.state.observation))
      f.write("Action: {}\n".format(grid.Action(exp.action).name))
      f.write("Reward: {}\n".format(exploration_reward))
      f.write("Distance: {}\n".format(distance))
      f.write("Next state: {}\n".format(exp.next_state.observation))
      f.write("=" * 80 + "\n")
      f.write("\n")


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
      '-c', '--configs', action='append', default=["configs/default.json"])
  arg_parser.add_argument(
      '-b', '--config_bindings', action='append', default=[],
      help="bindings to overwrite in the configs.")
  arg_parser.add_argument(
      "-x", "--base_dir", default="experiments",
      help="directory to log experiments")
  arg_parser.add_argument(
      "-p", "--checkpoint", default=None,
      help="path to checkpoint directory to load from or None")
  arg_parser.add_argument(
      "-f", "--force_overwrite", action="store_true",
      help="Overwrites experiment under this experiment name, if it exists.")
  arg_parser.add_argument(
      "-s", "--seed", default=0, help="random seed to use.", type=int)
  arg_parser.add_argument("exp_name", help="name of the experiment to run")
  args = arg_parser.parse_args()
  config = cfg.Config.from_files_and_bindings(
      args.configs, args.config_bindings)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  exp_dir = os.path.join(os.path.expanduser(args.base_dir), args.exp_name)
  if os.path.exists(exp_dir) and not args.force_overwrite:
    raise ValueError("Experiment already exists at: {}".format(exp_dir))
  shutil.rmtree(exp_dir, ignore_errors=True)  # remove directory if exists
  os.makedirs(exp_dir)

  with open(os.path.join(exp_dir, "config.json"), "w+") as f:
    config.to_file(f)
  print(config)

  env_class = get_env_class(config.get("environment"))

  with open(os.path.join(exp_dir, "metadata.txt"), "w+") as f:
    repo = git.Repo()
    f.write("Commit: {}\n\n".format(repo.head.commit))
    commit = repo.head.commit
    diff = commit.diff(None, create_patch=True)
    for patch in diff:
      f.write(str(patch))
      f.write("\n\n")
    f.write("Split: {}\n".format(env_class.env_ids()))


  # Use GPU if possible
  device = torch.device("cpu")
  if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")

  print("Device: {}".format(device))
  tb_writer = utils.EpisodeAndStepWriter(os.path.join(exp_dir, "tensorboard"))

  text_dir = os.path.join(exp_dir, "text")
  os.makedirs(text_dir)

  checkpoint_dir = os.path.join(exp_dir, "checkpoints")
  os.makedirs(checkpoint_dir)

  create_env = env_class.create_env
  exploration_env = create_env(0)
  instruction_env = env_class.instruction_wrapper()(exploration_env, [])
  instruction_config = config.get("instruction_agent")
  instruction_agent = get_instruction_agent(instruction_config, instruction_env)

  exploration_config = config.get("exploration_agent")
  exploration_agent = get_exploration_agent(exploration_config, exploration_env)

  # Should probably expose this more gracefully
  trajectory_embedder = (
      instruction_agent._dqn._Q._state_embedder._trajectory_embedder)
  exploration_agent.set_reward_relabeler(trajectory_embedder)

  # Due to the above hack, the trajectory embedder is being loaded twice.
  if args.checkpoint is not None:
    print("Loading checkpoint: {}".format(args.checkpoint))
    instruction_agent.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "instruction.pt")))
    exploration_agent.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "exploration.pt")))

  batch_size = 32
  rewards = collections.deque(maxlen=200)
  relabel_rewards = collections.deque(maxlen=200)
  exploration_lengths = collections.deque(maxlen=200)
  exploration_steps = 0
  instruction_steps = 0
  for step in tqdm.tqdm(range(1000000)):
    exploration_env = create_env(step)
    exploration_episode, _ = run_episode(
        # Exploration episode gets ignored
        env_class.instruction_wrapper()(
            exploration_env, [], seed=max(0, step - 1)),
        exploration_agent)

    # Needed to keep references to the trajectory and index for reward labeling
    for index, exp in enumerate(exploration_episode):
      exploration_agent.update(
          relabel.TrajectoryExperience(exp, exploration_episode, index))

    exploration_steps += len(exploration_episode)
    exploration_lengths.append(len(exploration_episode))

    # Don't share same random seed between exploration env and instructions
    instruction_env = env_class.instruction_wrapper()(
        exploration_env, exploration_episode, seed=step + 1)

    if step % 2 == 0:
      trajectory_embedder.use_ids(False)
    episode, _ = run_episode(
        instruction_env, instruction_agent,
        experience_observers=[instruction_agent.update])
    instruction_steps += len(episode)
    trajectory_embedder.use_ids(True)

    rewards.append(sum(exp.reward for exp in episode))

    # Log reward for exploration agent
    exploration_rewards, distances = trajectory_embedder.label_rewards(
        [exploration_episode])
    exploration_rewards = exploration_rewards[0]
    distances = distances[0]
    relabel_rewards.append(exploration_rewards.sum().item())

    if step % 100 == 0:
      path = os.path.join(text_dir, "{}.txt".format(step))
      log_episode(exploration_episode, exploration_rewards, distances, path)

    if step % 100 == 0:
      for k, v in instruction_agent.stats.items():
        if v is not None:
          tb_writer.add_scalar(
              "{}_{}".format("instruction", k), v, step,
              exploration_steps + instruction_steps)

      for k, v in exploration_agent.stats.items():
        if v is not None:
          tb_writer.add_scalar(
              "{}_{}".format("exploration", k), v, step,
              exploration_steps + instruction_steps)

      tb_writer.add_scalar(
          "steps/exploration", exploration_steps, step,
          exploration_steps + instruction_steps)
      tb_writer.add_scalar(
          "steps/instruction", instruction_steps, step,
          exploration_steps + instruction_steps)
      tb_writer.add_scalar(
          "reward/train", np.mean(rewards), step,
          exploration_steps + instruction_steps)
      tb_writer.add_scalar(
          "reward/exploration", np.mean(relabel_rewards), step,
          exploration_steps + instruction_steps)
      tb_writer.add_scalar(
          "steps/exploration_per_episode", np.mean(exploration_lengths),
          step, exploration_steps + instruction_steps)

    if step % 2000 == 0:
      visualize_dir = os.path.join(exp_dir, "visualize", str(step))
      os.makedirs(visualize_dir, exist_ok=True)

      test_rewards = []
      test_exploration_lengths = []
      trajectory_embedder.use_ids(False)
      for test_index in tqdm.tqdm(range(100)):
        exploration_env = create_env(test_index, test=True)
        exploration_episode, exploration_render = run_episode(
            env_class.instruction_wrapper()(
                exploration_env, [], seed=max(0, test_index - 1), test=True),
            exploration_agent, test=True)
        test_exploration_lengths.append(len(exploration_episode))

        instruction_env = env_class.instruction_wrapper()(
            exploration_env, exploration_episode, seed=test_index + 1, test=True)
        episode, render = run_episode(
            instruction_env, instruction_agent, test=True)
        test_rewards.append(sum(exp.reward for exp in episode))

        if test_index < 10:
          frames = [frame.image() for frame in render]
          save_path = os.path.join(
              visualize_dir, "{}-instruction.gif".format(test_index))
          frames[0].save(save_path, save_all=True, append_images=frames[1:],
                         duration=750, loop=0, optimize=True, quality=20)

          frames = [frame.image() for frame in exploration_render]
          save_path = os.path.join(
              visualize_dir, "{}-exploration.gif".format(test_index))
          frames[0].save(save_path, save_all=True, append_images=frames[1:],
                         duration=750, loop=0, optimize=True, quality=20)

      tb_writer.add_scalar(
          "reward/test", np.mean(test_rewards), step,
          exploration_steps + instruction_steps)
      tb_writer.add_scalar(
          "steps/test_exploration_per_episode",
          np.mean(test_exploration_lengths), step,
          exploration_steps + instruction_steps)

      # Visualize training split
      visualize_dir = os.path.join(exp_dir, "visualize", str(step), "train")
      os.makedirs(visualize_dir, exist_ok=True)
      for train_index in tqdm.tqdm(range(10)):
        exploration_env = create_env(train_index)
        # Test flags here only refer to making agent act with test flag and
        # not test split environments
        exploration_episode, exploration_render = run_episode(
            env_class.instruction_wrapper()(
                exploration_env, [], seed=max(0, train_index - 1)),
            exploration_agent, test=True)

        instruction_env = env_class.instruction_wrapper()(
            exploration_env, exploration_episode, seed=train_index + 1)
        episode, render = run_episode(
            instruction_env, instruction_agent, test=True)

        frames = [frame.image() for frame in render]
        save_path = os.path.join(
            visualize_dir, "{}-instruction.gif".format(train_index))
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                       duration=750, loop=0)

        frames = [frame.image() for frame in exploration_render]
        save_path = os.path.join(
            visualize_dir, "{}-exploration.gif".format(train_index))
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                       duration=750, loop=0)
      trajectory_embedder.use_ids(True)

      if exploration_steps + instruction_steps > int(5e6):
        return

    if step != 0 and step % 20000 == 0:
      print("Saving checkpoint")
      save_dir = os.path.join(checkpoint_dir, str(step))
      os.makedirs(save_dir)

      torch.save(instruction_agent.state_dict(),
                 os.path.join(save_dir, "instruction.pt"))
      torch.save(exploration_agent.state_dict(),
                 os.path.join(save_dir, "exploration.pt"))


if __name__ == '__main__':
  main()
