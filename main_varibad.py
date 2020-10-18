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
import main as main_script
import rl
import relabel
import utils
import wrappers


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
      '-c', '--configs', action='append', default=["configs/rl2.json"])
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

  env_class = main_script.get_env_class(config.get("environment"))

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
  multi_episode_env = wrappers.MultiEpisodeWrapper(instruction_env)
  agent = dqn.DQNAgent.from_config(
      config.get("agent"), multi_episode_env)

  if args.checkpoint is not None:
    print("Loading checkpoint: {}".format(args.checkpoint))
    agent.load_state_dict(torch.load(os.path.join(args.checkpoint, "agent.pt")))

  batch_size = 32
  rewards = collections.deque(maxlen=200)
  episode_lengths = collections.deque(maxlen=200)
  total_steps = 0
  for episode_num in tqdm.tqdm(range(1000000)):
    exploration_env = create_env(episode_num)
    instruction_env = env_class.instruction_wrapper()(
        exploration_env, [], seed=episode_num + 1,
        first_episode_no_optimization=True)
    multi_episode_env = wrappers.MultiEpisodeWrapper(instruction_env, 2)

    # Switch between IDs and not IDs for methods that use IDs
    # Otherwise, no-op
    if episode_num % 2 == 0:
      if hasattr(agent._dqn._Q._state_embedder, "use_ids"):
        agent._dqn._Q._state_embedder.use_ids(True)

    episode, _ = main_script.run_episode(multi_episode_env, agent)

    for index, exp in enumerate(episode):
      agent.update(relabel.TrajectoryExperience(exp, episode, index))

    if hasattr(agent._dqn._Q._state_embedder, "use_ids"):
      agent._dqn._Q._state_embedder.use_ids(False)

    total_steps += len(episode)
    episode_lengths.append(len(episode))
    rewards.append(sum(exp.reward for exp in episode))

    if episode_num % 100 == 0:
      for k, v in agent.stats.items():
        if v is not None:
          tb_writer.add_scalar(
              "{}_{}".format("agent", k), v, episode_num, total_steps)

      tb_writer.add_scalar("steps/total", total_steps, episode_num, total_steps)
      tb_writer.add_scalar(
          "reward/train", np.mean(rewards), episode_num, total_steps)
      tb_writer.add_scalar(
          "steps/steps_per_episode", np.mean(episode_lengths), episode_num,
          total_steps)

    if episode_num % 2000 == 0:
      visualize_dir = os.path.join(exp_dir, "visualize", str(episode_num))
      os.makedirs(visualize_dir, exist_ok=True)

      test_rewards = []
      test_episode_lengths = []
      for test_index in tqdm.tqdm(range(100)):
        exploration_env = create_env(test_index, test=True)
        instruction_env = env_class.instruction_wrapper()(
            exploration_env, [], seed=test_index + 1, test=True,
            first_episode_no_optimization=True)
        multi_episode_env = wrappers.MultiEpisodeWrapper(instruction_env, 2)
        episode, render = main_script.run_episode(
            multi_episode_env, agent, test=True)
        test_episode_lengths.append(len(episode))

        test_rewards.append(sum(exp.reward for exp in episode))

        if test_index < 10:
          frames = [frame.image() for frame in render]
          save_path = os.path.join(visualize_dir, "{}.gif".format(test_index))
          frames[0].save(save_path, save_all=True, append_images=frames[1:],
                         duration=750, loop=0)

      tb_writer.add_scalar(
          "reward/test", np.mean(test_rewards), episode_num, total_steps)
      tb_writer.add_scalar(
          "steps/test_steps_per_episode", np.mean(test_episode_lengths),
          episode_num, total_steps)

      # Visualize training split
      visualize_dir = os.path.join(
          exp_dir, "visualize", str(episode_num), "train")
      os.makedirs(visualize_dir, exist_ok=True)
      for train_index in tqdm.tqdm(range(10)):
        exploration_env = create_env(train_index)
        # Test flags here only refer to making agent act with test flag and
        # not test split environments
        instruction_env = env_class.instruction_wrapper()(
            exploration_env, [], seed=train_index + 1,
            first_episode_no_optimization=True)
        multi_episode_env = wrappers.MultiEpisodeWrapper(instruction_env, 2)
        episode, render = main_script.run_episode(
            multi_episode_env, agent, test=True)

        frames = [frame.image() for frame in render]
        save_path = os.path.join(visualize_dir, "{}.gif".format(train_index))
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                       duration=750, loop=0)

      if total_steps > int(5e6):
        return

    if episode_num != 0 and episode_num % 20000 == 0:
      print("Saving checkpoint")
      save_dir = os.path.join(checkpoint_dir, str(episode_num))
      os.makedirs(save_dir)

      torch.save(agent.state_dict(), os.path.join(save_dir, "agent.pt"))


if __name__ == '__main__':
  main()
