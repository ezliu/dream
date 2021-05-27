# Decoupling Exploration and Exploitation in Meta-Reinforcement Learning without Sacrifices
## Introduction

*Authors*: [Evan Zheran Liu](https://cs.stanford.edu/~evanliu/), [Aditi Raghunathan](https://stanford.edu/~aditir/), [Percy Liang](https://cs.stanford.edu/~pliang/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/)

Source code accompanying our ICML 2021 [paper](https://arxiv.org/abs/2008.02790).
Also see our [project web page](https://ezliu.github.io/dream/).

## Requirements

This code requires Python3.
The Python3 requirements are specified in `requirements.txt`.
We recommend creating a `virtualenv`, e.g.:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

To train a meta-RL policy, invoke the following command:

```
python3 main.py exp_name -b environment=\"benchmark\"
```

This will create a directory `experiments/exp_name`, which will contain:

- A tensorboard subdirectory at `experiments/exp_name/tensorboard`, which logs
  statistics, such as accumulated returns vs. number of training episodes, and
  also vs. number of training steps.
- A visualization subdirectory at `experiments/exp_name/visualize`, which will
  contain videos of the learned agent.
- A checkpoints subdirectory at `experiments/exp_name/checkpoints`, which will
  periodically save model checkpoints.
- Metadata about the run, such as the configs used.

The `benchmark` argument specifies which of the benchmarks from the paper to use.
The supported benchmarks are:
- The sparse-reward 3D visual navigation benchmark: `miniworld_sign`
- The cooking benchmark: `cooking`
- The distracting bus benchmark: `distraction`
- The map benchmark: `map`

Below, we provide the commands to reproduce the results from the paper.
Each block of commands trains DREAM, E-RL^2, IMPORT, and VariBAD respectively,
in the specified benchmark.

### Sparse-Reward 3D Visual Navigation:

```
python3 main.py dream -b environment=\"miniworld_sign\" -c configs/default.json -c configs/miniworld.json
python3 main_varibad.py e-rl2 -b environment=\"miniworld_sign\" -c configs/rl2.json -c configs/rl2-miniworld.json
python3 main_varibad.py import -b environment=\"miniworld_sign\" -c configs/import.json -c configs/import-miniworld.json
python3 main_varibad.py varibad -b environment=\"miniworld_sign\" -c configs/varibad.json -c configs/varibad-miniworld.json
```

NOTE: Running [MiniWorld](https://github.com/maximecb/gym-miniworld) headless typically requires `xvfb-run`.
To do this, the NVIDIA GPU drivers must also be compiled with the `--no-opengl-files` flag.
See the [MiniWorld Troubleshooting Guide](https://github.com/maximecb/gym-miniworld/blob/master/docs/troubleshooting.md) for more details.
Commands for headless running below:

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 main.py dream -b environment=\"miniworld_sign\" -c configs/default.json -c configs/miniworld.json
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 main_varibad.py e-rl2 -b environment=\"miniworld_sign\" -c configs/rl2.json -c configs/rl2-miniworld.json
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 main_varibad.py import -b environment=\"miniworld_sign\" -c configs/import.json -c configs/import-miniworld.json
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 main_varibad.py varibad -b environment=\"miniworld_sign\" -c configs/varibad.json -c configs/varibad-miniworld.json
```

### Distracting Bus:

```
python3 main.py dream -b environment=\"distraction\" -c configs/default.json
python3 main_varibad.py e-rl2 -b environment=\"distraction\" -c configs/rl2.json
python3 main_varibad.py import -b environment=\"distraction\" -c configs/import.json
python3 main_varibad.py varibad -b environment=\"distraction\" -c configs/varibad.json
```

### Map:

```
python3 main.py dream -b environment=\"map\" -c configs/default.json
python3 main_varibad.py e-rl2 -b environment=\"map\" -c configs/rl2.json
python3 main_varibad.py import -b environment=\"map\" -c configs/import.json
python3 main_varibad.py varibad -b environment=\"map\" -c configs/varibad.json
```

### Cooking:

```
python3 main.py dream -b environment=\"cooking\" -c configs/default.json
python3 main_varibad.py e-rl2 -b environment=\"cooking\" -c configs/rl2.json
python3 main_varibad.py import -b environment=\"cooking\" -c configs/import.json
python3 main_varibad.py varibad -b environment=\"cooking\" -c configs/varibad.json
```

# Citation

If you use this code, please cite our paper.

```
@article{liu2020decoupling,
  title={Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices},
  author={Liu, Evan Zheran and Raghunathan, Aditi and Liang, Percy and Finn, Chelsea},
  journal={arXiv preprint arXiv:2008.02790},
  year={2020}
}
```
