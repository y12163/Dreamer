# Dreamer

Dreamer is a visual Model-Based Reinforcement algorithm, that learns a world model which captures latent dynamics from high-level pixel images and trains a control agent entirely in imagined rollouts from the learned world model.

This work is my attempt at reproducing Dreamerv1 & v2 papers in pytorch specifically for continuous control tasks in deepmind control suite.

#### Differences in implementation:

 1. Replay Buffer manages episodes instead of transitions, making sure that we don't mix episodes when sampling
 2. Although less flexible, Convolution models where layed out step by step for readibility


## Code Structure
Code structure is similar to original work by Danijar Hafner in Tensorflow

`dreamer.py`  - main function for training and evaluating dreamer agent

`utils.py`    - Logger, miscellaneous utility functions

`models.py`   - All the NNs for world model and actor

`replay_buffer.py` - Experience buffer for training world model

`env_wrapper.py`  - Gym wrapper for Dm_control suite

Runs can be configured from the config.json

## Installation

Run:
`conda env create -f environment.yml`

#### For training
`python dreamer.py --config config.json --env <env-name> --train`
#### For Evaluation
`python dreamer.py --config config.json --evaluate --env <env-name> --checkpoint_path '<your_ckpt_path>'`



## Acknowledgements
This code is heavily inspired by following open-source works

dreamer by Danijar Hafner : https://github.com/danijar/dreamer

dreamer-pytorch by yusukeurakami : https://github.com/yusukeurakami/dreamer-pytorch

Dreamerv2 by Rajghugare : https://github.com/RajGhugare19/dreamerv2

Dreamer by adityabingi : https://github.com/adityabingi/Dreamer
