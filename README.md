# Dreamer

Dreamer is a visual Model-Based Reinforcement algorithm, that learns a world model which captures latent dynamics from high-level pixel images and trains a control agent entirely in imagined rollouts from the learned world model.

This work is my attempt at reproducing Dreamerv1 & v2 papers in pytorch specifically for continuous control tasks in deepmind control suite.

#### Differences in implementation:

 1. Replay Buffer manages episodes instead of transitions, making sure that we don't mix episodes when sampling
 2. Although less flexible, Convolution models where layed out step by step for readibility

#### Simple implementation of the Dreamer agent

<img src="https://github.com/user-attachments/assets/cb809c2f-135c-4c96-9dde-50c3e16e4fb6" width="150">
<img src="https://github.com/user-attachments/assets/da6e924c-45b4-4cb8-a3bc-619c4cc54663" width="150">
<img src="https://github.com/user-attachments/assets/3e69087a-0e17-478f-b187-cf16dd227ad8" width="150">
<img src="https://github.com/user-attachments/assets/0f440da0-066f-40f6-9787-c7b72988e379" width="150">

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
