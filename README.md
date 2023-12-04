# MLC Machine Learning Engineer (NLP) Task

<p align="center">
  <img width="30%" src="LogoGreen_New.png"/>
</p>

NOTE: If you have been assigned this task, _do not_ publicly fork this repo. Instead, follow the instructions in the task doc you have been provided.

This is the repo for the MLC Machine Learning Engineer (NLP) Task, which involves training a small neural network to perform language modelling on an example piece of text.

## Requirements
- Python 3
- Pip

## Installation
- Create a virtual enviornment (recommended name is `venv` so that it is git ignored by default)
- Activate the environment
- Run `pip install -r requirements.txt`

## Refs:
Codebase adapted from minGPT: https://github.com/karpathy/minGPT/

## Task Description

Your task is to make the highest-quality language model you can. You are provided with an
initial (faulty) attempt at a pipeline intended to train a neural network architecture to do
self-supervised character-level language modelling. The provided model is a fully-connected
2-layer neural network with a custom Heaviside activation function, which is set up to be trained
using a 0-1 loss, with 10 characters as input and the subsequent one as the target.
Here is the recommended way to approach this task:

- Understand the relevant bits of the codebase and identify key issues with the current
training pipeline (15 minutes)
- Come up with a plan on how to improve on the provided architecture and pipeline (15
minutes)
- Execute the planned changes (2 hours)
- Write up a report of implementation, findings, and further ideas. Make sure to describe
the trade-offs of the architectural decisions you made. (30 minutes) :white_check_mark:

Bonus points (if you have time):
- A colleague suggests using word embeddings instead of doing character-level
modelling. Describe the trade-offs that change offers. Suggest any alternatives that
come to mind. :white_check_mark:
- The chosen accuracy measure doesn’t provide a good understanding of model
performance. Suggest an alternative measure for model quality and describe the
tradeoff. :white_check_mark:
- The model is currently trained for a set, arbitrary number of iterations. Implement a more
informed stopping criterion. :white_check_mark:

## Initial Code with 0-1 Loss
The initial pipeline with fixed code can be found in branch: `init_debugging`

## Changes
* Added:       
1. Patience for informated stopping criteria
2. GPT model
3. Perplexity as evaluation metric
4. GPT model config added - `get_gpt_config` method in `main.py`

* Updated:
1. Loss Function - Cross Entropy
2. Feedforward model config" - `get_ff_config` method in `main.py`

I would ideally add the list of changes to CHANGELOG.md file, but adding the changes as section to keep things simple for the task.

## Training
* For training the `Feedforward` model (default) - `python main.py`
* For training the  `GPT` (gpt-nano)  model - `python main.py -m gpt`


## Additional Config Parameters:

Following new hyperparameters are added in the training configuration for GPT  model training (`get_gpt_config`) and given Feedforward model training (`get_ff_config`):

  * `C.trainer.patience` - *Number of interations to wait before validation training does not show improvement and training stops*
  * `C.trainer.validation_interval` - *Number of steps after which we run validation*
  * `C.trainer.min_relative_improvement` - *Threshold for relative improvement in validation loss, such that we consider an improvement in validation loss. Here 5% improvement is set in decimals as 0.05*
