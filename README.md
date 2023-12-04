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

## Training
* For training the `Feedforward` model (default) - `python main.py`
* For training the  `GPT` (gpt-nano)  model - `python main.py -m gpt`


## Additional Config Parameters:

Following new hyperparameters are added in the training configuration for GPT  model training (`get_gpt_config`) and given Feedforward model training (`get_ff_config`):

  * `C.trainer.patience` - *Number of interations to wait before validation training does not show improvement and training stops*
  * `C.trainer.validation_interval` - *Number of steps after which we run validation*
  * `C.trainer.min_relative_improvement` - *Threshold for relative improvement in validation loss, such that we consider an improvement in validation loss. Here 5% improvement is set in decimals as 0.05*
