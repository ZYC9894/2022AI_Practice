# Homework

## Installation
You need to  install two RL environments: OpenAI `Gym` 、OpenAI `Gym Atari`.

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run:
training policy gradient:
* `python main.py --train_pg`

training DQN:
* `python main.py --train_dqn`

## How to evaluate performance your algorithm:
PS: You need to load model which has best performance before testing the algorithm

testing policy gradient:

* `python test.py --test_pg`
  

testing DQN:
* `python test.py --test_dqn`
