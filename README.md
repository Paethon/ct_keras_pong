# ct_keras_pong
An AI that plays Atari 2600 Pong. Trained using reinforcement learning using OpenAI Gym and Keras

# Usage
## Installation
You need to have OpenAI Gym and Keras installed

`pip install gym[all] Keras`

## Train the agent
Run `python train_agent.py weightfile.h5` to train an agent. If
`weightfile.h5` already exists, the training will continue from the
last point, otherwise a new training run will be started and the
progress saved to a new weights file.

If you do not want to show the game screen to speed up training, set
the `render` variable in the script to `False`

### Provided Untrained Weights
The repository also provides `start_weights.h5` containing the randomly
initalized weights that were used to train the `trained_weights.h5`. The
method can be very sensitive to the initial weights and it might happen,
that your agent does not learn. For those cases make a copy of
`start_weights.h5` to have the same starting point I had during training.

## Use the agent

You can execute a game in real time with a weights file using `python
run_agent.py ` To run the agent with the trained weights provided run
`python run_agent.py trained_weights.h5`

# Results to Expect
| Untrained                         | Trained                       |
|:---------------------------------:|:-----------------------------:|
| ![Untrained](media/untrained.gif) | ![Trained](media/trained.gif) |

## Learning Progress

![Learing Progress](media/learning_progress.png)

# Acknowledgments
The code is in part base on https://github.com/mkturkcan/Keras-Pong

