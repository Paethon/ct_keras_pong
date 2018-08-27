# ct_keras_pong
An AI that plays Atari 2600 Pong. Trained using reinforcement learning using OpenAI Gym and Keras

# Usage
## Train the agent
Run `python train_agent.py weightfile.h5` to train an agent. If
`weightfile.h5` already exists, the training will continue from the
last point, otherwise a new training run will be started and the
progress saved to a new weights file.

If you do not want to show the game screen to speed up training, set
the `render` variable in the script to `False`

## Use the agent

You can execute a game in real time with a weights file using `python
run_agent.py ` To run the agent with the trained weights provided run
`python run_agent.py trained_weights.h5`

# Results to Expect
[[https://github.com/Paethon/ct_keras_pong/media/learning_progress.png]]

# Acknowledgments
The code is in part base on https://github.com/mkturkcan/Keras-Pong

