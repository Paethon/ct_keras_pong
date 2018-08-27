import sys
import gym
import numpy as np
import keras
from keras.layers import Dense
from keras.layers.core import Flatten
from time import sleep

# Ensure the correct dimension ordering
keras.backend.set_image_data_format('channels_last')

# Script Parameters
number_of_actions = 6


def preprocess(image):
    # Cut out playing field and subsample by 2
    image = image[35:195:2, ::2, 0]
    # Set background to "black"
    image[np.logical_or(image == 144, image == 109)] = 0
    # Set paddles and ball to "white"
    image[image != 0] = 1
    return image.astype(np.float)


# Initialize OpenAI Gym environment
env = gym.make("Pong-v0")
obs = env.reset()
previous_obs = preprocess(obs)

# Create the policy network in Keras
model = keras.models.Sequential()
model.add(Flatten(input_shape=((80, 80, 1))))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(Dense(number_of_actions, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# Load weights
model.load_weights(sys.argv[1])

while True:
    env.render()

    current_obs = preprocess(obs)
    state = current_obs - previous_obs  # This has been changed
    previous_obs = current_obs

    # Predict probabilities from the Keras model
    action_prob = model.predict_on_batch(state.reshape(1, 80, 80, 1))[0, :]
    action = np.random.choice(number_of_actions, p=action_prob)

    # Execute one action in the environment
    obs, reward, done, info = env.step(action)
    sleep(1/60)  # Let game run in something akin to realtime
    # One game is finished, so reset environment
    if done:
        obs = env.reset()
