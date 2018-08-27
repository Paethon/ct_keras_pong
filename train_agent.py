import numpy as np
import sys
from os.path import isfile
import gym
import keras
from keras.layers import Dense
from keras.layers.core import Flatten

# Ensure the correct dimension ordering
keras.backend.set_image_data_format('channels_last')

filename = sys.argv[1]

# Script Parameters
learning_rate = 0.001
render = True
number_of_actions = 6


def preprocess(image):
    # Cut out playing field and subsample by 2
    image = image[35:195:2, ::2, 0]
    # Set background to "black"
    image[np.logical_or(image == 144, image == 109)] = 0
    # Set paddles and ball to "white"
    image[image != 0] = 1
    return image.astype(np.float)


def propagate_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in range(r.size - 1, 0, -1):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


states, action_prob_grads, rewards, action_probs = [], [], [], []
reward_sum = 0
episode_number = 0

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
# Load trained weights from the last run if they exist
if isfile(filename):
    model.load_weights(filename)

reward_sums = []  # Used to calculate mean reward sum

filehandler = open('./training.log', 'w')

while True:
    if render:
        env.render()

    current_obs = preprocess(obs)
    state = current_obs - previous_obs
    previous_obs = current_obs

    # Predict probabilities from the Keras model
    action_prob = model.predict_on_batch(state.reshape(1, 80, 80, 1))[0, :]
    action = np.random.choice(number_of_actions, p=action_prob)

    # Execute one action in the environment
    obs, reward, done, info = env.step(action)
    reward_sum += reward

    # Remember what we need for training the model
    states.append(state)
    action_probs.append(action_prob)
    rewards.append(reward)
    # Also remember gradient of the action probabilities
    y = np.zeros(number_of_actions)
    y[action] = 1
    action_prob_grads.append(y - action_prob)

    if done:
        # One game is completed (i.e. one of the players has gotten 21 points)
        # Time to train the policy network
        episode_number += 1

        # Remember last 40 reward sums to calculate mean reward sum
        reward_sums.append(reward_sum)
        if len(reward_sums) > 40:
            reward_sums.pop(0)

        # Print the current performance of the agent and write to log-file
        s = 'Episode %d Total Episode Reward: %f , Mean %f' % (
            episode_number, reward_sum, np.mean(reward_sums))
        print(s)
        filehandler.write(s + '\n')
        filehandler.flush()
            
        # Propagate the rewards back to actions where no reward was given
        # Rewards for earlier actions are attenuated
        rewards = np.vstack(rewards)
        action_prob_grads = np.vstack(action_prob_grads)
        rewards = propagate_rewards(rewards)

        # Accumulate observed states, calculate updated action probabilities
        X = np.vstack(states).reshape(-1, 80, 80, 1)
        Y = action_probs + learning_rate * rewards * action_prob_grads
        # Train policy network
        model.train_on_batch(X, Y)

        # Save current weights of the model
        model.save_weights(filename)
        
        # Reset everything for the next game
        states, action_prob_grads, rewards, action_probs = [], [], [], []
        reward_sum = 0
        obs = env.reset()
