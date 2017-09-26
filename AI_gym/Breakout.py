import gym
import tensorflow as tf
import numpy as np
from brains.Deep_Q import shallow_Q
from utils.history import *
from datetime import datetime

env = gym.make("Breakout-v0")

network = shallow_Q(input_dims=[None, 210, 160, 3], resize_dims=[110, 84], final_dims=[84, 84], num_actions=4)
history = History()
discount_factor = .99


def play_and_record_game(watch_game, exploration_const):
    game_screens = []
    game_actions = []
    game_rewards = []

    done = False
    observation = env.reset()
    num_lives = 5
    while not done:
        if watch_game:
            env.render()
        action = network.act(observation, exploration_const)
        observation, reward, done, info = env.step(action)

        if info['ale.lives'] < num_lives:
            reward -= 1

        game_screens.append(observation)
        game_actions.append(action)
        game_rewards.append(reward)

        num_lives = info['ale.lives']

    game_rewards.reverse()
    return_rewards = []
    running_reward = 0
    for r in game_rewards:
        # At each time step, starting from the last, we get our current reward plus the discount factor multiplied
        # out to each 'future' reward.
        running_reward = r + discount_factor * running_reward
        return_rewards.append(running_reward)

    # So the return rewards match up with the appropriate screens
    return_rewards.reverse()
    for i in range(len(return_rewards)):
        history.add(game_screens[i], game_actions[i], return_rewards[i])


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for epoch in range(20):
        start = datetime.now()
        while history.num_obs < 200000:
            if epoch < 5:
                # Explore much more in earlier epochs. This equation anneals the epsilon value from 0.1 to 0.9 over the
                # first million frames.
                epsilon = 0.9 - (0.8 - 0.8/5.0*epoch) * (1000000 - 200000*epoch - history.num_obs)/1000000
                play_and_record_game(False, epsilon)
            else:
                play_and_record_game(False, 0.9)
            if max(history._reward_history) > 100 or min(history._reward_history) < -100:
                print(history.num_obs)
                print(epoch)
        num_batches = 0
        for screens, actions, rewards in history.get_mini_batches(100):
            num_batches += 1
            if num_batches % 50 == 0:
                print(np.mean(rewards))
            network.train(sess, screens, actions, rewards)
        print("Epoch {0} took {1} seconds.".format(epoch, (datetime.now() - start).seconds))

        history.empty()

    user_choice = input("Press enter to watch a game")
    while user_choice != 'done':
        user_choice = input("Press enter to watch a game")
        play_and_record_game(True, 1.0)
