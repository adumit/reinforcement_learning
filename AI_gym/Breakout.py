import gym
import tensorflow as tf
import numpy as np
from brains.Deep_Q import shallow_Q
from utils.history import *
from datetime import datetime
from collections import Counter
import os
import re

stable_network = shallow_Q('stable', input_dims=[None, 210, 160, 3], resize_dims=[110, 84],
                           final_dims=[84, 84], num_actions=4)
learning_network = shallow_Q('learner', input_dims=[None, 210, 160, 3], resize_dims=[110, 84],
                             final_dims=[84, 84], num_actions=4, variables=stable_network.variables)
history = History()
discount_factor = .99


def play_and_record_game(e, watch_game, exploration_const, print_q_vals=False):
    game_screens = []
    game_actions = []
    game_rewards = []

    done = False
    observation = e.reset()
    observation = observation
    num_lives = 5
    game_length = 0
    while not done:
        game_length += 1
        if watch_game:
            e.render()

        if game_length == 10:
            action = learning_network.act(observation, exploration_const, True)
        else:
            action = learning_network.act(observation, exploration_const)
        observation, reward, done, info = e.step(action)

        game_screens.append(observation)
        game_actions.append(action)
        game_rewards.append(reward)

        if info['ale.lives'] < num_lives:
            done = True

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

    return game_screens, game_actions, return_rewards, game_rewards

restore = True
total_epochs = 10000
games_per_epoch = 10
cur_epochs = 0
regex = re.compile('[0-9]+\.')
train = True

with tf.Session() as sess:
    saver = tf.train.Saver()
    if restore:
        checkpoint_vals = []
        for fname in os.listdir('../checkpoints/'):
            epoch_val = regex.findall(fname)
            if epoch_val:
                checkpoint_vals.append(int(epoch_val[0][:-1]))
        max_val = max(checkpoint_vals)
        saver.restore(sess, "../checkpoints/epochs-{0}".format(max_val))
        cur_epochs = max_val
        print("Loaded checkpoint with {0} epochs".format(max_val))
    else:
        tf.initialize_all_variables().run()

    if train:
        for epoch in range(cur_epochs, total_epochs):
            start = datetime.now()
            epoch_rewards = []
            epoch_actions = []
            print_q_vals = False
            render_game = False
            games_played = 0
            env = gym.envs.make("Breakout-v0")
            while games_played < 10:
                games_played += 1
                if games_played == 9:
                    print_q_vals = True
                    if epoch % 10 == 0:
                        render_game = False
                if float(epoch) < total_epochs/10:
                    # Explore much more in earlier epochs. This equation anneals the epsilon value from 0.1 to 0.9 over
                    # the first million frames.
                    epsilon = 0.9 - (0.8 - 0.8*epoch/total_epochs) *\
                                    (1000000 - games_per_epoch*epoch - history.num_obs)/1000000
                    screens, actions, q_rewards, real_rewards = play_and_record_game(env, render_game, epsilon, print_q_vals)
                else:
                    screens, actions, q_rewards, real_rewards = play_and_record_game(env, render_game, 0.9, print_q_vals)
                epoch_actions += actions
                epoch_rewards.append(np.sum(real_rewards))
                learning_network.train(sess, screens, actions, q_rewards)
            print("Average reward over {2} games on epoch {0} was {1}".format(epoch, np.mean(epoch_rewards), len(epoch_rewards)))
            print("Action distribution for this epoch: {0}".format(Counter(epoch_actions)))
            num_batches = 0
            for screens, actions, q_rewards in history.get_mini_batches(32):
                num_batches += 1
                stable_network.train(sess, screens, actions, q_rewards)
            sess.run(learning_network.assign_params)

            print("Epoch {0} took {1} seconds.".format(epoch, (datetime.now() - start).seconds))

            history.empty()

            if epoch > cur_epochs and epoch % 50 == 0:
                saver.save(sess, '../checkpoints/epochs-{0}'.format(epoch))
                print("SAVED!")

    user_choice = input("Press enter to watch a game")
    while user_choice != 'done':
        env = gym.envs.make("Breakout-v0")
        play_and_record_game(env, True, 0.99)
        user_choice = input("Press enter to watch a game")
