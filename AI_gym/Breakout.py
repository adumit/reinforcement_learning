import gym
import tensorflow as tf
from brains.Deep_Q import shallow_Q
from utils.history import *

env = gym.make("Breakout-v0")

network = shallow_Q(input_dims=[None, 210, 160, 3], exploration_const=.9, num_actions=4)
history = History()
discount_factor = .99


def play_and_record_game(watch_game):
    screens = []
    actions = []
    rewards = []

    done = False
    observation = env.reset()
    num_lives = 5
    while not done:
        if watch_game:
            env.render()
        action = network.act(observation)
        observation, reward, done, info = env.step(action)

        if info['ale.lives'] < num_lives:
            reward -= 1

        screens.append(observation)
        actions.append(action)
        rewards.append(reward)

        num_lives = info['ale.lives']

    rewards.reverse()
    return_rewards = []
    running_reward = 0
    for r in rewards:
        # At each time step, starting from the last, we get our current reward plus the discount factor multiplied
        # out to each 'future' reward.
        running_reward += r + discount_factor * running_reward
        return_rewards.append(running_reward)

    # So the return rewards match up with the appropriate screens
    return_rewards.reverse()
    for i in range(len(return_rewards)):
        history.add(screens[i], actions[i], return_rewards[i])


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for epoch in range(100):
        while history.num_obs < 5000:
            play_and_record_game(False)
        for screens, actions, rewards in history.get_mini_batches(200):
            network.train(sess, screens, actions, rewards)

        history.empty()
    _ = input("Press enter to watch a game")
    play_and_record_game(True)
