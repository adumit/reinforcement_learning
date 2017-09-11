import gym
import tensorflow as tf
from brains.Deep_Q import shallow_Q

env = gym.make("Breakout-v0")

network = shallow_Q(input_dims=[1, 210, 160, 3], exploration_const=.9, num_actions=4)

print(env.action_space)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for _ in range(10):
        observation = env.reset()
        for __ in range(100):
            env.render()
            action = network.act(observation)
            observation, reward, done, info = env.step(action)
