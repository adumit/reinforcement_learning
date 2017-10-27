import random
from numpy import array_split


class History:
    def __init__(self):
        self._screen_history = []
        self._action_history = []
        self._reward_history = []
        self.num_obs = 0

    def add(self, screen, action, reward):
        self._screen_history.append(screen)
        self._action_history.append(action)
        self._reward_history.append(reward)
        self.num_obs += 1

    def get_mini_batches(self, mini_batch_size):
        zipped = list(zip(self._screen_history, self._action_history, self._reward_history))
        random.shuffle(zipped)
        screens, actions, rewards = zip(*zipped)

        if len(screens) % mini_batch_size != 0:
            num_to_append = mini_batch_size - (len(screens) % mini_batch_size)
            screens += screens[:num_to_append]
            actions += actions[:num_to_append]
            rewards += rewards[:num_to_append]

        num_batches = int(len(screens) / mini_batch_size)
        for i in range(num_batches):
            yield (screens[i*mini_batch_size:(i+1)*mini_batch_size],
                   actions[i*mini_batch_size:(i+1)*mini_batch_size],
                   rewards[i*mini_batch_size:(i+1)*mini_batch_size])

    def empty(self):
        self._action_history = []
        self._screen_history = []
        self._reward_history = []
        self.num_obs = 0
