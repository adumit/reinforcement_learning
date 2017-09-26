import tensorflow as tf
import random


def multiply_list(in_list):
    out = 1
    for entry in in_list:
        out *= int(entry)
    return out


def conv2D(name, input_tensor, weight_shape, strides):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name+'_weights', shape=weight_shape)
        conv = tf.nn.conv2d(input_tensor, weights, strides, padding='SAME', name=name+"_conv")
        biases = tf.get_variable(name+"_biases", shape=[weight_shape[3]])
        pre_activation = tf.nn.bias_add(conv, biases, name=name+"_preact")
        conv_out = tf.nn.relu(pre_activation, name=name+"_out")

        return conv_out


def FC(name, input_tensor, out_dim):
    with tf.variable_scope(name) as scope:
        reshape = tf.reshape(input_tensor, shape=[tf.shape(input_tensor)[0], -1], name=name+"_reshape")
        weights = tf.get_variable(name+"_weights",
                                  shape=[multiply_list(input_tensor.shape[1:]), out_dim])
        biases = tf.get_variable(name+'_biases', shape=out_dim)
        pre_activation = tf.nn.bias_add(tf.matmul(reshape, weights), biases, name=name+"_preact")
        fc_out = tf.nn.relu(pre_activation, name=name+"_out")

        return fc_out


class shallow_Q:
    def __init__(self, input_dims, resize_dims, final_dims, num_actions):
        self.num_actions = num_actions

        self.input = tf.placeholder(dtype=tf.float32, shape=input_dims)
        self.reshaped = tf.image.resize_images(self.input, size=resize_dims, method=tf.image.ResizeMethod.BILINEAR)
        self.cropped = tf.image.resize_image_with_crop_or_pad(self.reshaped, target_height=final_dims[0], target_width=final_dims[1])
        self.conv1 = conv2D("conv1", input_tensor=self.cropped, weight_shape=[8, 8, 3, 32], strides=[1, 4, 4, 1])
        print("CONV1 shape:", self.conv1.shape)
        self.conv2 = conv2D("conv2", input_tensor=self.conv1, weight_shape=[4, 4, 32, 64], strides=[1, 2, 2, 1])
        print("CONV2 shape:", self.conv2.shape)
        self.fc1 = FC("fc1", self.conv2, out_dim=128)
        print("FC1 shape:", self.fc1)
        self.fc2 = FC("fc2", self.fc1, out_dim=256)
        print("FC2 shape:", self.fc2)
        self.q = FC("output", self.fc2, out_dim=num_actions)

        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.action = tf.placeholder('int64', [None], name='action')
        self.action_one_hot = tf.one_hot(self.action, num_actions, 1.0, 0.0, name='action_one_hot')
        self.q_acted = tf.reduce_sum(self.q * self.action_one_hot, reduction_indices=1, name='q_acted')

        self.delta = self.target_q_t - self.q_acted

        self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.delta)), name='loss')
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.q_action = tf.argmax(self.q, dimension=1)

    def act(self, input_env, exploration_const):
        epsilon = random.random()
        if epsilon < exploration_const:
            action = int(self.q_action.eval({self.input: [input_env]}))
        else:
            action = random.randint(0, self.num_actions - 1)
        return action

    def train(self, sess, train_screens, train_actions, train_rewards):
        sess.run([self.optimizer], feed_dict={
            self.input: train_screens,
            self.action: train_actions,
            self.target_q_t: train_rewards})

