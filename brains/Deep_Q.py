import tensorflow as tf
import random
import numpy as np


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
    def __init__(self, name_space, input_dims, resize_dims, final_dims, num_actions, variables=None):
        self.num_actions = num_actions

        with tf.variable_scope(name_space):
            self.input = tf.placeholder(dtype=tf.float32, shape=input_dims)
            self.grayscaled = tf.image.rgb_to_grayscale(self.input)
            print("Grayscaled shape:", self.grayscaled.shape)
            self.cropped = tf.image.crop_to_bounding_box(self.grayscaled, 34, 0, 160, 160)
            print("Cropped shape:", self.cropped.shape)
            self.resized = tf.image.resize_images(self.cropped, final_dims, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            print("Resized shape:", self.resized.shape)
            self.conv1 = conv2D("conv1", input_tensor=self.resized, weight_shape=[8, 8, 1, 32], strides=[1, 4, 4, 1])
            print("CONV1 shape:", self.conv1.shape)
            self.conv2 = conv2D("conv2", input_tensor=self.conv1, weight_shape=[4, 4, 32, 64], strides=[1, 2, 2, 1])
            print("CONV2 shape:", self.conv2.shape)
            self.fc1 = FC("fc1", self.conv2, out_dim=256)
            print("FC1 shape:", self.fc1)
            self.w_out = tf.get_variable('output'+"_weights", shape=[256, num_actions])
            self.b_out = tf.get_variable('output' + '_biases', shape=num_actions)
            self.q = tf.nn.bias_add(tf.matmul(self.fc1, self.w_out), self.b_out, name='output')

            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')
            self.action_one_hot = tf.one_hot(self.action, num_actions, 1.0, 0.0, name='action_one_hot')
            self.q_acted = tf.reduce_sum(self.q * self.action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - self.q_acted
            self.clipped_delta = tf.where(tf.abs(self.delta) < 1.0,
                                          0.5 * tf.square(self.delta),
                                          tf.abs(self.delta) - 0.5, name='clipped_delta')

            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)

            self.q_action = tf.argmax(self.q, dimension=1)

            self.variables = tf.trainable_variables()
            if variables is not None:
                # self.assign_params = [self.variables[i+int(len(self.variables)/2)].assign(self.variables[i]) for
                #                       i in range(int(len(self.variables)/2))]
                self.assign_params = [self.variables[i].assign(variables[i]) for
                                      i in range(int(len(self.variables)/2))]

    def act(self, input_env, exploration_const, print_q_vals=False):
        epsilon = random.random()
        if epsilon < exploration_const:
            q_vals = self.q.eval({self.input: [input_env]})
            if print_q_vals:
                print(q_vals)
            action = np.argmax(q_vals)
        else:
            action = random.randint(0, self.num_actions - 1)
        return action

    def train(self, sess, train_screens, train_actions, train_rewards):
        sess.run([self.optimizer], feed_dict={
            self.input: train_screens,
            self.action: train_actions,
            self.target_q_t: train_rewards})

