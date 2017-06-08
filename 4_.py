#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author: daniel
"""
import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf

env_d = 'LunarLander-v2'
env = gym.make(env_d)
env.reset()

#parameters
n0 = 8  #input state size
n1 = 15
n2 = 15
n3 = 4  #output action size
batch_size = 10
total_episodes = 30000
learning_rate = 0.008

observation = tf.placeholder()  #observation = input state

def agent(observation):
    W1 = tf.Variable(tf.zeros([n1, n0]))
    W2 = tf.Variable(tf.zeros([n2, n1]))
    W3 = tf.Variable(tf.zeros([n3, n2]))
    b1 = tf.Variable(tf.zeros([n1]))
    b2 = tf.Variable(tf.zeros([n2]))
    b3 = tf.Variable(tf.zeros([n3]))
    h1 = tf.tanh(tf.matmul(W1, observation) + b1)
    h2 = tf.tanh(tf.matmul(W2, h1) + b2)
    return tf.nn.softmax(tf.matmul(W3, h2) + b3)

y = agent(observation)
log_y = -tf.log(y)

init = tf.global_variables_initializer()
def main(argv):
    with tf.Session() as sess:
        sess.run(init)
        reward_sum = 0

        obsrv = env.reset() # Obtain an initial observation of the environment
        while episode_number <= total_episodes:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observation: obsrv})
            # sample action from distribution
            action = np.argmax(np.multinomial(1,action_probs))

            #calculate pi(action|observ)
            policy = -tf.log(y)[action]

            grad = tf.gradients(policy, obsrv)

            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)



            if episode_number % batch_size == 0:
                xxx = 4
            if done:
                episode_number += 1
                obsrv = env.reset()

if __name__ == '__main__':
    tf.app.run()
