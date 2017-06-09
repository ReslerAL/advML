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
a = tf.placeholder(tf.int8)
tvars = tf.trainable_variables()
policy = -tf.log(y)[a]
adam = tf.train.AdamOptimizer(learning_rate)
grad2var = adam.compute_gradients(policy)
updateGrads = adam.apply_gradients(grad2var)


init = tf.global_variables_initializer()
def main(argv):
    with tf.Session() as sess:
        sess.run(init)
        gradBuffer = sess.run(tvars)
        for ix in range(len(gradBuffer)): gradBuffer[ix] = 0.0 * gradBuffer[ix]
        reward_sum = 0
        grads = []
        rewards = []
        reward_total = 0
        expct_grad = 0

        obsrv = env.reset() # Obtain an initial observation of the environment
        while episode_number <= total_episodes:


            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observation: obsrv})
            # sample action from distribution
            action = np.argmax(np.multinomial(1,action_probs))

            #calculate gradient of this action - grad(-log(pi(action|obsrv)))

            grads += [[sess.run(grad2var, feed_dict={a:action, observation: obsrv})]]

            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            rewards += [reward]
            reward_total += reward

            if done:

                expct_grad += calGrads(grads, rewards, reward_total)

                if episode_number % batch_size == 0:
                    #parameters update
                    expct_grad /= float(batch_size)

                    grads = []
                    rewards = []
                    reward_total = 0
                    expct_grad = 0

                episode_number += 1
                obsrv = env.reset()


if __name__ == '__main__':
    tf.app.run()

#grads = list of list, each list is a tuple of gradients
#remember - grads of every iteration should be multiple by different reward weight
def calGrads(grads, rewards, reward_total):
    dic = {}
    for 
    result = [0]
    for i in range(len(grads)):
        grads_tuple = grads[i][0]
        for elem in grads_tuple:
            result += grads[i]*reward_total
        reward_total -= rewards[i]
    return result

"""
reminder:
every iteration we take the gradient of the policy and the reward and add to lists:
    i.e. we buffering it for later computation in the end of the episode
when we finish episode we take the lists and compue one sample of the expectations 
after finishing batch_size of samples\episodes we consider it as approximization of the expectation,
 we average the samples to calculate the gradient and then we update the model 
    according to the gradient according to the apply_gradients method 
"""