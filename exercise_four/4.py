#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from q_learning_action import QLearningAction
from plot import Plot
from copy import deepcopy

# Exercise 3

# Implement Q-learning on FrozenLake. Make use of an ε-greedy policy where ε = 0.1, a discount- ing of 0.99,
# and a learning rate of 0.1. Plot the total reward per episode against the episode number to see if the algorithm
# is converging/stabilising (not fluctuating). If not stabilising, try running it for more episodes or tweak
# the learning rate, always keeping it within [0.0, 1.0). Once you have a stable learning algorithm,
# save a copy of the final Q function for later use. (Demo)


def q_learning_greedy_probability_policy():

    env = gym.make('FrozenLake-v0')
    q_learning = QLearningAction(env.action_space.n, env.observation_space.n, epsilon=0.1, learning_rate=0.1)
    q_learning.set_general_state_action_values([0.5, 1, 0.5, 0.5])
    episode_rewards = []
    all_over_reward = 0.0
    for i_episode in range(10000):

        # We start a new episode with have to reset the environment and stats
        observation = env.reset()
        accumulated_reward = 0.0
        action = q_learning.greedy_probability_policy(observation)

        for t in range(100):
            previous_action, previous_observation = int(action), int(observation)
            observation, reward, done, info = env.step(action)
            action = q_learning.greedy_probability_policy(observation)
            accumulated_reward += reward
            q_learning.update_state_action_function(previous_observation, previous_action, reward, action, observation)

            if done:
                print "Episode finished after {} timesteps".format(t+1)
                print "Total reward for episode %i: %i" % (i_episode, accumulated_reward)
                all_over_reward += accumulated_reward
                episode_rewards.append(accumulated_reward)
                break

            # Accumulate more reward
            accumulated_reward += reward

    plot = Plot()
    plot.plot_evolution(episode_rewards)
    print q_learning.q_table


q_learning_greedy_probability_policy()
