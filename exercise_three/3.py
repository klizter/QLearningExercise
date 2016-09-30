#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from q_learning import QLearning
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
    q_learning = QLearning(env.action_space.n, env.observation_space.n, epsilon=0.1, learning_rate=0.1)
    q_learning.set_general_state_action_values([0.5, 1, 0.5, 0.5])
    episode_rewards = []
    all_over_reward = 0.0
    for i_episode in range(7000):

        # We start a new episode with have to reset the environment and stats
        observation = env.reset()
        accumulated_reward = 0.0

        for t in range(100):

            # Show current state
            # env.render()

            # Choose action based on current experience
            action = q_learning.greedy_probability_policy(observation)

            # Save previous state, and commit action, resulting new current state
            previous_observation = observation
            observation, reward, done, info = env.step(action)

            # Accumulate more reward
            accumulated_reward += reward

            # Train algorithm based on new experience
            q_learning.update_state_action_function(previous_observation, action, reward, observation)

            #
            if done:
                print "Episode finished after {} timesteps".format(t+1)
                print "Total reward for episode %i: %i" % (i_episode, accumulated_reward)
                all_over_reward += accumulated_reward
                episode_rewards.append(accumulated_reward)
                break

    plot = Plot()
    plot.plot_evolution(episode_rewards)
    print q_learning.q_table


q_learning_greedy_probability_policy()
