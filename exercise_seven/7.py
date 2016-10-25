#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from q_learning_hash import QLearningHash
from plot import Plot
from visualization .visualization import visualize_learning


# Exercise 7

# Apply Q-learning and the method implemented in exercise 4 on Taxi. Make use of an ε-greedy policy where ε = 0.1,
# a discounting of 0.99. You are free to tweak the learning rate (keeping it within [0.0, 1.0)) and the number of
# episodes you run the two learning algorithms for (Demo). Explain what data structure you use to represent the
# Q function (Report in writing). Show the performance of your algorithms by plotting the total reward
# per episode against the episode number (Report in writing).


# Action mapping
# 0: South
# 1: North
# 2: East
# 3: West
# 4: Pickup
# 5: Dropoff


def q_learning_greedy_probability_policy():

    env = gym.make('Taxi-v1')
    q_learning = QLearningHash(env.action_space.n, env.observation_space.n, epsilon=0.1, learning_rate=0.1)
    q_learning.set_general_state_action_values([500.0, 500.0, 800.0, 800.0, 1900.0, 1900.0])
    episode_rewards = []
    all_over_reward = 0.0
    total_timesteps = 0
    total_episodes = 2000
    for i_episode in range(total_episodes):

        # We start a new episode with have to reset the environment and stats
        observation = env.reset()
        accumulated_reward = 0.0

        while True:

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
                # print "Episode finished after {} timesteps".format(t+1)
                print "Total reward for episode %i: %i" % (i_episode, accumulated_reward)
                all_over_reward += accumulated_reward
                episode_rewards.append(accumulated_reward)
                break

    plot = Plot()
    plot.plot_evolution(episode_rewards)
    print "Average number of timesteps per episode: %d" % (float(total_timesteps) / float(total_episodes))
    print q_learning.q_table
    visualize_learning('Taxi-v1', q_learning)

q_learning_greedy_probability_policy()
