#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from q_learning import QLearning


def greedy_probability_policy():

    env = gym.make('FrozenLake-v0')
    q_learning = QLearning(env.action_space.n, env.observation_space.n, 0.9)
    q_learning.set_general_state_action_values([0.5, 1, 0.5, 0.5])

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = q_learning.greedy_probability_policy(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print "Episode finished after {} timesteps".format(t+1)
                break

greedy_probability_policy()
