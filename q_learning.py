#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import uniform, randrange
import pickle


class QLearning(object):

    def __init__(self, total_actions, total_states, epsilon=0.1, learning_rate=0.8, discount_factor=0.99):
        self.q_table = [[0.0000 for _ in xrange(total_actions)] for _ in xrange(total_states)]
        self.epsilon = epsilon
        self.alpha = learning_rate
        self.gamma = discount_factor

    # Assign the same action values for all states
    def set_general_state_action_values(self, action_values):
        for i in xrange(len(self.q_table)):
            self.q_table[i] = list(action_values)
        self.report_q_table()

    # Choose the highest action value
    def greedy_policy(self, state):
        return self.q_table[state].index(max(self.q_table[state]))

    # Choose the highest action with probability
    # Idea is to choose more random action at the start, and as the learning progresses
    # have a higher probability of choosing the best action
    def greedy_probability_policy(self, state):
        if uniform(0, 1) <= self.epsilon:
            return randrange(0, len(self.q_table[state]))
        else:
            return self.greedy_policy(state)

    def update_state_action_function(self, pre_state, action, reward, post_state):
        self.q_table[pre_state][action] += self.alpha * (reward + (self.gamma * max(self.q_table[post_state])) - self.q_table[pre_state][action])

    """ Reporting """

    def report_q_table(self):
        state_number = 0
        for state_actions in self.q_table:
            print str(state_number) + ":\t\t" + "\t\t".join(str(round(action_value, 4))[1:] for action_value in state_actions)
            state_number += 1


    """ Write Dump """

    def write_q_function_dump(self):
        dump_file = open('./../dump.txt', 'w')
        pickle.dump(self.q_table, dump_file)
        dump_file.close()
