from random import uniform, randrange


class QLearning:

    def __init__(self, total_actions, total_states, epsilon=0.9):
        self.q_table = [[0 for _ in xrange(total_actions)] for _ in xrange(total_states)]
        self.epsilon = epsilon

    # Assign the same action values for all states
    def set_general_state_action_values(self, action_values):
        for i in xrange(len(self.q_table)):
            self.q_table[i] = action_values
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

    """ Reporting """

    def report_q_table(self):
        state_number = 0
        for state_actions in self.q_table:
            print str(state_number) + ":\t\t" + "  ".join(str(action_value) for action_value in state_actions)
            state_number += 1
