from q_learning import QLearning


class QLearningHash(QLearning):

    def __init__(self, total_actions, total_states, epsilon=0.1, learning_rate=0.1, discount_factor=0.99):
        super(QLearningHash, self).__init__(total_actions, total_states, epsilon, learning_rate, discount_factor)
        self.total_actions = total_actions
        self.q_table = dict()

        action_values = [_ for _ in xrange(total_actions)]
        for state in xrange(total_states):
            self.q_table[state] = list(action_values)

    # Assign the same action values for all states
    def set_general_state_action_values(self, action_values):
        if len(action_values) != self.total_actions:
            print "Number of action values (%i) did not match expected (%i) " % (len(action_values), self.total_actions)

        for i in xrange(len(self.q_table)):
            self.q_table[i] = list(action_values)

    """ Reporting """

    def report_q_table(self):
        for state_number in xrange(len(self.q_table)):
            print str(state_number) + ":\t\t" + "\t\t".join(str(round(action_value, 4))[1:] for action_value in self.q_table[state_number])
