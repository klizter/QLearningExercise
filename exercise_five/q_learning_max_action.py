from q_learning import QLearning
import pickle


class QLearningMaxAction(QLearning):

    def __init__(self, total_actions, total_states, epsilon=0.1, learning_rate=0.1, discount_factor=0.99):
        super(QLearningMaxAction, self).__init__(total_actions, total_states, epsilon=epsilon, learning_rate=learning_rate, discount_factor=discount_factor)
        self.q_max_table = list()

    def update_state_action_function(self, pre_state, action, reward, post_state):
        self.q_table[pre_state][action] += self.alpha * (reward + (self.gamma * max(self.q_max_table[post_state])) - self.q_table[pre_state][action])

    """ Read Dump """

    def read_q_function_dump(self):
        dump_file = open('./../dump.txt', 'r')
        self.q_max_table = pickle.load(dump_file)
        dump_file.close()
