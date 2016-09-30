from q_learning import QLearning


class QLearningAction(QLearning):

    def __init__(self, total_actions, total_states, epsilon=0.1, learning_rate=0.1, discount_factor=0.99):
        super(QLearningAction, self).__init__(total_actions, total_states, epsilon=epsilon, learning_rate=learning_rate, discount_factor=discount_factor)

    def update_state_action_function(self, pre_state, pre_action, reward, next_action, new_state):
        self.q_table[pre_state][pre_action] += self.alpha * (reward + (self.gamma * self.q_table[new_state][next_action]) - self.q_table[pre_state][pre_action])
