import numpy as np
import Constants
import Utils


class Agent:
    def __init__(self, fourier_basis_order, epsilon):
        self.actions = Constants.actions
        self.feature_constants = Utils.get_fourier_basis_constants(fourier_basis_order)
        # self.feature_weights = np.random.standard_normal(size=(len(self.actions), len(self.feature_constants)))
        self.feature_weights = np.zeros(shape=(len(self.actions), len(self.feature_constants)))
        self.epsilon = epsilon

    def reset(self):
        # self.feature_weights = np.random.standard_normal(size=(len(self.actions), len(self.feature_constants)))
        self.feature_weights = np.zeros(shape=(len(self.actions), len(self.feature_constants)))

    def get_fourier_basis_function(self, current_state):
        scaled_current_state = [Utils.scale_to_fourier_basis(current_state[0], Constants.states[0]),
                                Utils.scale_to_fourier_basis(current_state[1], Constants.states[1])]
        return np.cos(np.pi * np.dot(self.feature_constants, scaled_current_state))

    def get_action(self, current_state):
        return np.random.choice(a=self.actions, p=Utils.get_action_distribution(
            np.argmax(np.dot(self.feature_weights, self.get_fourier_basis_function(current_state))), len(self.actions),
            self.epsilon))

    def get_action_value(self, current_state, current_action):
        return np.dot(self.feature_weights[current_action], self.get_fourier_basis_function(current_state))

    def sarsa_update(self, current_state, current_action, reward, next_state, next_action, lr, gamma):
        self.feature_weights[current_action] += lr * (
            reward + gamma * self.get_action_value(next_state, next_action) -
            self.get_action_value(current_state, current_action)) * self.get_fourier_basis_function(current_state)

        pass

    def q_learning_update(self, current_state, current_action, reward, next_state, lr, gamma):
        max_action_value = np.max(np.dot(self.feature_weights, self.get_fourier_basis_function(next_state)))
        current_action_value = self.get_action_value(current_state, current_action)

        self.feature_weights[current_action] += lr * (reward + gamma * max_action_value - current_action_value) * \
            self.get_fourier_basis_function(current_state)

        pass


def main():
    agent = Agent(fourier_basis_order=3, epsilon=Constants.epsilon)
    print agent.feature_weights[2]


if __name__ == '__main__':
    main()
