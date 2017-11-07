import Constants
import numpy as np
from sklearn.utils.extmath import cartesian


def scale_to_fourier_basis(value, bounds):
    return (value - bounds[0]) / (bounds[1] - bounds[0])


def update_states_to_bounds(state):
    state[0] = max(state[0], Constants.states[0][0])
    state[0] = min(state[0], Constants.states[0][1])

    state[1] = max(state[1], Constants.states[1][0])
    state[1] = min(state[1], Constants.states[1][1])

    return state


def get_fourier_basis_constants(fourier_basis_order):
    return cartesian([np.arange(0, fourier_basis_order+1, 1), np.arange(0, fourier_basis_order+1, 1)])


def get_action_distribution(max_action, num_actions, epsilon):
    action_distribution = np.ones(shape=num_actions) * epsilon / num_actions
    action_distribution[Constants.actions[max_action]] = 1 - epsilon + (epsilon / num_actions)
    return action_distribution


def get_trial_splits(max_trials):
    starts = range(0, max_trials, 100)
    ranges = []
    for index, start in enumerate(starts):
        if index < len(starts):
            ranges.append([start, start+100])
    return ranges
    pass


def main():
    print get_trial_splits(100)


if __name__ == '__main__':
    main()
