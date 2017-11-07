from __future__ import division
import Environment
import Agent
import Constants
import numpy as np
import matplotlib.pyplot as plt
# import gym
import sys
import Utils
from joblib import Parallel, delayed


def plot_rewards_and_episodes(sarsa_rewards, q_learning_rewards):
    sarsa, = plt.plot(range(len(sarsa_rewards)), sarsa_rewards, 'b-')
    q_learning, = plt.plot(range(len(q_learning_rewards)), q_learning_rewards, 'r-')
    plt.legend([sarsa, q_learning], ['Sarsa', 'Q-Learning'])
    plt.axis([0, len(q_learning_rewards), -1000, 0])
    plt.title("Undiscounted Returns vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Undiscounted Returns")
    plt.show()


def train_sarsa(num_trials, num_episodes, lr, epsilon, gamma, fourier_basis_order, total_trials):
    undiscounted_returns = np.zeros(shape=num_episodes)

    environment = Environment.Environment()
    agent = Agent.Agent(fourier_basis_order=fourier_basis_order, epsilon=epsilon)

    for trial in xrange(num_trials[0], num_trials[1]):
        print "Trial: ", trial
        agent.reset()
        for episode in xrange(num_episodes):
            environment.reset()
            current_state = environment.current_state
            current_action = agent.get_action(current_state)
            for time_step in xrange(Constants.episode_end_time_step):
                next_state, reward, done = environment.step(current_action)
                if done:
                    # print time_step
                    break
                next_action = agent.get_action(next_state)
                agent.sarsa_update(current_state, current_action, reward, next_state, next_action, lr, gamma)
                current_state = next_state
                current_action = next_action

            undiscounted_returns[episode] += -1 * environment.time_step / total_trials

    return undiscounted_returns


def train_q_learning(num_trials, num_episodes, lr, epsilon, gamma, fourier_basis_order, total_trials):
    undiscounted_returns = np.zeros(shape=num_episodes)

    environment = Environment.Environment()
    # environment = gym.make("MountainCar-v0")
    agent = Agent.Agent(fourier_basis_order=fourier_basis_order, epsilon=epsilon)
    for trial in xrange(num_trials[0], num_trials[1]):
        print "Trial: ", trial
        agent.reset()
        for episode in xrange(num_episodes):
            environment.reset()
            # current_state = [-0.5, 0]
            current_state = environment.current_state
            current_action = agent.get_action(current_state)
            while True:
                # environment.render()
                next_state, reward, done = environment.step(Constants.action_representation[current_action])
                if done:
                    # print environment.time_step
                    break
                agent.q_learning_update(current_state, current_action, reward, next_state, lr, gamma)
                current_state = next_state
                current_action = agent.get_action(next_state)
            undiscounted_returns[episode] += -1 * environment.time_step / total_trials

    return undiscounted_returns


def main(args):
    total_trials, num_episodes = 10000, 200
    lr, epsilon, gamma, fourier_basis_order = 0.05, 0.5, 1.0, 1

    if args == "1":
        print "Sarsa"
        trial_ranges = Utils.get_trial_splits(total_trials)
        sarsa_results = Parallel(n_jobs=10)(
            delayed(train_sarsa)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                 gamma=gamma, fourier_basis_order=fourier_basis_order, total_trials=total_trials)
            for trial_range in trial_ranges)
        np.savetxt("sarsa_rewards.txt", np.sum(sarsa_results, axis=0))

    elif args == "2":
        print "Q-Learning"
        trial_ranges = Utils.get_trial_splits(total_trials)
        q_learning_results = Parallel(n_jobs=10)(
            delayed(train_q_learning)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                      gamma=gamma, fourier_basis_order=fourier_basis_order, total_trials=total_trials)
            for trial_range in trial_ranges)
        np.savetxt("q_learning_rewards.txt", np.sum(q_learning_results, axis=0))

    elif args == "3":
        sarsa_results = np.loadtxt("sarsa_rewards.txt")
        q_learning_results = np.loadtxt("q_learning_rewards.txt")
        plot_rewards_and_episodes(sarsa_results, q_learning_results)


if __name__ == '__main__':
    main(sys.argv[1])
