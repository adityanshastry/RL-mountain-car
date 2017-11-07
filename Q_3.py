from joblib import Parallel, delayed
import numpy as np
import Utils
import Trainer
import sys


def main(args):
    total_trials, num_episodes = 10000, 200
    lr, epsilon, gamma, fourier_basis_order = 0.05, 0.5, 1.0, 1

    if args == "1":
        print "Sarsa"
        trial_ranges = Utils.get_trial_splits(total_trials)
        sarsa_results = Parallel(n_jobs=10)(
            delayed(Trainer.train_sarsa)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                         gamma=gamma, fourier_basis_order=fourier_basis_order,
                                         total_trials=total_trials)
            for trial_range in trial_ranges)
        np.savetxt("sarsa_rewards.txt", np.sum(sarsa_results, axis=0))

    elif args == "2":
        print "Q-Learning"
        trial_ranges = Utils.get_trial_splits(total_trials)
        q_learning_results = Parallel(n_jobs=10)(
            delayed(Trainer.train_q_learning)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                              gamma=gamma, fourier_basis_order=fourier_basis_order,
                                              total_trials=total_trials)
            for trial_range in trial_ranges)
        np.savetxt("q_learning_rewards.txt", np.sum(q_learning_results, axis=0))

    elif args == "3":
        sarsa_results = np.loadtxt("sarsa_rewards.txt")
        q_learning_results = np.loadtxt("q_learning_rewards.txt")
        Trainer.plot_rewards_and_episodes(sarsa_results, q_learning_results)


if __name__ == '__main__':
    main(sys.argv[1])
