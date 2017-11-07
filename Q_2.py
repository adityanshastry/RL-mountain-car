from joblib import Parallel, delayed
import numpy as np
import Utils
import Trainer
import sys


def parallelize(num_trials, num_episodes, lr, epsilon, gamma, fourier_basis_order, total_trials):
    return np.random.uniform(0, 1, size=10)


def test():
    total_trials, num_episodes = 100, 200
    lr_range = np.linspace(0.01, 0.05, num=5)
    epsilon_range = np.linspace(0.6, 0.9, num=4)
    gamma_range = np.linspace(0.6, 0.9, num=4)
    fourier_basis_order_range = np.linspace(1, 4, num=4)
    trial_ranges = Utils.get_trial_splits(total_trials)

    results = Parallel(n_jobs=10)(
        delayed(parallelize)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                     gamma=gamma, fourier_basis_order=fourier_basis_order,
                                     total_trials=total_trials)
        for lr in lr_range
        for epsilon in epsilon_range
        for gamma in gamma_range
        for fourier_basis_order in fourier_basis_order_range
        for trial_range in trial_ranges)

    results = np.array(results)
    print results.shape
    best = np.argmin(np.sum(results, axis=1))
    print best
    best_params = np.unravel_index(best, (1, 4, 4, 4, 5))
    print best_params
    print "Best sarsa params: " \
          "\n Fourier Basis : ", fourier_basis_order_range[best_params[1]], \
        "\n Gamma : ", gamma_range[best_params[2]], "\n Epsilon : ", epsilon_range[best_params[3]], \
        "\n Learning Rate : ", lr_range[best_params[4]]


def main(args):
    total_trials, num_episodes = 100, 200
    lr_range = np.linspace(0.01, 0.05, num=5)
    epsilon_range = np.linspace(0.6, 0.9, num=4)
    gamma_range = np.linspace(0.6, 0.9, num=4)
    fourier_basis_order_range = np.linspace(1, 4, num=4)

    if args == "1":
        trial_ranges = Utils.get_trial_splits(total_trials)
        sarsa_results = Parallel(n_jobs=10)(
            delayed(Trainer.train_sarsa)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                         gamma=gamma, fourier_basis_order=fourier_basis_order,
                                         total_trials=total_trials)
            for lr in lr_range
            for epsilon in epsilon_range
            for gamma in gamma_range
            for fourier_basis_order in fourier_basis_order_range
            for trial_range in trial_ranges)
        np.savetxt("sarsa_hyperparameters.txt", sarsa_results)

    elif args == "2":
        trial_ranges = Utils.get_trial_splits(total_trials)
        q_learning_results = Parallel(n_jobs=10)(
            delayed(Trainer.train_q_learning)(num_trials=trial_range, num_episodes=num_episodes, lr=lr, epsilon=epsilon,
                                              gamma=gamma, fourier_basis_order=fourier_basis_order,
                                              total_trials=total_trials)
            for lr in lr_range
            for epsilon in epsilon_range
            for gamma in gamma_range
            for fourier_basis_order in fourier_basis_order_range
            for trial_range in trial_ranges)
        np.savetxt("q_learning_hyperparameters.txt", q_learning_results)

    elif args == "3":
        sarsa_results = np.loadtxt("sarsa_hyperparameters.txt")
        q_learning_results = np.loadtxt("q_learning_hyperparameters.txt")
        best_sarsa = np.argmin(np.sum(sarsa_results, axis=1))
        best_sarsa_params = np.unravel_index(best_sarsa, (1, 4, 4, 4, 5))
        best_q_learning = np.argmin(np.sum(q_learning_results, axis=1))
        best_q_learning_params = np.unravel_index(best_q_learning, (1, 4, 4, 4, 5))

        print "Best sarsa params: " \
              "\n Fourier Basis : ", fourier_basis_order_range[best_sarsa_params[1]], \
            "\n Gamma : ", gamma_range[best_sarsa_params[2]], "\n Epsilon : ", epsilon_range[best_sarsa_params[3]], \
            "\n Learning Rate : ", lr_range[best_sarsa_params[4]]

        print "Best q_learning params: " \
              "\n Fourier Basis : ", fourier_basis_order_range[best_q_learning_params[1]], \
            "\n Gamma : ", gamma_range[best_q_learning_params[2]], "\n Epsilon : ", epsilon_range[best_q_learning_params[3]], \
            "\n Learning Rate : ", lr_range[best_q_learning_params[4]]

        Trainer.plot_rewards_and_episodes(sarsa_results[best_sarsa], q_learning_results[best_q_learning])

    pass


if __name__ == '__main__':
    main(sys.argv[1])
    # test()
