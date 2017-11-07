import numpy as np

states = np.array([[-1.2, 0.5],
                   [-0.07, 0.07]]
                  )

actions = np.array([0, 1, -1])
action_representation = {
    0: 0, 1: 1, -1: 2
}
episode_end_time_step = 20000

epsilon = 0.5
lr = 0.05
gamma = 1.0
