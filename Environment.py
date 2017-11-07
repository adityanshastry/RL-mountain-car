import numpy as np
import Constants
import Utils


class Environment:

    def __init__(self):
        self.time_step = 0
        self.current_state = np.array([-0.5, 0])
        self.state_position_range = Constants.states[0]
        self.state_velocity_range = Constants.states[1]
        self.reward = -1
        self.done = False
        pass

    def reset(self):
        self.time_step = 0
        self.current_state = np.array([-0.5, 0])
        self.done = False
        self.reward = -1
        pass

    def step(self, current_action):

        self.time_step += 1

        next_velocity_state = self.current_state[1] + 0.001 * current_action - 0.0025 * np.cos(3 * self.current_state[0])
        next_position_state = self.current_state[0] + next_velocity_state

        next_position_state, next_velocity_state = Utils.update_states_to_bounds(
            [next_position_state, next_velocity_state])

        if next_position_state == self.state_position_range[1] or self.time_step == Constants.episode_end_time_step:
            self.done = True
            self.reward = 0

        if next_position_state == self.state_position_range[0]:
            next_velocity_state = 0

        self.current_state = np.array([next_position_state, next_velocity_state])

        return self.current_state, self.reward, self.done


def main():
    pass


if __name__ == '__main__':
    main()
