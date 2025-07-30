# Base class import
from RL_base import RLBase


class QLearning(RLBase):
    """ Q-learning algorithm with constant step-size following an epsilon-greedy policy """

    def _update_Q(self, old_state, action, reward, new_state):
        """
        Updates the state-action-value function.

        Parameters
        ----------
        old_state : int
            The old state in which the first action was taken.
        old_action : int
            The action taken in the old state.
        reward : float
            The reward received from the old state and action transition.
        new_state : int
            The state which the old state-action pair led to.

        """
        # Find the maximizing action in the new state
        maximizing_action = max(self.Q[new_state], key=self.Q[new_state].get)

        # Update Q
        self.Q[old_state][action] = self.Q[old_state][action] + \
                                        self.step_size * \
                                        (reward +
                                         self.discount_rate *
                                         self.Q[new_state][maximizing_action] -
                                         self.Q[old_state][action])

    def _run_one_episode(self, old_state):
        """
        Runs one episode.

        Parameters
        ----------
        old_state : int
            The starting state for the agent.

        Returns
        -------
        The total reward and the number of environment samples as float/int and ints.

        """
        # Initialize metrics
        current_episode_reward = 0
        current_episode_samples = 0

        # Loop until the episode is done
        done = False
        while not done:
            # Choose an action based on the old state
            old_action = self._get_action_epsilon_greedy(old_state)

            # Take the old action
            new_state, reward, done, _ = self.env.step(old_action)

            # Update Q
            self._update_Q(old_state, old_action, reward, new_state)

            # Update the old state and action
            old_state = new_state

            # Update the metrics
            current_episode_reward += reward
            current_episode_samples += 1

        return current_episode_reward, current_episode_samples
