# System imports
import numpy as np

# Base class import
from RL_base import RLBase


class MonteCarloControl(RLBase):
    """ On-policy first-visit Monte Carlo Control implementation following an epsilon-greedy policy """

    def _update_Q(self, state, action, G):
        """
        Updates the state-action-value function.

        Parameters
        ----------
        state : int
            The state of the state-action pair value to update.
        action : int
            The action of the state-action pair value to update.
        G : float/int
            The return observed in so far in the episode.

        """
        # Append G to the list of returns for the state-action pair
        self.returns[state][action].append(G)

        # Calculate the average return and update Q
        self.Q[state][action] = np.mean(self.returns[state][action])

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

        # Initialize lists of states, actions and rewards
        state_action_pairs = []
        rewards = [0] # We don't have a reward in time step 0

        # Generate a complete episode
        done = False
        while not done:
            # Choose an action based on the old state
            action = self._get_action_epsilon_greedy(old_state)

            # Take the old action
            new_state, reward, done, _ = self.env.step(action)

            # Update lists
            state_action_pairs.append((old_state, action))
            rewards.append(reward)

            # Update the old state and action
            old_state = new_state

            # Update the metrics
            current_episode_reward += reward
            current_episode_samples += 1

        # Initialize G
        G = 0

        # Create a dictionary containing the first visits of all visited state-action pairs
        unique_pairs = set(state_action_pairs)
        first_visits = {state_action_pair: state_action_pairs.index(state_action_pair)
                        for state_action_pair in unique_pairs}

        # Loop through all time steps of the episode backwards
        for t in range(current_episode_samples-1, -1, -1):

            # Update G
            G = self.discount_rate * G + rewards[t+1]

            # Get current state-action pair
            current_state_action_pair = state_action_pairs[t]

            # Check if we have encountered the state-action pair at time t previously in the episode
            if t == first_visits[current_state_action_pair]:
                # If not, then update the state-action values
                self._update_Q(state=current_state_action_pair[0], action=current_state_action_pair[1], G=G)

        return current_episode_reward, current_episode_samples
