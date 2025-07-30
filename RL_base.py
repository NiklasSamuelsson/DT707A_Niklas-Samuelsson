# System imports
import numpy as np
import matplotlib.pyplot as plt


class RLBase:
    """ Contains common methods for basic RL algorithms """
    def __init__(self, env, random_Q_init, initial_Q_val, step_size, epsilon, discount_rate):
        """
        Contains all common methods for simple temporal difference methods and Monte Carlo Control
        using an epsilon-greedy policy.

        Parameters
        ----------
        env : OpenAI discrete style gym environment
            Environment in which an agent will act. Must have a dictionary called "P" with all state-action
            pairs and their transitions.
        random_Q_init : bool
            Whether to initialize Q with samples from a standard normal distribution. If True, uses standard normal
            samples. If False, uses the initial_Q_val.
        initial_Q_val : float/int
            The value to use for all stat-action pairs in Q. random_Q_init must be False.
        step_size : float
            The learning rate.
        epsilon : float
            The probability by which to select a random action instead of the greedy one.
        discount_rate : float
            The rate by which to discount future rewards.

        """
        # Get environment
        self.env = env
        self.random_Q_init = random_Q_init
        self.initial_Q_val = initial_Q_val
        self.terminal_states = []
        self.__find_terminal_states()

        # Initialize parameters
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.Q = None
        self.__reset_Q()

        # Initialize metrics
        self.episode_rewards_list = None
        self.episode_rewards_master_list = None
        self.episode_rewards_average_list = None

        self.episode_env_samples_list = None
        self.episode_env_samples_master_list = None
        self.episode_env_samples_average_list = None

    def __reset_episode_lists(self):
        """ Resets the metric lists used to track every episode """
        self.episode_rewards_list = []
        self.episode_env_samples_list = []

    def __reset_master_lists(self):
        """ Resets the metric lists used to track all replications """
        self.episode_rewards_master_list = []
        self.episode_env_samples_master_list = []

    def __reset_average_lists(self):
        """ Resets the metric lists used to calculate the average over all replications for each episode """
        self.episode_rewards_average_list = []
        self.episode_env_samples_average_list = []

    def __calculate_averages(self, master_list, average_list, no_episodes, no_replications):
        """
        Calculates the average for each episode over all replications.
        For example, if 10 replications are run, we will have 10 episode number 1, 10 episode number 2 etc.
        It calculates the average for ALL episodes number 1, then all episodes number 2 etc.

        Parameters
        ----------
        master_list : list of lists
            The list on which to average
        average_list : list
            The resulting list to modify
        no_episodes :
            The number of episodes run.
        no_replications :
            The number of times the number of episodes are run.

        """
        # Calculate the average of lists per episode
        for i in range(no_episodes):
            average_list.append(0)
            for ii in range(no_replications):
                average_list[i] += master_list[ii][i]
            average_list[i] /= no_replications

    def __find_terminal_states(self):
        """ Finds the terminal states given the environment dynamics """
        # Get the environment dynamics
        P = self.env.P

        # Find all the stat-action pairs that leads to a terminal state and record the terminal state
        for state in P.keys():
            for action in P[state].keys():
                if P[state][action][0][3]:
                    self.terminal_states.append(P[state][action][0][1])

        # Only save unique terminal states
        self.terminal_states = list(set(self.terminal_states))

    def __reset_Q(self):
        """ Resets the state-action-value-function to all zeroes """
        # Build Q for all state-action pairs
        if self.random_Q_init:
            self.Q = {state: {action: (np.random.randn() if state not in self.terminal_states else 0)
                              for action in self.env.P[state]} for state in self.env.P}
        else:
            self.Q = {state: {action: (self.initial_Q_val if state not in self.terminal_states else 0)
                              for action in self.env.P[state]} for state in self.env.P}

    def __reset_returns(self):
        """ Resets the return dictionary used for Monte Carlo Control """
        # Build return for all state-action pairs
        self.returns = {state: {action: []
                        for action in self.env.P[state]} for state in self.env.P}

    def _get_greedy_action(self, state):
        """
        Gets a greedy action from Q in the given state. Break ties by selecting randomly from a uniform
        probability distribution.

        Parameters
        ----------
        state : int
            The state for which to get a greedy action.

        Returns
        -------
        A greedy action.

        """
        # Select randomly according to a uniform distribution if we have a tie
        max_value = max(self.Q[state].values())
        keys = [key for key, value in self.Q[state].items() if value == max_value]
        action = np.random.choice(a=keys, size=1)[0]

        return action

    def _get_action_epsilon_greedy(self, state):
        """
        Get the action according to an epsilon greedy policy

        Parameters
        ----------
        state : int
            The state from which to select an action.

        Returns
        -------
        An int which represents an action.

        """
        # Take action according to an epsilon-greedy policy
        # Random selection
        if np.random.choice(a=["greedy", "random"], size=1, p=[1-self.epsilon, self.epsilon])[0] == "random":
            action = np.random.choice(a=list(self.Q[state].keys()), size=1)[0]
        # Greedy selection
        else:
            action = self._get_greedy_action(state)

        return action

    def _update_Q(self):
        """
        Unique function for each algorithm that needs to be implemented for each subclass.
        Needs to update Q.

        """
        raise NotImplementedError("Need to implement a function to update Q.")

    def _run_one_episode(self, old_state):
        """
        Unique function for each algorithm that needs to be implemented for each subclass.
        Needs to take old_state as an argumement (int) and return the total reward and
        the number of environment samples as float/int and ints.
        """
        raise NotImplementedError("Need to implement one episode run.")
        return current_episode_reward, current_episode_samples

    def train(self, no_episodes):
        """
        Will let the agent train by running it for the specified number of episodes.
        Resets the state-action-value function before the first episode.

        Parameters
        ----------
        no_episodes : int
            The number of episodes to run.

        """
        # Initialize Q, returns (for MCC) and episode rewards
        self.__reset_Q()
        self.__reset_returns()
        self.__reset_episode_lists()

        # Loop for each episode
        for episode in range(no_episodes):
            if episode % 10 == 0:
                print("Episode", episode, "starting")

            # Reset the environment and record the state
            first_old_state = self.env.reset()

            # Complete one episode
            current_episode_reward, current_episode_samples = self._run_one_episode(first_old_state)

            # Update metric lists
            self.episode_rewards_list.append(current_episode_reward)
            self.episode_env_samples_list.append(current_episode_samples)

    def run_replications(self, no_replications, no_episodes):
        """
        Makes no_replications number of runs of the specified number of episodes.
        So for examples we choose to run the agent for 100 episodes for 10 replications, the agent will have run
        a total of 100*10 = 1000 episodes. The state-action-value function is reset before the start of each
        new replication.

        Parameters
        ----------
        no_replications : int
            The number of time to repeat the episode runs.
        no_episodes : int
            The number of episodes that will be run within each replication.

        """
        # Reset metric lists
        self.__reset_master_lists()
        self.__reset_average_lists()

        # Run replications
        for i in range(no_replications):
            if i % 1 == 0:
                print("REPLICATION", i)
            self.train(no_episodes)

            # Update master lists
            self.episode_rewards_master_list.append(self.episode_rewards_list.copy())
            self.episode_env_samples_master_list.append(self.episode_env_samples_list.copy())

        # Calculate the average reward per episode
        self.__calculate_averages(self.episode_rewards_master_list, self.episode_rewards_average_list,
                                no_episodes, no_replications)

        # Calculate the average no. env. samples per episode
        self.__calculate_averages(self.episode_env_samples_master_list, self.episode_env_samples_average_list,
                                no_episodes, no_replications)

    def plot_results(self, list_to_plot, ylim_min=-4000, ylim_max=0):
        """
        Plots the specified list.

        Parameters
        ----------
        list_to_plot : list
            List to plot.
        ylim_min : int/float
            The minimum limit of the vertical axis to show.
        ylim_max : int/float
            The maximum limit of the vertical axis to show.

        """
        # Plot selected list
        plt.plot(list_to_plot)
        plt.ylim(ylim_min, ylim_max)
        plt.show()
