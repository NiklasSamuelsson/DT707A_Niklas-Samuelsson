# System imports
import gym
import numpy as np


class PolicyIteration:
    """ Policy and Value iteration agent for environments with known and perfect models of the dynamics """

    def __init__(self, env, theta, discount_rate, value_iteration):
        """
        Contains functionality for both policy and value iteration agents.

        Parameters
        ----------
        env : OpenAI discrete style gym environment
            Environment in which an agent will act. Must have a dictionary called "P" with all state-action
            pairs and their transitions.
        theta : float
            The threshold for when to stop doing policy evaluations for policy iteration.
        discount_rate : float
            The rate by which to discount future rewards.
        value_iteration : bool
            Whether to use value iteration or policy iterations. If True, then use value iteration. If False,
            then use policy iterations.

        """
        # Initiate environment
        self.env = env
        self.terminal_states = []
        self.actionable_states = []
        self.__find_actionable_states()

        # Initialize parameters
        self.theta = theta
        self.discount_rate = discount_rate
        self.value_iteration = value_iteration
        self.delta = None
        self.__reset_delta()

        # Initialize V and the policy
        self.V = None
        self.__reset_V()
        self.policy = None
        self.__reset_policy()
        self.policy_stable = None
        self.__reset_policy_stable()

        # Initialize metrics
        self.no_iterations_master_list = None

    def __reset_delta(self):
        """ Resets delta to a value larger than the threshold to guarantee one lap in the loop """
        self.delta = self.theta + 1

    def __reset_V(self):
        """ Resets the state-value function based on the environment's states """
        self.V = {state: 0 for state in self.env.P}

    def __reset_policy(self):
        """ Resets the policy to random actions for each state """
        self.policy = {state: np.random.choice(a=list(self.env.P[state].keys()), size=1)[0] for state in self.env.P}

    def __reset_policy_stable(self):
        """ Resets the policy_stable trigger to False to make sure we try to improve the policy after each evaluation """
        self.policy_stable = False

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

    def __find_actionable_states(self):
        """ Finds all states which are not terminal states so we can loop throug them """
        self.__find_terminal_states()
        self.actionable_states = list(set(list(self.env.P.keys())) - set(self.terminal_states))

    def __policy_evaluation(self):
        """ Performs the evaluation of the state-value function under the current policy"""
        # Reset delta to guarantee at least one evaluation for all states
        self.__reset_delta()

        # Loop as long as delta is larger than the threshold
        while self.delta > self.theta:
            self.delta = 0
            for state in self.actionable_states:
                # Save the current value for V
                v = self.V[state]

                # Get the action for the current policy
                action = self.policy[state]

                # Loop through all state-action combinations
                updated_value = 0
                for p_tuple in self.env.P[state][action]:
                    # Get the new state for the transition
                    new_state = p_tuple[1]

                    # Get the probability for the transition
                    p = p_tuple[0]

                    # Get the reward for the transition
                    reward = p_tuple[2]

                    # Update V
                    updated_value += p * (reward + self.discount_rate * self.V[new_state])

                # Assign the new value to V
                self.V[state] = updated_value

                # Update the difference
                self.delta = max(self.delta, abs(v - self.V[state]))

            # Only do one sweep of policy evaluation if we're running value iteration
            if self.value_iteration:
                self.delta = 0

    def __policy_improvement(self):
        """ Performs the improvement of the policy given the current state-value function """
        # We'll set this one to false if we perform any updates
        self.policy_stable = True

        # Loops through all states
        for state in self.actionable_states:
            # Save current action
            old_action = self.policy[state]

            # Select greedy action
            action_values = {}
            for action in self.env.P[state]:
                action_value = 0
                for p_tuple in self.env.P[state][action]:
                    # Get the new state for the transition
                    new_state = p_tuple[1]

                    # Get the probability for the transition
                    p = p_tuple[0]

                    # Get the reward for the transition
                    reward = p_tuple[2]

                    # Update the action value
                    action_value += p * (reward + self.discount_rate * self.V[new_state])

                # Update the action values
                action_values[action] = action_value

            # Get the highest valued action
            max_action = max(action_values, key=action_values.get)

            # Update the policy
            self.policy[state] = max_action

            if old_action != self.policy[state]:
                self.policy_stable = False

    def train(self):
        """ The main training loop. Will alternate between policy evaluation and policy improvement until done """
        # Reset everything
        self.env.reset()
        self.__reset_policy()
        self.__reset_V()
        self.__reset_delta()
        self.__reset_policy_stable()

        # Loop as long as the policy is not stable
        self.no_iterations = 0
        while not self.policy_stable:
            # Start with policy evaluation
            self.__policy_evaluation()

            # Continue with policy improvement
            self.__policy_improvement()

            # Update metrics
            self.no_iterations += 1

    def run_replications(self, no_replications):
        """
        Runs the training part a number of times and saves the metrics results in a list.

        Parameters
        ----------
        no_replications : int
            The number of replications to run.

        """
        # Reset metric list
        self.no_iterations_master_list = []

        # Run the number of replications and record the results
        for i in range(no_replications):
            self.train()
            self.no_iterations_master_list.append(self.no_iterations)
