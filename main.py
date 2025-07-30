# System imports
import gym
import matplotlib.pyplot as plt

# Reinforcement learning methods imports
from SARSA import SARSA
from Q_learning import QLearning
from Expected_SARSA import ExpectedSARSA
from MCC import MonteCarloControl

# Dynamic programming methods imports
from DP import PolicyIteration

# CREATE AN ENVIRONMENT
#env = gym.make("CliffWalking-v0")
#env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8") # map_name can also be "4x4"
#env = gym.make("Taxi-v3")

# SET ARGUMENTS
# The agents can share the same arguments but then they also share the same environment, which is fine since we
# always reset it. But I have decided to create one dictionary each to avoid any potential mistakes or cheating.
args_init_sarsa = {"env": gym.make("CliffWalking-v0"),
                   "random_Q_init": False,
                   "initial_Q_val": 0, # Not used if random_Q_init is True
                   "step_size": 0.2, # Not used for MCC
                   "epsilon": 0.1,
                   "discount_rate": 1}

args_init_q = {"env": gym.make("CliffWalking-v0"),
               "random_Q_init": False,
               "initial_Q_val": 0, # Not used if random_Q_init is True
               "step_size": 0.2, # Not used for MCC
               "epsilon": 0.1,
               "discount_rate": 1}

args_init_esarsa = {"env": gym.make("CliffWalking-v0"),
                    "random_Q_init": False,
                    "initial_Q_val": 0, # Not used if random_Q_init is True
                    "step_size": 0.2, # Not used for MCC
                    "epsilon": 0.1,
                    "discount_rate": 1}

args_init_mcc = {"env": gym.make("CliffWalking-v0"),
                 "random_Q_init": False,
                 "initial_Q_val": 0, # Not used if random_Q_init is True
                 "step_size": 0.2, # Not used for MCC
                 "epsilon": 0.1,
                 "discount_rate": 1}

args_init_pi = {"env": gym.make("CliffWalking-v0"),
                "theta": 0.0001,
                "discount_rate": 0.99,
                "value_iteration": False}

args_init_vi = {"env": gym.make("CliffWalking-v0"),
                "theta": 0.0001,
                "discount_rate": 0.99,
                "value_iteration": True}

# Run time arguments
no_replications = 10
no_episodes = 5000

"""
# INITIATE AGENTS
agent_sarsa = SARSA(**args_init_sarsa)
agent_q = QLearning(**args_init_q)
agent_esarsa = ExpectedSARSA(**args_init_esarsa)
agent_mcc = MonteCarloControl(**args_init_mcc)
agent_pi = PolicyIteration(**args_init_pi)
agent_vi = PolicyIteration(**args_init_vi)

# RUN FOR ONE REPLICATION
agent_sarsa.train(no_episodes)
agent_q.train(no_episodes)
agent_esarsa.train(no_episodes)
import time
start = time.time()
agent_mcc.train(no_episodes)
end = time.time()
agent_pi.train()
agent_vi.train()

# RUN FOR SEVERAL REPLICATIONS
agent_sarsa.run_replications(no_replications, no_episodes)
agent_q.run_replications(no_replications, no_episodes)
agent_esarsa.run_replications(no_replications, no_episodes)
agent_mcc.run_replications(no_replications, no_episodes)
agent_pi.run_replications(no_replications)
agent_vi.run_replications(no_replications)

# PLOT AVERAGE REWARD PER EPISODE OVER REPLICATIONS
plt.plot(agent_sarsa.episode_rewards_average_list, label="SARSA, epsilon-greedy")
plt.plot(agent_q.episode_rewards_average_list, label="Q-learning, epsilon-greedy")
plt.plot(agent_esarsa.episode_rewards_average_list, label="Expected SARSA, epsilon-greedy")
plt.plot(agent_mcc.episode_rewards_average_list, label="MCC, epsilon-greedy")
plt.ylim(-800, 0)
plt.title("Number of runs = " + str(no_replications) +
          ". Step size = " + str(args_init["step_size"]) +
          ". Discount rate = " + str(args_init["discount_rate"]) +
          ". Epsilon = " + str(args_init["epsilon"]) + ".")
plt.xlabel("Episode")
plt.ylabel("Average sum of rewards")
plt.legend()

# PLOT AVERAGE NUMBER OF ENVIRONMENT SAMPLES PER EPISODE OVER REPLICATIONS
plt.plot(agent_sarsa.episode_env_samples_average_list, label="SARSA, epsilon-greedy")
plt.plot(agent_q.episode_env_samples_average_list, label="Q-learning, epsilon-greedy")
plt.plot(agent_esarsa.episode_env_samples_average_list, label="Expected SARSA, epsilon-greedy")
plt.plot(agent_mcc.episode_env_samples_average_list, label="Expected SARSA, epsilon-greedy")
plt.ylim(0, 800)
plt.title("Number of runs = " + str(no_replications) +
          ". Step size = " + str(args_init["step_size"]) +
          ". Discount rate = " + str(args_init["discount_rate"]) +
          ". Epsilon = " + str(args_init["epsilon"]) + ".")
plt.xlabel("Episode")
plt.ylabel("Average number of environment samples")
plt.legend()
"""