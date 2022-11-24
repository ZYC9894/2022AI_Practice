### MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='A program to run assignment 1 implementations.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--env", 
					help="The name of the environment to run your algorithm on.", 
					choices=["Deterministic-4x4-FrozenLake-v0","Deterministic-8x8-FrozenLake-v0"],
					default="Deterministic-4x4-FrozenLake-v0")

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)

	############################

	# YOUR IMPLEMENTATION HERE #

	"(F) For each pair of states in [1, nS] and actions in [1, nA]"
	"(T) For each pair of states in [0, nS) and actions in [0, nA)"

	"(F) P[state][action] is a tuple of the form (probability, nextstate, reward, terminal)"
	"(T) P[state][action] is a LIST OF TUPLES of the form (probability, nextstate, reward, terminal)"

	# v_{t+1}(s)= sum_{a \in A} \pi(a|s)(R(s,a)+\gamma\sum_{s'\in S}P(s'|s,a)v_{t}(s'))
	# P[s][a] has value for all <s,a> pairs, so for all s, it has nA actions. \pi(a|s) = 1/nA
	# P[s][a] has only one 'nextstate' for each <s,a> pair, so \sum_{s'\in S}P(s'|s,a)v_{t}(s')=P(s'|s,a)v_{t}(s')
	
	new_tol = tol + 1

	while new_tol >= tol:
		prev_value_function = value_function.copy()
		value_function = np.zeros(nS)
		for s in range(nS):
			a = policy[s]
			if len(P[s][a]) == 1: # P[s][a] has only one tuple
				probability, nextstate, reward, terminal = P[s][a][0]

				# v'(s) = reward + gamma * probability * v(s')
				value_function[s] = reward + gamma * probability * prev_value_function[nextstate]
			else:
				# v'(s) = sum(probability(s'|s,a) * (reward(s'|s, a) + gamma * v(s')))
				arr = [probability * (reward + gamma * prev_value_function[nextstate]) for probability, nextstate, reward, terminal in P[s][a]]
				value_function[s] = sum(arr)

		new_tol = max(abs(value_function - prev_value_function))
				

	############################
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	new_policy = np.zeros(nS, dtype='int')

	############################

	# YOUR IMPLEMENTATION HERE #

	for s in range(nS):
		value = np.zeros(nA)
		for a in range(nA):
			if len(P[s][a]) == 1: # P[s][a] has only one tuple
				probability, nextstate, reward, terminal = P[s][a][0]
				#reward + gamma * probability * v(s')
				value[a] = reward + gamma * probability * value_from_policy[nextstate] 
			else:
				# sum_s' (probability * (reward + gamma * v(s')))
				arr = [probability * (reward + gamma * value_from_policy[nextstate]) for probability, nextstate, reward, terminal in P[s][a]]
				value[a] = sum(arr)

		#v'(s) = argmax_a ( reward + gamma * probability * v(s'))
		#new_policy[s] = value.index(max(value))
		new_policy[s] = np.argmax(value)


	############################
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	############################

	# YOUR IMPLEMENTATION HERE #


	i = 0
	diff = 1
	while diff != 0:
		prev_value_function = value_function.copy()
		prev_policy = policy.copy()

		# update value function and policy
		value_function = policy_evaluation(P, nS, nA, policy, tol = tol)
		
		policy = policy_improvement(P, nS, nA, value_function, policy)

		print("\n############ Loop", i, " ###########", ": ", max(abs(policy - prev_policy)))
		print('policy', policy)
		
		# diff == 0, then stop
		diff = max(abs(policy - prev_policy))
		i += 1

	print('value_function: ', value_function)
	print('policy', policy)
	############################
	return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################

	# YOUR IMPLEMENTATION HERE #

	new_tol = tol + 1
	while new_tol >= tol:
		prev_value_function = value_function.copy()
		for s in range(nS):
			value = np.zeros(nA)
			for a in range(nA):
				if len(P[s][a]) == 1: # P[s][a] has only one tuple
					probability, nextstate, reward, terminal = P[s][a][0]
					#reward + gamma * probability * v(s')
					value[a] = reward + gamma * probability * prev_value_function[nextstate] 
				else:
					# sum_s' (probability * (reward + gamma * v(s')))
					arr = [probability * (reward + gamma * prev_value_function[nextstate]) for 
						probability, nextstate, reward, terminal in P[s][a]]
					value[a] = sum(arr)

			#v'(s) = argmax_a ( reward + gamma * probability * v(s'))
			value_function[s] = max(value)

		new_tol = max(abs(value_function - prev_value_function))
		print(new_tol)

	policy = policy_improvement(P, nS, nA, value_function, policy)

	print('value_function: ', value_function)
	print('policy', policy)
	############################

	return value_function, policy


def render_single(env, policy, max_steps=100):
	"""
	This function does not need to be modified
	Renders policy once on environment. Watch your agent play!

	Parameters
	----------
	env: gym.core.Environment
		Environment to play on. Must have nS, nA, and P as attributes.
	Policy: np.array of shape [env.nS]
		The action to take at a given state
	"""
	episode_reward = 0
	ob = env.reset()
	done = False
	for t in range(max_steps):
		env.render()
		time.sleep(0.25)
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	env.render()
	if not done:
		print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
	else:
		print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
	# read in script argument
	args = parser.parse_args()
	print(type(args.env),args.env)
	# Make gym environment

	# env = gym.make('Deterministic-4x4-FrozenLake-v0')
	env = gym.make('Deterministic-8x8-FrozenLake-v0')

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)


