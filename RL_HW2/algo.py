import numpy as np
import random
from abc import abstractmethod
from collections import defaultdict
class QAgent:
	def __init__(self, gamma, alpha, epsilon):
		self.q_table = {}
		self.actions = [_ for _ in range(4)]
		self.gamma = gamma
		self.alpha = alpha
		self.epsilon = epsilon
		for i in range(8):
			for j in range(8):
				for k in range(4):
					self.q_table[str(i)+str(j)+str(k)] = 0
		pass

	@abstractmethod
	def select_action(self, obs):
		if np.random.uniform() > self.epsilon:
			q_value_list = []
			max_action_list = []
			for i in range(4):
				state = str(int(obs[0])) + str(int(obs[1])) + str(i)
				q_value_list.append(self.q_table[state])
			for i in range(4):
				state = str(int(obs[0])) + str(int(obs[1])) + str(i)
				if self.q_table[state] == np.max(q_value_list):
					# print(self.q_table[state])
					max_action_list.append(i)
			action = np.random.choice(max_action_list)
		else:
			action = np.random.choice(self.actions)
		if self.epsilon > 0:
			self.epsilon -= 0.00001
		if self.epsilon < 0:
			self.epsilon = 0

		return action

	def update_q_table(self, obs, obs_next, action, reward):
		max_action = -1
		max_q = -100000
		for i in range(4):
			state = str(int(obs[0])) + str(int(obs[1])) + str(i)
			if self.q_table[state] > max_q:
				max_q = self.q_table[state]
				max_action = i
		next_action = max_action
		next_s = str(int(obs_next[0])) + str(int(obs_next[1])) + str(next_action)
		now_s = str(int(obs[0])) + str(int(obs[1])) + str(action)
		new_q = reward + self.gamma * self.q_table[next_s]
		# if new_q > 0 :
		# 	print('new_q',new_q)
		# if (1 - self.alpha)*self.q_table[now_s] + self.alpha*new_q > 0:
		# 	print('state',now_s,'q_value',self.q_table[now_s])
		self.q_table[now_s] = (1 - self.alpha)*self.q_table[now_s] + self.alpha*new_q


