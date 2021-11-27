import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
import json
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import os

class EnergyStorageEnv(gym.Env):

	""" Define environment which energy storage works in.
		Observations are hourly power spot price, storage level, and
		average price of stored power.
	"""

	metadata = {'render.modes': ['human']}

	def __init__(self):

		self.start_date = datetime.fromisoformat("2015-06-01")  # relevant for price simulation
		self.cur_date = self.start_date  # keep track of current date
		self.time_step = 0  # variable used for slicing mean and var values
		self.end_date = datetime.fromisoformat("2015-07-01")
		self.time_index = pd.Series(pd.date_range(start=self.start_date, end=self.end_date, freq="H"))
		self._get_spot_price_params()  # might be necessary to specify path here?
		self.observation_space = 3
		self.action_space = ["up", "down", "cons"]
		self.penalty = -0.05

		# storage specifics
		self.max_stor_lev = 0.005  # in MWh
		self.max_wd = -0.0024  # in MW
		self.max_in = 0.00165  # in MW
		self.stor_eff = 0.9  # 10% loss for each conversion
		
		# set initial parameters for price, storage level, storage value, and cumulative reward
		self.stor_lev = 0.0
		self.stor_val = 0.0
		# self.cur_price = float(self.mean_std.loc[(self.mean_std["year"] == self.start_date.year) & (self.mean_std["month"] == self.start_date.month), "Mean"][0])
		self.cur_price = float(self.mean_std[self.time_step, 2])
		# self.cum_reward = 0.0

	def _get_spot_price_params(self) -> None:
	
		""" this function imports the price parameters from a json file "power_price_model.json"
			which is part of the package and to be supplied by the user of the environment
		"""
		
		# import json file as a dictionary
		file = os.path.join("envs", "power_price_model.json")
		# file = "power_price_model.json"
		with open(file) as f:
			d = json.load(f)
		
		# set individual price parameters
		self.prob_neg_jump = d["Prob.Neg.Jump"]
		self.prob_pos_jump = d["Prob.Pos.Jump"]
		self.exp_jump_distr = d["Exp.Jump.Distr"]  # lambda parameter of jump distribution
		self.est_mean_rev = d["Est.Mean.Rev"]
		self.est_mean = pd.DataFrame(d["Est.Mean"])
		self.est_mean["year"] = self.est_mean["year"].astype(int)
		self.est_mean["month"] = self.est_mean["month"].astype(int)
		self.est_std = pd.DataFrame(d["Est.Std"])
		self.est_std["month"] = self.est_std["month"].astype(int)

		# merge average vola to mean price
		mean_std = self.est_mean.merge(self.est_std, on="month")

		# generate numpy object containing mean and variance
		df = pd.DataFrame(data={"index": self.time_index, "month": self.time_index.dt.month, "year": self.time_index.dt.year})
		df.set_index("index", inplace=True)
		self.mean_std = df.merge(mean_std, on=["month", "year"]).to_numpy()  # column-order: ["month", "year", "mean", "std"]

	def _generate_jump(self, mean):
		if mean > self.cur_price:
			jump_occurrence = (np.random.uniform(0, 1, 1) <= self.prob_pos_jump / 100)  # TODO %timeit: vergleich mit if then block (Vermeidung Boolean Multiplication)
			jump = jump_occurrence * np.random.exponential(self.exp_jump_distr, 1)
		else:
			jump_occurrence = (np.random.uniform(0, 1, 1) <= self.prob_neg_jump / 100) # TODO: same as above
			jump = - (jump_occurrence * np.random.exponential(self.exp_jump_distr, 1))

		# print(f"Jump stattgefunden: {jump_occurrence}")
		# print(f"Jump size: {jump}")

		return jump

	def next_price(self) -> None:
		""" simulate next price increment and update current date
		"""

		# get mean and std of current month
		# month = self.cur_date.month
		# year = self.cur_date.year
		# month = self.mean_std[self.time_step, 0]
		# year = self.mean_std[self.time_step, 1]

		mean = self.mean_std[self.time_step, 2]
		std = self.mean_std[self.time_step, 3]

		# mean = float(self.mean_std.loc[(self.mean_std["year"] == year) & (self.mean_std["month"] == month), "Mean"]) # TODO: slice based on numpy index (keep track of starting and current month)
		# std = float(self.mean_std.loc[(self.mean_std["year"] == year) & (self.mean_std["month"] == month), "estimated.monthly.std"])
		# print(f"Mean: {mean}")
		# print(f"std: {std}")

		# generate noise
		noise = np.random.normal(loc=0, scale=std, size=1)
		# print(f"Noise: {noise}")

		jump = self._generate_jump(mean)

		price_inc = float(self.est_mean_rev * (mean - self.cur_price) + noise + jump)
		# print(f"price inc {price_inc}")
		# the price process was estimated on hourly data
		# as price increments are hourly, the "dt" part is set to one

		# update observations
		self.cur_price += price_inc
		# self.cur_date += timedelta(hours=1)
		self.time_step += 1

	def _storage_level_change(self, action):
		""" this function transforms the discrete action into a change in the level of the storage
		"""
		
		if action == 0:
			num_action = min(self.max_in, (self.max_stor_lev - self.stor_lev))
		elif action == 1:
			num_action = - min(abs(self.max_wd), self.stor_lev)
		else:
			num_action = 0

		# calculate new storage level after action
		new_stor_lev = self.stor_lev + self.stor_eff * num_action

		return num_action, new_stor_lev

	def _update_stor_val(self, num_action):
		""" this function updates the storage value after the new action
		"""
		if num_action > 0.0:
			self.stor_val = (self.stor_val * self.stor_lev + num_action * self.cur_price) / (self.stor_lev + num_action)

	def step(self, action):

		"""
		The agent takes a step in the environment.

		Parameters
		----------
		action : change in storage level [up, down, no_action] 
				(bang-bang property of these kind of problems well known, 
				hence discretization possible)

		Returns
		-------
		ob, reward, episode_over
			ob : List[float]
				an environment-specific object representing your observation of
				the environment.
			reward : float
				amount of reward achieved by the previous action. The scale
				varies between environments, but the goal is always to increase
				your total reward.
		"""
		
		# update observations
		num_action, new_stor_lev = self._storage_level_change(action)

		# update storage value
		self._update_stor_val(num_action)

		# update storage level
		if self.stor_lev == new_stor_lev:
			action = 2

			# calculate reward
			reward = self.penalty
			# self.cum_reward += reward
		else:
			self.stor_lev = new_stor_lev
			# calculate reward
			reward = - num_action * self.cur_price

		# update current price after the action was taken
		self.next_price()

		# generate list from observations for returning them to the agent
		year = self.mean_std[self.time_step, 1]
		month = self.mean_std[self.time_step, 0]

		observations = np.array([self.cur_price, self.stor_lev, self.stor_val])

		if (year == float(self.end_date.year)) & (month == float(self.end_date.month)):
			drop = True
		else:
			drop = False

		return observations, reward, drop, action  # self.cum_reward

	def reset(self):

		# reset cur_date to start_date
		self.cur_date = self.start_date
		self.time_step = 0

		# set storage level to zero
		self.stor_lev = 0.0

		# set storage value to zero
		self.stor_val = 0.0

		# set price to initial price
		self.cur_price = self.mean_std[self.time_step, 2]

		# set cum_reward to zero
		# self.cum_reward = 0.0

		observations = np.array([self.cur_price, self.stor_lev, self.stor_val])
		return observations

	def render(self, mode: str = "human", close: bool = False) -> None:
		return None

	def close(self):
		pass

if __name__ == "__main__":
	import gym
	# import cProfile
	import gym_energy_storage
	env = gym.make('energy_storage-v0')
	env.reset()
	env.step("up")
	# cProfile.run('env.next_price')

	import timeit

	result = timeit.timeit("env.next_price()", globals=globals(), number=5000) / 5000
	print(f"avg time required for _next_price: {result}")

	env.reset()

	result = timeit.timeit("env.step('up')", globals=globals(), number=5000) / 5000
	print(f"avg time required for step: {result}")

	# result = timeit.timeit('float(env.mean_std.loc[(env.mean_std["year"] == 2015) & (env.mean_std["month"] == 6), "Mean"])', globals=globals(), number=10000) / 10000
	# print(f"avg time required for step: {result}")
