import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
import json
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
		self.end_date = datetime.fromisoformat("2015-06-02")
		self.time_index = pd.Series(pd.date_range(start=self.start_date, end=self.end_date, freq="H"))
		self._get_spot_price_params()  # might be necessary to specify path here?
		self.observation_space = 5
		self.action_space = [0, 1, 2]  # ["up", "down", "cons"]
		self.penalty = -5
		self.cur_price = float(self.mean_std[self.time_step, 2])
		self.sim_prices = self.sim_price()
		length_window = 4
		self.mean_prices = self._mean_price(length_window)

		# storage specifics
		self.max_stor_lev = 10  # in MWh
		self.max_wd = -2.5  # in MW
		self.max_in = 1.5  # in MW
		self.stor_eff = 1.0  #  0.9  # 10% loss for each conversion
		self.round_acc = self.max_stor_lev / 1000

		# set initial parameters for price, storage level, storage value, and cumulative reward
		self.stor_lev = 0.0
		self.stor_val = 0.0


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
			jump_occurrence = (np.random.uniform(0, 1, 1) <= self.prob_pos_jump / 100)
			jump = jump_occurrence * np.random.exponential(self.exp_jump_distr, 1)
		else:
			jump_occurrence = (np.random.uniform(0, 1, 1) <= self.prob_neg_jump / 100)
			jump = - (jump_occurrence * np.random.exponential(self.exp_jump_distr, 1))

		return jump

	def _next_price(self, cur_price) -> None:
		""" simulate next price increment and update current date
		"""

		mean = self.mean_std[self.time_step, 2]
		std = self.mean_std[self.time_step, 3]

		# generate noise
		noise = np.random.normal(loc=0, scale=std, size=1)
		# print(f"Noise: {noise}")

		jump = self._generate_jump(mean)

		price_inc = float(self.est_mean_rev * (mean - cur_price) + noise + jump)

		self.cur_price = price_inc + cur_price

	def sim_price(self):

		"""
		this function constructs the series of simulated prices
		"""

		price_list = []
		price_list.append(self.cur_price)
		for t in range(len(self.time_index)):
			self._next_price(self.cur_price)
			price_list.append(self.cur_price)

		self.cur_price = float(self.mean_std[self.time_step, 2])  # reset self.cur_price to initial value
		return np.array(price_list)

	def _mean_price(self, length_window):
		v = sliding_window_view(self.sim_prices, length_window)
		moving_averages = v.mean(axis=-1)
		num_first_values = self.sim_prices.shape[0] - moving_averages.shape[0]
		first_values = []
		for val in range(num_first_values):
			first_values.append(self.sim_prices[0:(val+1)].mean())
		return np.append(first_values, moving_averages)
		
	def _storage_level_change(self, action):
		""" this function transforms the discrete action into a change in the level of the storage
		"""
		def trunc_action(step_size):
			if abs(step_size) < self.round_acc:
				return 0.0
			else:
				return float(step_size)

		if action == 0:
			num_action = min(self.max_in, (self.max_stor_lev - self.stor_lev))
			num_action = trunc_action(num_action)
		elif action == 1:
			num_action = - min(abs(self.max_wd), self.stor_lev)
			num_action = trunc_action(num_action)
		else:
			num_action = 0.0

		# calculate new storage level after action
		if num_action > 0.0:
			new_stor_lev = self.stor_lev + self.stor_eff * num_action  # apply efficiency factor to injection
		else:
			new_stor_lev = self.stor_lev + num_action

		return num_action, new_stor_lev

	def _update_stor_val(self, num_action):
		""" this function updates the storage value after the new action
		"""
		if num_action > 0.0:
			self.stor_val = (self.stor_val * self.stor_lev + num_action * self.stor_eff * self.cur_price) / (self.stor_lev + num_action * self.stor_eff)

	def step(self, action):

		"""
		The agent takes a step in the environment.
		"""

		# update observations
		num_action, new_stor_lev = self._storage_level_change(action)

		# update storage value
		self._update_stor_val(num_action)

		# update storage level and calculate reward (using old price)
		if new_stor_lev == self.stor_lev:
			# action = 2
			# calculate reward
			reward = self.penalty
			action = 2
		else:
			self.stor_lev = new_stor_lev
			# calculate reward
			reward = - num_action * self.cur_price

		# update time period and price
		self.time_step += 1
		self.cur_price = self.sim_prices[self.time_step]


		# generate list from observations for returning them to the agent
		observations = np.array([self.time_step, self.cur_price, self.stor_lev, self.stor_val, self.mean_prices[self.time_step]])

		if (self.cur_date.year == self.end_date.year) & (self.cur_date.month == self.end_date.month) & (self.cur_date.day == self.end_date.day):
			drop = True
		else:
			drop = False
			self.cur_date += timedelta(hours=1)

		return observations, reward, drop, action

	def reset(self):

		# reset cur_date to start_date
		self.cur_date = self.start_date
		self.time_step = 0

		# set storage level to zero
		self.stor_lev = 0.0

		# set storage value to zero
		self.stor_val = 0.0

		# set price to initial price
		self.cur_price = float(self.mean_std[self.time_step, 2])

		# simulate new prices
		self.sim_prices = self.sim_price()

		observations = np.array([self.time_step, self.cur_price, self.stor_lev, self.stor_val, self.cur_price])
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
	# cProfile.run('env.next_price')

	import timeit

	# result = timeit.timeit("env.next_price()", globals=globals(), number=5000) / 5000
	# print(f"avg time required for _next_price: {result}")

	env.reset()

	# result = timeit.timeit("env.step('up')", globals=globals(), number=5000) / 5000
	# print(f"avg time required for step: {result}")

	# result = timeit.timeit('float(env.mean_std.loc[(env.mean_std["year"] == 2015) & (env.mean_std["month"] == 6), "Mean"])', globals=globals(), number=10000) / 10000
	# print(f"avg time required for step: {result}")
