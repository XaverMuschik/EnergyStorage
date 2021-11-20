import gym
from gym import error, spaces, utils
from gym.utils import seeding

class EnergyStorageExtrahardEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
  	# define relative path to open power price model
  	# self.path_prices = ...
  
  	# call function to import spot price model
  	# set initial parameters for price, storage level, and storage value 	
    	pass
    	
  def _get_spot_price_params()
  	# import json file
  	# brush price data for further use
  	
  	
  def step(self, action):
  	# update storage level
  	# update storage value ?
  	# update current price
    	pass
    	
  def reset(self):
  	# set storage level to zero
  	# set storage value to zero
  	# set price to initial price
    	pass
    	
  def render(self, mode='human'):
     	pass
     	
  def close(self):
    	pass
