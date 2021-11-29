import unittest
import gym_energy_storage
import gym
from datetime import timedelta

class TestEnv(unittest.TestCase):

    def test_time_increment(self):
        env = gym.make('energy_storage-v0')
        env.reset()
        self.assertEqual(env.cur_date, env.start_date)
        env.step("up")
        new_date = env.start_date + timedelta(hours=1)
        self.assertEqual(env.cur_date, new_date)



if __name__ == "__main__":
    unittest.main()