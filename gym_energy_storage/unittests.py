import unittest
import gym_energy_storage
import gym
from datetime import timedelta
from datetime import datetime

class TestEnv(unittest.TestCase):

    def test_time_increment(self):
        env = gym.make('energy_storage-v0')
        env.reset()
        self.assertEqual(env.cur_date, env.start_date)
        env.step("up")
        new_date = env.start_date + timedelta(hours=1)
        self.assertEqual(env.cur_date, new_date)

    def test_initial_params(self):
        env = gym.make('energy_storage-v0')
        env.reset()

        self.assertEqual(env.start_date, datetime.fromisoformat("2015-06-01"))  # date is hardcoded
        self.assertEqual(env.cur_date, datetime.fromisoformat("2015-06-01"))  # date is hardcoded
        self.assertEqual(env.time_step, 0)
        self.assertEqual(env.end_date, datetime.fromisoformat("2015-07-01"))  # date hardcoded
        self.assertEqual(env.observation_space, 6)
        self.assertEqual(env.action_space, ["up", "down", "cons"])
        self.assertEqual(env.penalty, -0.5)
        self.assertEqual(env.cur_price, float(env.mean_std[env.time_step, 2]))

        # storage specifics
        self.assertEqual(env.max_stor_lev, 10)  # in MWh
        self.assertEqual(env.max_wd, -2.5)  # in MW
        self.assertEqual(env.max_in, 1.5)  # in MW
        self.assertEqual(env.stor_eff, 0.9)  # 10% loss for each conversion
        self.assertEqual(env.round_acc, env.max_stor_lev / 1000)

        # set initial parameters for price, storage level, storage value, and cumulative reward
        self.assertEqual(env.stor_lev, 0.0)
        self.assertEqual(env.stor_val, 0.0)


if __name__ == "__main__":
    unittest.main()