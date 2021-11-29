import unittest
import gym_energy_storage
import gym
from datetime import timedelta
from datetime import datetime
import numpy as np


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

    # def test_price_sim(self):
    #     np.random.RandomState(1304)
    #     env = gym.make('energy_storage-v0')
    #     env.reset()
    #     sim_price_1 = env.sim_prices
    #
    #     np.random.RandomState(1304)
    #     env.reset()
    #     sim_price_2 = env.sim_prices
    #     self.assertTrue(np.allclose(sim_price_1, sim_price_2))

    def testNextStateUpUp(self):
        env = gym.make('energy_storage-v0')
        env.reset()

        # step 1
        obs_act, reward_act, drop_act, action_act = env.step(0)
        obs_expected = np.array([env.start_date.day,
                                 env.start_date.month,
                                 env.start_date.year,
                                 env.sim_prices[1],  # current price
                                 env.max_in * env.stor_eff,  # storage level
                                 env.sim_prices[0]  # storage value
                                 ])
        self.assertTrue(np.allclose(obs_act, obs_expected))
        self.assertEqual(reward_act, -env.max_in * env.sim_prices[0])
        self.assertEqual(drop_act, False)
        self.assertEqual(action_act, 0)

        # step 2
        obs_act, reward_act, drop_act, action_act = env.step(0)
        obs_expected = np.array([env.start_date.day,
                                 env.start_date.month,
                                 env.start_date.year,
                                 env.sim_prices[2],
                                 2 * env.max_in * env.stor_eff,
                                 (env.sim_prices[0] + env.sim_prices[1]) / 2
                                 ])
        self.assertTrue(np.allclose(obs_act, obs_expected))
        self.assertAlmostEqual(reward_act, -env.max_in * env.sim_prices[1])
        self.assertEqual(drop_act, False)
        self.assertEqual(action_act, 0)

    def testNextStateUpDown(self):
        env = gym.make('energy_storage-v0')
        env.reset()

        # step up 1
        obs_act, reward_act, drop_act, action_act = env.step(0)
        stor_level = env.stor_lev  # save old storage level after step up

        # step down
        obs_act, reward_act, drop_act, action_act = env.step(1)
        obs_expected = np.array([env.start_date.day,
                                 env.start_date.month,
                                 env.start_date.year,
                                 env.sim_prices[2],
                                 max(env.max_in * env.stor_eff + env.max_wd, 0),
                                 env.sim_prices[0]  # storage value unchanged after wd
                                 ])
        self.assertTrue(np.allclose(obs_act, obs_expected))
        reward_expected = min(-env.max_wd, stor_level) * env.sim_prices[1]
        self.assertAlmostEqual(reward_act, reward_expected)
        self.assertEqual(drop_act, False)
        self.assertEqual(action_act, 1)

    def testNextStateCons(self):
        env = gym.make('energy_storage-v0')
        env.reset()

        # step cons
        obs_act, reward_act, drop_act, action_act = env.step(2)
        obs_expected = np.array([env.start_date.day,
                                 env.start_date.month,
                                 env.start_date.year,
                                 env.sim_prices[1],
                                 0,
                                 0
                                 ])
        self.assertTrue(np.allclose(obs_act, obs_expected))
        reward_expected = env.penalty
        self.assertAlmostEqual(reward_act, reward_expected)
        self.assertEqual(drop_act, False)
        self.assertEqual(action_act, 2)

    def testEndOfPeriod(self):
        """ this test checks if the break criterion is correctly met """
        env = gym.make('energy_storage-v0')
        env.reset()
        env.end_date = datetime.fromisoformat("2015-06-02")

        for t in range(len(env.time_index)-1):
            _, _, drop_act, _ = env.step(0)
            test = env.cur_date
            self.assertEqual(drop_act, False)
        print(env.cur_date)
        _, _, drop_act, _ = env.step(0)
        self.assertEqual(drop_act, True)

if __name__ == "__main__":
    unittest.main()