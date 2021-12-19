import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
from datetime import timedelta


class storageValLSM():

    def __init__(self):

        np.random.seed(1405)

        self.max_wd = -2.0
        self.max_in = 1.0
        self.max_stor = 10.0
        self.min_stor = 0.0
        self.grid_size = 0.5  # step size of volume grid
        self.grid_steps = int((self.max_stor - self.min_stor) / self.grid_size) + 1  # add one to include zero and maximum
        self.number_price_paths = 50

        self.start_date = datetime.fromisoformat("2015-06-01")  # relevant for price simulation
        self.cur_date = self.start_date  # keep track of current date
        self.time_step = 0  # variable used for slicing mean and var values
        self.end_date = datetime.fromisoformat("2015-07-01")
        self.time_index = pd.Series(pd.date_range(start=self.start_date, end=self.end_date, freq="H"))
        self.time_steps = len(self.time_index)

        self.acc_payoff = np.zeros(((self.time_steps+1), self.grid_steps, self.number_price_paths))

        self.payoff_per_period = np.zeros(((self.time_steps+1), self.grid_steps, self.number_price_paths))
        self.best_actions_per_period = np.zeros(((self.time_steps+1), self.grid_steps, self.number_price_paths))

        # simulate prices
        self._get_spot_price_params()
        self.cur_price = float(self.mean_std[0, 2])
        self.prices = np.zeros((self.time_steps+1, self.number_price_paths))
        self.sim_prices()
        print("initializatin finished")

    def sim_prices(self):

        """ this function generates a matrix of """

        for path in range(self.number_price_paths):
            self.prices[:, path] = self._sim_price_per_path()

    def _get_spot_price_params(self) -> None:

        """ this function imports the price parameters from a json file "power_price_model.json"
            which is part of the package and to be supplied by the user of the environment
        """

        # import json file as a dictionary
        file = os.path.join("power_price_model.json")
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
        df = pd.DataFrame(
            data={"index": self.time_index, "month": self.time_index.dt.month, "year": self.time_index.dt.year})
        df.set_index("index", inplace=True)
        self.mean_std = df.merge(mean_std,
                                 on=["month", "year"]).to_numpy()  # column-order: ["month", "year", "mean", "std"]

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

        jump = self._generate_jump(mean)

        price_inc = float(self.est_mean_rev * (mean - cur_price) + noise + jump)

        self.cur_price = price_inc + cur_price

    def _sim_price_per_path(self):

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

    def _regress_cont_val(self, Y_t1, S_t1):
        """
        regresses the accumulated CFs in t+1 on the spot price of this period
        FOR A FIXED VOLUME LEVEL
        used to predict the continuation value in t

        params:
        Y_t1: vector of accumulated cash flows in t1 (for each price path)
        S_t1: vector of spot prices in t1 (for each price path)
        """
        model_coefs = np.polyfit(S_t1, Y_t1, 3)

        return model_coefs

    def _predict_cont_val(self, S_t1, model_coefs):
        """ predicts the continuation value for a given volume level and spot price """

        return np.polyval(model_coefs, S_t1)
        # ToDo: check if this yiels expected result

    def _identify_best_action(self, vol_level, S_t, C_t):
        """
        this function identifies the best action for a given volume level

        params:
        C_t: vector of continuation values in t (for each price path)
        S_t: vector of spot prices in t (for each price path)

        returns two vectors:
            - a vector containing the payoff associated with the best action
            - a vector containing the best actions
        """

        # determine action space
        if vol_level + self.max_in > self.max_stor:
            max_in = self.max_stor - vol_level
        else:
            max_in = self.max_in

        if vol_level + self.max_wd < 0.0:
            max_wd = - vol_level
        else:
            max_wd = self.max_wd

        best_action = np.repeat(max_wd, len(S_t))
        value = np.repeat(-1e10, len(S_t))
        payoff = np.repeat(0.0, len(S_t))

        # for action in range(max_wd, max_in, self.grid_size):
        vol_index = vol_level / self.grid_size
        action = max_wd
        while action <= max_in:

            # determine how much each action adds to the vol_index
            action_index = action / self.grid_size
            vol_index_new = vol_index + action_index

            value_new = C_t[vol_index_new,:] - action * S_t
            payoff_new = - action * S_t

            # update best action
            best_action[value_new > value] = action  # np.repeat(action, len(S_t))[value_new > value]

            # update payoff
            payoff[value_new > value] = payoff_new[value_new > value]

            # update value
            value[value_new > value] = value_new[value_new > value]  # ToDo: debug whether or not this works!

            # increase action for next iteration
            action += self.grid_size

        return payoff, best_action

    def _update_accumulated_cash_flows(self, timestep, vol_index, payoff, best_action):

        """
        this function updates the accumulated cash flows of a given period for a given storage level
        """

        # ToDo: acc_payoff needs to be class attribute!
        # this function does not work in a vectorized fashion :(

        for path in range(self.number_price_paths):
            # get next periods accumulated cf based on best action
            vol_index_change = int(best_action[path] / self.grid_size)
            acc_payoff_next_period = self.acc_payoff[(timestep + 1), (vol_index + vol_index_change), path]

            # update the acc_cf for time, volume, path
            self.acc_payoff[timestep, vol_index, path] = payoff[path] + acc_payoff_next_period

    def backwards_iteration(self):

        for time in range(self.time_steps-1, 0, -1):

            # loop over volume index per timestep to generate continuation values for the volume grid
            vol = self.min_stor
            vol_index = 0
            continuation_values = np.zeros((self.grid_steps, self.number_price_paths))
            while vol <= self.max_stor:

                # 1. regression for prediction of cont values based on current prices and accumulated CFs
                model = self._regress_cont_val(self.acc_payoff[time+1, vol_index, :], self.prices[time+1, :])

                # 2. predict continuation values
                continuation_values[vol_index,:] = self._predict_cont_val(self.prices[time, :], model)

                # increase volume for next iteration
                vol += self.grid_size
                vol_index += 1

            # using the continuation values-grid, identify best action and derive payoffs
            vol = self.min_stor
            vol_index = 0
            while vol <= self.max_stor:

                # 3. identify best action and corresponding cash flow
                    ## ToDo: continuation values for each resulting volume level need to be provided to identify best action
                    ## (otherwise, max withdrawal is always best!!!)
                payoff, best_action = self._identify_best_action(vol, self.prices[time, :], continuation_values)

                # 4. update matrix containing accumulated cash flows
                self._update_accumulated_cash_flows(time, vol_index, payoff, best_action)

                # 5. store best action and payoff for model evaluation
                self.payoff_per_period[time, vol_index, :] = payoff
                self.best_actions_per_period[time, vol_index, :] = best_action

                # increase volume for next iteration
                vol += self.grid_size
                vol_index += 1

        self.storage_value = np.mean(self.acc_payoff[1, 0, :])
        print(f"Storage value: {self.storage_value}")
        return self.storage_value


if __name__ == "__main__":
    test = storageValLSM()
    test.backwards_iteration()