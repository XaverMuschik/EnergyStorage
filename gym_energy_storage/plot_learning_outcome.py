import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import time

from matplotlib.table import Table

class plot_learning_result():

    def __init__(self, stor_val, mean_price, model, len_period, max_stor, time_step=300, stor_min=0, stor_max=10, price_min=0, price_max=100):
        """ initialize class with:
            - time-step
            - min and max storage levels to be plotted
            - min and max price to be plotted
            - storage value
            - color scheme
            - model for calculation of probs for choosing each action (up/ down/ cons)
        """

        self.time_step = time_step
        self.stor_grid = np.arange(start=stor_min, stop=stor_max, step=1)
        self.price_grid = np.arange(start=price_min, stop=price_max, step=10)
        self.stor_val = stor_val
        self.mean_price = mean_price
        self.colors = {'up': 'forestgreen', 'cons': 'gold', 'down': 'darkorange'}
        self.model = model
        self.len_period = len_period
        self.max_stor = max_stor

    def main(self):

        data = pandas.DataFrame(index=self.price_grid, columns=self.stor_grid).fillna(0)
        self.checkerboard_table(data)
        plt.show()

    def normalize(self, state):
        """ normalizes observations to interval [0,1] """
        def scale(min_arg, max_arg, arg):
            return (arg - min_arg) / (max_arg - min_arg)

        normalized = state.copy()
        normalized[0, 0] = scale(0.0, self.len_period, state[0, 0])  # normalize time step
        normalized[0, 1] = scale(-200.0, 200.0, state[0, 1])  # normlize price
        normalized[0, 2] = scale(0.0, self.max_stor, state[0, 1])  # normlize storage level
        normalized[0, 3] = scale(-200.0, 200.0, state[0, 3])  # normalize storage value
        normalized[0, 4] = scale(-100.0, 100.0, state[0, 4])  # normalize mean price
        return normalized

    def checkerboard_table(self, data, fmt='{:.0f}'):
        fig, ax = plt.subplots()
        fig.set_figheight(8)  # scaling of graphic is done here!
        fig.set_figwidth(15)  # scaling of graphic is done here!
        ax.set_axis_off()
        # ax.title.set_text('Plot probs for up/ down/ cons. y: price')  # add title
        ax.xaxis.set_label_position('top')
        # ax.set_xlabel('storage level')
        ax.set_title('Plot probs for up/ down/ cons. x: storage level, y: price')
        # ax.set_ylabel('price')
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = data.shape
        width, height = 1.0 / ncols, 1.0 / nrows / 2.0

        def calc_values(time_step, price, stor_vol):
            observations = np.asarray([time_step, price, stor_vol, self.stor_val, self.mean_price]).reshape(1, -1)
            return self.model(self.normalize(observations)).numpy()[0]

        # Add cells
        for stor_lev in data.columns:
            for price_lev in data.index:
                # apply model to values - here: calculate norm as simplified example

                prob_up, prob_down, prob_cons = calc_values(self.time_step, price_lev, stor_lev)  # model will return three values for the probs
                probs = np.array([prob_up, prob_down, prob_cons])
                val = np.argmax(probs)  # identify action with highest probability
                vals = f'{prob_up:.2f}' + " / " + f'{prob_down:.2f}' + " / " + f'{prob_cons:.2f}'

                # color based on highest value and dict provided as input to function
                if val == 0:
                    color = self.colors["up"]
                elif val == 1:
                    color = self.colors["down"]
                else:
                    color = self.colors["cons"]

                # add cell to graphic
                tb.add_cell(price_lev, stor_lev, width, height, text=vals,
                            loc='center', facecolor=color)

        # Row Labels...
        for i in data.index:
            tb.add_cell(i, -1, width, height, text=fmt.format(i), loc='right',
                        edgecolor='none', facecolor='none')
        # Column Labels...
        for j in data.columns:
            tb.add_cell(-1, j, width, height / 2, text=fmt.format(j), loc='center',
                        edgecolor='none', facecolor='none')
        ax.add_table(tb)
        # return fig
        name = "outcome_learning_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        file = os.path.join("Figures_Outcome_Learning", name)
        fig.savefig(file)
        plt.close(fig)