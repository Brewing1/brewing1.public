
"""
This file enables us to plot learning curves to investigate the bias/variance
in our models. We fix our test set as 2014-2016, and progressively increase the
number of years used in the training set starting from just 2013.
"""

import variables
from data_handler import Data
from elo import ELO
from predictions import Predictions
from errors import Errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def error_abs(x, years, rounds, round_type, elo, data):
    variables.var_dict = {'k': x[0], 'R': x[1],
                        'HGA': x[2], 'm': 0, 'p': 1}

    d = Predictions().predict_results(elo, years, rounds, round_type, data)
    result = Errors().abs_error(d)[0]
    return result

# This function gives us the training and cross validation errors of a particular
# model, as well as the optimized paramter values
def get_errors_k():
    CV = [2014, 2015, 2016]
    rounds = Data.ALL_ROUNDS
    original_guess = [0.05, 0.5, 3]

    J_train, J_CV = ([] for i in range(2))

    i = [2013]
    # Loop through the training sets, adding a year each time
    while 2006 not in i:
        d = Data(odds=False)
        e = ELO()
        # Find the optimal paramter values for this particular training set
        a = optimize.minimize(lambda x: error_abs(x, i, rounds, 'All', e, d),
            original_guess, method='Nelder-Mead')

        variables.var_dict = {'k': a['x'][0], 'R': a['x'][1],
                            'HGA': a['x'][2], 'm': 0, 'p': 1}

        # Before calling the predict_results function we update the ELO
        # values using the previous few years. This will give us more accurate results
        new_elo1 = e.get_new_elo([i[0]-4,i[0]-3,i[0]-2,i[0]-1], data=d)
        # Predict results on the training set
        p1 = Predictions().predict_results(new_elo1,i,round_type="All", data=d, odds=False)
        J_train.append(Errors().abs_error(p1)[0])

        f = ELO()
        new_elo2 = f.get_new_elo(i,data=d)
        # Predict the results on the test set
        p2 = Predictions().predict_results(new_elo2,CV,round_type="All", data=d, odds=False)
        J_CV.append(Errors().abs_error(p2)[0])

        i.insert(0, i[0]-1)

    with open('learning_curves.txt', 'wb') as f:
        for i in J_train:
            f.write(str(i) + ',')
        f.write('\n')
        for j in J_CV:
            f.write(str(j) + ',')

# This function plots the learning curves in the file lcurves.png
def plot_curves():
    with open('learning_curves.txt', 'r') as f:
        a = f.readline().split(',')
        b = f.readline().split(',')
        del a[-1]
        del b[-1]
        J_train = [float(x) for x in a]
        J_CV = [float(x) for x in b]

    x = [i+1 for i in range(7)]

    plt.plot(x, J_train)
    plt.plot(x, J_CV)

    plt.title('Learning Curves with Parameters k, R, HGA')
    plt.xlabel('Number of Training Years')
    plt.ylabel('Average Absolute Error')
    plt.legend(['J(train)', 'J(CV)'], loc='upper left')

    plt.savefig('lcurves.png')

if __name__ == "__main__":
    get_errors_k()
    plot_curves()
