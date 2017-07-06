"""
This file runs optimization algorithms which are part of the 'scipy' package. The
purpose is to find the set of parameters (found in variables.py) for our model
that minimize/maximize certain error metrics.
This file uses the 'Nelder-Mead' and 'L-BFGS-B' optimization methods. These are
both suitable due to the complexity of our model, with the latter allowing
constraints placed on the parameters.
"""

from data_handler import Data
from elo import ELO
from predictions import Predictions
from errors import Errors
import variables

from collections import OrderedDict
import numpy as np
from scipy import optimize
import os
import time
import copy


ALL_ROUNDS = range(1,27)
# We start by declaring some error metrics that will be used as cost functions for
# our optimization.

# This corresponds to the Average Absolute Error
def error_margin(x, train_years, rounds, round_type, elo, data):

    variables.var_dict = {'k': x[0], 'R': x[1],
                        'HGA': x[2], 'm': 0.1, 'p': 1}

    d = Predictions().predict_results(elo, train_years, rounds, round_type, data)
    result = Errors().abs_error(d)[0]
    return result

# Corresponds to percentage of matches predicted incorrectly (Win/Loss/Draw)
def error_prob(x, train_years, rounds, round_type, elo, data):

    variables.var_dict = {'k': [x][0], 'R': [x][1],
                        'HGA': [x][2], 'm': 0.1, 'p': 1}

    d = Predictions().predict_results(elo, train_years, rounds, round_type, data)
    percent_correct = Errors().WL_error(d)[0]
    percent_wrong = 1 - percent_correct
    return percent_wrong

# Corresponds to the log-likelihood function (derived from the Maximum
# Likelihood Estimation
def error_log(x, train_years, rounds, round_type, elo, data):

    variables.var_dict = {'k': [x][0], 'R': [x][1],
                        'HGA': [x][2], 'm': 0.1, 'p': 1}

    d = Predictions().predict_results(elo, train_years, rounds, round_type, data)
    result = Errors().log_error2(d,True)[0]
    return result


# This function does all the work (running the optimization and writing the results
# to a text file)
def run_optimization(error_function, original_guess, method, train_years,
            rounds, round_type, test_set, bounds=None):
    start_time = time.time()
    # Creates class instances that will be passed to other functions.
    data = Data()
    elo = ELO()

    bounds = []
    options = {}
    # This runs the optimization and saves the results to the variable 'a'.
    # Bounds and options are only available for the 'L-BFGS-B' method
    a = optimize.minimize(lambda x: error_function(x, train_years, rounds, round_type,
            elo, data), original_guess, method=method,bounds=bounds,options=options)

    # Changes the parameters in our program to the optimization output. This is done
    # so we can calculate the errors on the training and test sets.
    variables.var_dict = {'k': a['x'][0], 'R': a['x'][1],
                        'HGA': a['x'][2], 'm': 0.1, 'p': 1}

    # Calculates the errors of this set of constants on the training set using
    # various error metrics from the Errors module
    n = ELO()
    p_train = Predictions().predict_results(n, train_years, rounds, round_type, data)
    error_margin_train = Errors().abs_error(p_train)[0]
    error_prob_train = Errors().WL_error(p_train)[0]
    error_log_train = Errors().log_error(p_train)[0]

    # Update the ELO ratings from the training years
    new = n.get_new_elo(train_years, rounds, round_type, data)
    # Calculates errors on test set
    p_test = Predictions().predict_results(new, test_set, rounds, round_type, data)
    error_margin_test = Errors().abs_error(p_test)[0]
    error_prob_test = Errors().WL_error(p_test)[0]
    error_log_test = Errors().log_error(p_test)[0]

    # Now we put information from the optimzation in a dictionary to be
    # written to a file.
    d = OrderedDict()
    d.update({"Error Type": error_function}); d.update({"Method": method})
    d.update({"Message": a['message']})
    d.update({'Function Evaluations': a['nfev']})
    d.update({"Time taken": time.time()-start_time})
    d.update({"Starting Values": original_guess})
    d.update({"Minimum Error evaluation": a['fun']})
    d.update({"Margin Error on Training Set": error_margin_train})
    d.update({"Correct Predictions for Training Set": error_prob_train})
    d.update({"Log Error for Training Set": error_log_train})
    d.update({"Training Years": train_years})
    d.update({"Rounds": rounds})
    d.update({"Round Type": round_type});
    d.update({"k": a['x'][0]})
    d.update({"R": a['x'][1]})
    d.update({"HGA": a['x'][0]})
    d.update({"Test Years": test_set})
    d.update({"Margin Error on Test Set": error_margin_test})
    d.update({"Correct Predictions on Test Set": error_prob_test})
    d.update({"Log Error on Test Set": error_log_test})

    # Writes the above dictionary to the file scipy_optimize/test.txt.
    with open(os.path.join(os.path.dirname(__file__),
            '../scipy_optimize/test.txt'), "a") as f:
        f.write("\n")
        for key in d:
            f.write(key + ": " + str(d[key]) + "\n")


if __name__ == "__main__":

    xmin = [1, 0.1, 0]
    xmax = [500, 0.9, 5]
    # rewrite the bounds in the way required by L-BFGS-B
    param_bounds = [(low, high) for low, high in zip(xmin, xmax)]

    try:
        run_optimization(error_margin, [3.2], "Nelder-Mead", [2007, 2008, 2009, 2010,
                2011, 2012, 2013], ALL_ROUNDS, "all", [2014, 2015, 2016])
    except ValueError:
        print "Our ELO's got too out of control"
        raise

    try:
        run_optimization(error_margin, [3.2], "L-BFGS-B", [2010, 2011, 2012, 2013],
                ALL_ROUNDS, "all", [2014, 2015, 2016], bounds=param_bounds)
    except ValueError:
        print "Our ELO's got too out of control"
        raise
