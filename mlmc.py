from typing import Tuple

import numpy as np

from pricing_models.pricing_model import PricingModel
from options.option import Option
from utils.util import batch_mean_update, batch_var_update

def MLMC(pricing_model: PricingModel, 
         option: Option,
         r: float,
         T: int,
         M: int,
         K: float,
         eps: float,
         default_num_paths: int=10**4) -> Tuple[np.ndarray, np.ndarray]:
    '''
    implementation of multilevel monte carlo 
    (https://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf)

    args:
        pricing_model: model for pricing options
        option: option object
        r: risk-free rate
        T: time to maturity
        M: base constant for discretization
        K: strike price of option
        eps: target error
        default_num_paths: default number of simulation paths
    
    returns:
        tuple of (mean, variance) of MLMC estimate of option payoff
    '''
    L = 2
    converged = False

    while not converged:
        mean_list = []
        var_list = []
        h_list = []
        num_paths = []

        # initial estimates
        for level in range(L + 1):
            prices = pricing_model.two_level_mc(r, T, M, level, default_num_paths)

            if level == 0:
                payoff_f = option.payoff(prices, K)
                payoff_c = np.zeros_like(payoff_f)
            else:
                payoff_f = option.payoff(prices[0], K)
                payoff_c = option.payoff(prices[1], K)

            mean_list.append(np.mean(payoff_f - payoff_c))
            var_list.append(np.var(payoff_f - payoff_c))
            h_list.append(T / np.power(M, level))
            num_paths.append(len(payoff_f))

        running_sum_vh = np.sum([np.sqrt(v/h) for v, h in zip(var_list, h_list)])

        for level in range(L + 1):
            n_optimal = int(np.ceil(0.5 / eps ** 2 * np.sqrt(var_list[level] * h_list[level]) * running_sum_vh))
            
            # additional sample paths
            if n_optimal > num_paths[level]:
                num_update = n_optimal - num_paths[level]
                prices = pricing_model.two_level_mc(r, T, M, level, num_update)
                payoff_diff = option.payoff(prices[0], K) - option.payoff(prices[1], K)
                mean = np.mean(payoff_diff)
                var = np.mean(payoff_diff)

                mean_list[level] = batch_mean_update(num_paths[level], 
                                                    mean_list[level],
                                                    num_update,
                                                    mean)
                var_list[level] = batch_var_update(num_paths[level],
                                                mean_list[level],
                                                var_list[level],
                                                num_update,
                                                mean,
                                                var)

        if np.abs(mean_list[L] - mean_list[L - 1] / M) < 1/np.sqrt(2) * (M **2 - 1) * eps:
            converged = True
        else:
            L += 1
        
    return np.mean(mean_list), np.mean(var_list)