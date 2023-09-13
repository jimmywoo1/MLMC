from typing import Union, Tuple
import numpy as np
from pricing_model import PricingModel

class HestonModel(PricingModel):
    '''
    Heston pricing model using Euler-Maruyama discretization 
    with full truncation

    attributes
        _init_price: initial price
        _init_vol: initial volatility
        _kappa: mean reversion rate
        _theta: expected value of volatility
        _sigma: volatility term for modelling dV
    '''
    def __init__(self,
                 init_price: float,
                 init_vol: float,
                 kappa: float,
                 theta: float,
                 sigma: float):
        '''
        constructor for the HestonModel class

        args:
            init_price: initial price
            init_vol: initial volatility
            kappa: mean reversion rate
            theta: expected value of volatility
            sigma: volatility term for modelling dV
        '''
        super().__init__(init_price, init_vol)
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma

    def two_level_mc(self,
                     r: float,
                     T: float,
                     M: int,
                     level: int,
                     num_paths: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        '''
        performs two-level monte carlo simulations to generate fine/coarse 
        pair of paths based on same brownian paths using the Heston model

        args:
            r: risk-free rate
            T: time to maturity
            M: base constant for discretization
            level: level for MLMC
            num_paths: number of sample paths
        
        returns:
            tuple of (fine_path, coarse_path)
        '''         
        price_c = None

        # timesteps
        hf = T / np.power(M, level)
        num_steps_fine = int(1 / hf)
        
        # brownian increments for fine path
        dW_f, dZ_f = np.random.normal(0, np.sqrt(hf), (num_paths, num_steps_fine, 2)).T
        vol_f = np.full((num_paths, num_steps_fine), fill_value=self._init_vol)
        price_f = np.full((num_paths, num_steps_fine), fill_value=self._init_price)

        if level == 0:
            return price_f

        # timesteps
        hc = T / np.power(M, level - 1)
        num_steps_coarse = int(1 / hc)

        # brownian increments for coarse path
        dW_c = dW_f.reshape(-1, M, num_paths).sum(axis=1)
        dZ_c = dZ_f.reshape(-1, M, num_paths).sum(axis=1)
        vol_c = np.full((num_paths, num_steps_coarse), fill_value=self._init_vol)
        price_c = np.full((num_paths, num_steps_coarse), fill_value=self._init_price)

        for i in range(num_steps_coarse):
            # increments for fine path
            for j in range(M):
                idx = i*M + j
                vol_f[:, idx] += self._kappa * (self._theta - np.maximum(vol_f[:, idx], 0)) * hf + self._sigma * np.sqrt(np.maximum(vol_f[:, idx], 0)) * dZ_f[idx]
                price_f[:, idx] += hf * r * price_f[:, idx] + np.maximum(vol_f[:, idx], 0) * price_f[:, idx] * dW_f[idx]

            # increments for coarse path
            if level > 0:
                vol_c[:, i] += self._kappa * (self._theta - np.maximum(vol_c[:, i], 0)) * hc + self._sigma * np.sqrt(np.maximum(vol_c[:, i], 0)) * dZ_c[i]
                price_c[:, i] += hc * r * price_c[:, i] + np.maximum(vol_c[:, i], 0) * price_c[:, i] * dW_c[i]

        paths = (price_f, price_c) if level > 0 else price_f

        return paths 