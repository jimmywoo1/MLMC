from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

class PricingModel(ABC):
    '''
    abstract base class for options pricing models

    attributes
        _init_price: initial price
        _init_vol: initial volatility
    '''
    def __init__(self,
                 init_price: float,
                 init_vol: float):
        '''
        constructor for the PricingModel class

        args:
            init_price: initial price
            init_vol: initial volatility
        '''
        self._init_price = init_price
        self._init_vol = init_vol

    @abstractmethod
    def two_level_mc(self,
                     r: float,
                     T: float,
                     M: int,
                     level: int,
                     num_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        performs two-level monte carlo simulations to generate fine/coarse 
        pair of paths based on same brownian paths

        args:
            r: risk-free rate
            T: time to maturity
            M: base constant for discretization
            level: level for MLMC
            num_paths: number of sample paths
        
        returns:
            tuple of (fine_path, coarse_path)
        '''
        pass
