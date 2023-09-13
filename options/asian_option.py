import numpy as np
from option import Option

class AsianOptions(Option):
    '''
    class for arithmetic Asian options

    attributes
        _r: risk free rate
        _T: time to maturity
    '''
    def __init__(self,
                 risk_free: float,
                 time_to_maturity: float):
        '''
        constructor for the AsianOption class

        args:
            r: risk free rate
            T: time to maturity
        '''
        super().__init__(risk_free, time_to_maturity)

    def payoff(self,
               prices: list,
               strike_price: float,
               is_call: bool=True) -> float:
        '''
        computes the payoff of the current option

        args:
            prices: time series of prices
            strike_price: strike price of option
            is_call: True if call option, False otherwise
        
        returns:
            payoff of the option
        '''
        if is_call:
            a = np.mean(prices, axis=1)
            b = strike_price
        else:
            a = strike_price
            b = np.mean(prices, axis=1)

        return np.maximum(a - b, 0) * np.exp(-self._r * self._T)