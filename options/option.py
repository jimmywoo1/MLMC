from abc import ABC, abstractmethod

class Option(ABC):
    '''
    abstract base class for options

    attributes
        _r: risk free rate
        _T: time to maturity
    '''
    def __init__(self,
                 r: float,
                 T: float):
        '''
        constructor for the Option class

        args:
            r: risk free rate
            T: time to maturity
        '''
        self._r = r
        self._T = T

    @abstractmethod
    def payoff(self,
               prices: list,
               strike_price: float,
               is_call: bool) -> float:
        '''
        computes the payoff of the current option

        args:
            prices: time series of prices
            strike_price: strike price of option
            is_call: True if call option, False otherwise
        
        returns:
            payoff of the option
        '''
        pass