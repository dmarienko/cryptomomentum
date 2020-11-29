import pandas as pd
import numpy as np


class RangeSelector:
    def __init__(self):
        self._from = None
        self._to = None
        self._column = 'close'
    
    def select(self, start, end=None, column='close'):
        self._from, self._to, self._column = start, end, column
        return self
        
    def get_data(self, data):
        _f, _e = self._from, self._to
        
        _f = data.index[0] if _f is None else _f
        _e = data.index[-1] if _e is None else _e
        
        _f = data.index[max(_f, 0)] if isinstance(_f, int) else _f
        _e = data.index[min(_e, len(data)-1)] if isinstance(_e, int) else _e
        
        # test fractional slices
        if _f is not None and isinstance(_f, float) and _f < 1:
            _f = data.index[int(len(data) * _f)]
            
        if _e is not None and isinstance(_e, float) and _e < 1:
            _e = data.index[int(len(data) * _e)]
        
        return data[slice(_f, _e)] if _f is not None and _e is not None else data
    

class MarketRegime(RangeSelector):
    """
    Abstract class for market regime provider
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(**kwargs)
        self._regime = None
    
    def fit(self, data, prevent_shift=False):
        self._regime = self.calculate(self.get_data(data), prevent_shift)
        self._regime.name = self.__class__.__name__
        return self
    
    def calculate(self, data, prevent_shift: bool):
        return pd.Series(np.nan, data.index)
    
    def regime(self, time=None):
        if time is None:
            return self._regime
        elif isinstance(time, str):
            indexes = [pd.Timestamp(time)]
        elif not isinstance(time, (list, tuple, pd.Index)):
            if isinstance(time, (pd.Series, pd.DataFrame)):
                indexes = time.index
            else:
                indexes = [time]
        else:
            indexes = time
        
        _d_start = self._regime.index[0]
        _d_end = self._regime.index[-1]
        _xr = self._regime
        _xi = {i: _xr.index.get_loc(i, method='pad') for i in indexes if i >= _d_start and i <= _d_end}
        
        # restore index as resquested in time
        _t = _xr.iloc[list(_xi.values())]
        return _t.reset_index().set_index(pd.Series(_xi.keys(), name='time')).drop(columns='time')
    
    def get_id(self):
        cn = self.__class__.__name__
        return cn + '(' + ','.join([f'{n}={v}' for n,v in self.__dict__.items() if not n.startswith('_')]) + ')'