import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from copy import deepcopy

from tools.analysis.timeseries import pivot_point
from tools.analysis.tools import ohlc_resample, scols, srows, shift_signals

from qlearn.tracking.trackers import Tracker, TakeStopTracker, TurtleTracker, DispatchTracker, PipelineTracker, TimeExpirationTracker
from qlearn.core.base import BasicMarketEstimator, MarketDataComposer
from qlearn.core.pickers import SingleInstrumentPicker


class RollingRange(TransformerMixin):
    """
    Produces range top/bottom measures on rolling base
    """
    def __init__(self, timeframe, period, tz='UTC'):
        self.period = period
        self.timeframe = timeframe
        self.tz = tz
    
    def fit(self, x, **kwargs):
        return self
    
    def transform(self, x):
        ohlc = ohlc_resample(x, self.timeframe, resample_tz=self.tz)
        hilo = scols(
            ohlc.rolling(self.period, min_periods=self.period).high.apply(lambda x: max(x)),
            ohlc.rolling(self.period, min_periods=self.period).low.apply(lambda x: min(x)),
            names=['RangeTop', 'RangeBot'])
        hilo.index = hilo.index + pd.Timedelta(self.timeframe)
        hilo = ohlc.combine_first(hilo).fillna(method='ffill')[hilo.columns]
        return x.assign(RangeTop = hilo.RangeTop, RangeBot = hilo.RangeBot)
    
    
class RangeBreakoutDetector(BasicMarketEstimator):
    def fit(self, X, y, **fit_params):
        return self

    def predict(self, X):
        meta = self.metadata()
        hilo = X[['RangeTop', 'RangeBot']]
        
        ohlc = X.shift(1)
        d_1 = ohlc.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'})
        t = pd.concat((ohlc[['open', 'close']], d_1, hilo), axis=1)
        t.fillna(method='ffill', inplace=True)
        sigs = srows(
            pd.Series(+1, t[(t.open_1 < t.RangeTop) & (t.close_1 < t.RangeTop) & (t.close > t.RangeTop)].index),
            pd.Series(-1, t[(t.open_1 > t.RangeBot) & (t.close_1 > t.RangeBot) & (t.close < t.RangeBot)].index)
        )
        return sigs
    

class PivotsRange(TransformerMixin):
    """
    Produces pivots levels
    """
    def __init__(self, timeframe, method='classic', tz='UTC'):
        self.timeframe = timeframe
        self.method = method
        self.tz = tz
    
    def fit(self, x, **kwargs):
        return self
    
    def transform(self, x):
        pp = pivot_point(x, method=self.method, timeframe=self.timeframe, timezone=self.tz)
        return pd.concat((x, pp), axis=1)
    
    
class PivotsBreakoutDetector(BasicMarketEstimator):
    def __init__(self, resistances, supports):
        tolist = lambda x: [x] if not isinstance(x, (list, tuple)) else x
        self.res_levels = tolist(resistances)
        self.sup_levels = tolist(supports)
        
    def fit(self, X, y, **fit_params):
        return self

    def predict(self, x):
        meta = self.metadata()
        
        t = scols(x, x.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'}))
        cols = x.columns
        breaks = srows(
            # breaks up levels specified as resistance
            *[pd.Series(+1, t[(t.open_1 < t[ul]) & (t.close_1 < t[ul]) & (t.close > t[ul])].index) for ul in self.res_levels if ul in cols], 
        
            # breaks down levels specified as supports
            *[pd.Series(-1, t[(t.open_1 > t[bl]) & (t.close_1 > t[bl]) & (t.close < t[bl])].index) for bl in self.sup_levels if bl in cols], 
             keep='last') 
        return breaks

    
def generate_pivots_signals(pivot, data, shift):
    pvt_e_lo = MarketDataComposer(
        make_pipeline(deepcopy(pivot), PivotsBreakoutDetector(['R4', 'R3', 'R2', 'R1'], [])), 
        SingleInstrumentPicker(), None).fit(data, None).predict(data)

    pvt_x_lo = MarketDataComposer(
        make_pipeline(deepcopy(pivot), PivotsBreakoutDetector([], ['P', 'S1', 'S2', 'S3', 'S4'])), 
        SingleInstrumentPicker(), None).fit(data, None).predict(data)

    pvt_e_sh = MarketDataComposer(
        make_pipeline(deepcopy(pivot), PivotsBreakoutDetector([], ['S1', 'S2', 'S3', 'S4'])), 
        SingleInstrumentPicker(), None).fit(data, None).predict(data)

    pvt_x_sh = MarketDataComposer(
        make_pipeline(deepcopy(pivot), PivotsBreakoutDetector(['R4', 'R3', 'R2', 'R1', 'P'], [])), 
        SingleInstrumentPicker(), None).fit(data, None).predict(data)
    
    return shift_signals(srows(1 * pvt_e_lo, 1 * pvt_e_sh, 2 * pvt_x_lo, 2 * pvt_x_sh), shift)