import pandas as pd
import numpy as np

from tools.analysis.timeseries import *
from tools.utils.utils import mstruct

def find_rhl_breakouts(data, timeframe, period, timezone='UTC'):
    """
    Find out breakouts (point of interests) of rolling high / low
    """
    htf = ohlc_resample(data, timeframe, resample_tz=timezone)
    top = htf.rolling(period, min_periods=period).high.apply(lambda x: max(x))
    bot = htf.rolling(period, min_periods=period).low.apply(lambda x: min(x))

    hilo = pd.concat((top.rename('Top'), bot.rename('Bot')), axis=1)
    hilo.index = hilo.index + pd.Timedelta(timeframe)
    hilo = data.combine_first(hilo).fillna(method='ffill')[hilo.columns]
    d_1 = data.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'})
    t = pd.concat((data[['open', 'close']], d_1, hilo), axis=1)

    poi = pd.concat((
        pd.Series(+1, t[(t.open_1 < t.Top) & (t.close_1 < t.Top) & (t.close > t.Top)].index),
        pd.Series(-1, t[(t.open_1 > t.Bot) & (t.close_1 > t.Bot) & (t.close < t.Bot)].index))).sort_index()
    
    return mstruct(poi=poi, hilo=hilo)


def rough_simulation(poi, trade_long, trade_short, fwd_returns, holding_time, commissions):
    """
    Very rough tradeability simulation.
    Uses:
      poi (points of interests) 
      size for trade longs/shorts
      forward returns matrix
      holding time (in forward returns units)
      and commissions per unit
    """
    buys = poi[poi > 0]    # long when breaks rolling high
    sells = poi[poi < 0]   # short when breask rolling low
    holding_time = f'F{holding_time}' if isinstance(holding_time, int) else holding_time
    
    pfl = pd.concat((+trade_long * r.loc[buys.index][holding_time], 
                     -trade_short * r.loc[sells.index][holding_time]), axis=0).sort_index()

    # we use double commissions (open and close position)
    comms = 2*data.loc[pfl.index].close * commissions
    
    # apply commissions correctly
    pp = pfl[pfl > 0]
    pp = pp - comms[pp.index]
    pn = pfl[pfl < 0]
    pn = pn - comms[pn.index]
    return pd.concat((pp , pn)).sort_index()