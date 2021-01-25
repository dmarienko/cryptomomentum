import pandas as pd
import numpy as np

from tools.analysis.timeseries import *
from tools.utils.utils import mstruct, red, green, yellow, blue, magenta, cyan, white, dict2struct
from tools.analysis.tools import scols, srows
from tools.loaders.binance import load_binance_data

from ira.simulator.Position import CryptoFuturesPosition
from ira.simulator.utils import shift_signals
from ira.analysis.portfolio import split_cumulative_pnl
from ira.utils.nb_functions import z_load, z_test_signals_inplace



def load_data(instrument):
    return {instrument: load_binance_data(instrument, '1m', path='../data')}


def prepare_data(instrument, timeframes=['1D', '1H', '5Min']):
    """
    Just convenient method for prepring data
    """
    import re
    data = load_binance_data(instrument, '1m', path='../data')
    r = {'instrument': instrument, 'M1': data}
    for i in timeframes:
        mg = re.match('(\\d+)(\\w+)', i)
        if len(mg.groups()) > 1:
            tf = f'{mg[2][:1]}{mg[1]}'
            r[tf] = ohlc_resample(data, i, resample_tz='UTC')
    return dict2struct(r)


def ibs(data):
    """
    IBS indicator
    """
    if isinstance(data, dict):
        return pd.DataFrame.from_dict({k: ibs(v) for k, v in data.items()})
    else: 
        return (data.close - data.low) / (data.high - data.low)
    

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


def rough_simulation(poi, data, trade_long, trade_short, fwd_returns, holding_time, commissions):
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
    
    pfl = pd.concat((+trade_long * fwd_returns.loc[buys.index][holding_time], 
                     -trade_short * fwd_returns.loc[sells.index][holding_time]), axis=0).sort_index()

    # we use double commissions (open and close position)
    comms = 2*data.loc[pfl.index].close * commissions
    
    # apply commissions correctly
    pp = pfl[pfl > 0]
    pp = pp - comms[pp.index]
    pn = pfl[pfl < 0]
    pn = pn - comms[pn.index]
    return pd.concat((pp , pn)).sort_index()


def fast_backtest(instrument, signals, ohlc):
    signals.columns = [instrument]
    return z_test_signals_inplace(shift_signals(signals, seconds=infer_series_frequency(ohlc[:10]).seconds - 1), 
                                  {instrument: ohlc}, 'crypto', spread=1, verbose=False)

def load_simulation_data(strategy_id, host=None):
    """
    Loading simulation data
    """
    r = mstruct()
    r.portfolio = split_cumulative_pnl(z_load(strategy_id, dbname='IRA_simulations_portfolio_logs', host=host)['data'])
    r.executions = z_load(strategy_id, dbname='IRA_simulations_execution_logs', host=host)['data']
    r.executions.set_index('creation_time', inplace=True)
    r.executions = r.executions[~np.isnat(r.executions.index)]
    r.executions = r.executions[['instrument', 'side', 'type', 'quantity', 'status', 'fill_avg_price']]
    r.executions = r.executions[r.executions.status=='FILLED']
    return r


def collect_entries_data(r):
    """
    Collect entries from simulation entries
    """
    pnl, s_ent = {}, {}
    symbols = [s.split('_')[0] for s in r.portfolio.columns[r.portfolio.columns.str.match('.*_PnL')]]

    p = {s: CryptoFuturesPosition(s) for s in symbols}
    t_st, p_st, pl = 0, 0, 0
    for t, s in scols(r.executions, r.executions.quantity.cumsum().rename('Ce')).iterrows():
        q0 = p[s.instrument].quantity
        pnl[t] = p[s.instrument].update_position(t, s.Ce, s.fill_avg_price)
        pl += pnl[t]
        if q0 == 0 and s.Ce != 0:
            t_st = t
            p_st = pl

        if q0 != 0 and s.Ce == 0:
            s_ent[t_st] = {'Instrument': s.instrument, 'Closed': t, 'Duration': t - t_st,'PnL': pl - p_st}
            t_st, p_st = 0, 0

    return pd.DataFrame.from_dict(s_ent, orient='index')       
