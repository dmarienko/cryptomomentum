import pandas as pd
import numpy as np

from tools.analysis.tools import ohlc_resample, scols, srows
from tools.analysis.timeseries import infer_series_frequency, pivot_point


class BreaksDetector:
    
    def get_upper_levels(self):
        pass
    
    def get_lower_levels(self):
        pass
    
    def get_breaks(self, ohlc):
        pass
    
    
class RollingHiLo(BreaksDetector):
    def __init__(self, ohlc, period):
        self.timeframe = pd.Timedelta(infer_series_frequency(ohlc[:10]))
        top = ohlc.rolling(period, min_periods=period).high.apply(lambda x: max(x))
        bot = ohlc.rolling(period, min_periods=period).low.apply(lambda x: min(x))
        
        self.hilo = pd.concat((top.rename('Top'), bot.rename('Bot')), axis=1)
        self.hilo.index = self.hilo.index + pd.Timedelta(self.timeframe)
        self.hilo = ohlc.combine_first(self.hilo).fillna(method='ffill')[self.hilo.columns]
        
    def get_upper_levels(self):
        return self.hilo['Top']
    
    def get_lower_levels(self):
        return self.hilo['Bot']
    
    def get_breaks(self, ohlc):
        d_1 = ohlc.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'})
        t = pd.concat((ohlc[['open', 'close']], d_1, self.hilo), axis=1)
        t.fillna(method='ffill', inplace=True)

        self.breaks = pd.concat((
            pd.Series(+1, t[(t.open_1 < t.Top) & (t.close_1 < t.Top) & (t.close > t.Top)].index),
            pd.Series(-1, t[(t.open_1 > t.Bot) & (t.close_1 > t.Bot) & (t.close < t.Bot)].index))).sort_index()
        return self.breaks
    
    
class PivotsBreaks(BreaksDetector):
    def __init__(self, ohlc, res_level, sup_level, pp_frame='D', timezone='UTC', method='classic'):
        """
        Find out breakouts (point of interests) of pivot points levels
        """
        tolist = lambda x: [x] if not isinstance(x, (list, tuple)) else x
        self.res_levels = tolist(res_level)
        self.sup_levels = tolist(sup_level)
        self.pp = pivot_point(ohlc, method=method, timeframe=pp_frame, timezone=timezone)
    
    def get_breaks(self, ohlc):
        d_1 = ohlc.shift(1)[['open', 'close']].rename(columns={'open': 'open_1', 'close': 'close_1'})
        t = pd.concat((ohlc[['open', 'close']], d_1, self.pp), axis=1)
    
        self.breaks = srows(
            # breaks up levels specified as resistance
            *[pd.Series(+1, t[(t.open_1 < t[ul]) & (t.close_1 < t[ul]) & (t.close > t[ul])].index) for ul in self.res_levels if ul in self.pp.columns], 
        
            # breaks down levels specified as supports
            *[pd.Series(-1, t[(t.open_1 > t[bl]) & (t.close_1 > t[bl]) & (t.close < t[bl])].index) for bl in self.sup_levels if bl in self.pp.columns], 
             keep='last', sort=True) 
        
        return self.breaks
    
    
class TurtlesGenerator:
    
    def __init__(self, 
                 entry_detector: BreaksDetector, exit_detector: BreaksDetector, 
                 account_size: float, dollar_per_point:int, max_units:int=4, after_lose_only=True, 
                 atr_period = '1D', resample_tz='UTC'):
        self.account_size = account_size
        self.dollar_per_point = dollar_per_point
        self.max_units = max_units
        
        self.trading_after_lose_only = after_lose_only

        self.entry_detector = entry_detector
        self.exit_detector = exit_detector
        self.resample_tz = resample_tz
        self.atr_period = atr_period
    
    def get_signals(self, ohlc, trade_on='close'):
        self.sigs = {}
        self.pos = 0
        self.sl = np.nan
        
        brks_entry = self.entry_detector.get_breaks(ohlc)
        brks_exit = self.exit_detector.get_breaks(ohlc)
        mx = pd.concat((brks_entry, brks_exit, ohlc[trade_on]), keys=['breaks_entry', 'breaks_exit', trade_on], axis=1)
        
        # preparing days df
        days = ohlc_resample(ohlc, self.atr_period, resample_tz=self.resample_tz)

        days['pdc'] = days.close.shift(1)
        days['TR'] = days.apply(lambda x : max(x['high'] - x['low'], x['high'] - x['pdc'], x['pdc'] - x['low']), axis=1)
        days['N'] = np.nan
        
        # first N based on 20-day average true range
        firstN = np.mean(days[20:]['TR'])
        n_col = days.columns.get_loc('N')
        days.iloc[21, n_col] = firstN 

        # calculate N for every day
        prevN = firstN
        for i in range(22, len(days)):
            row = days.iloc[i]
            currN = (19 * prevN + row['TR']) / 20
            days.iloc[i, n_col] = currN
            prevN = currN
        
        # run 'trading' and collect signals in sigs dict
        entries = []
        curr_day = days.iloc[0]
        last_lose = False
        
        for t, brk, brk_exit, price in zip(mx.index, mx.breaks_entry.values, mx.breaks_exit.values, mx[trade_on].values):
            if curr_day.name + pd.Timedelta(self.atr_period) <= t:
                curr_day = days.iloc[days.index.get_loc(t, method='pad')]
                if self.pos != 0:
                    self.__calculate_sl(curr_day, entries[-1]) 
            
            if self.pos == 0 and not np.isnan(brk) and not np.isnan(curr_day['N']):
                # initial entry
                tradesize = self.__calculate_tradesize(brk, curr_day['N'])
                entries = []
                entries.append(price)
                self.__change_pos(tradesize, t, last_lose)
                self.__calculate_sl(curr_day, entries[-1]) 
                
            elif self.pos !=0 and len(entries) < self.max_units:
                # additional entries
                half_n = curr_day['N'] / 2
                if (self.pos > 0 and price > entries[-1] + half_n) or (self.pos < 0 and price < entries[-1] - half_n):
                    tradesize = self.__calculate_tradesize(np.sign(self.pos), curr_day['N'])
                    self.__change_pos(self.pos + tradesize, t, last_lose)
                    entries.append(price)
                    
            if (self.pos < 0 and price >= self.sl) or (self.pos > 0 and price <= self.sl):
                # stop loss exit
                self.__change_pos(0, t, last_lose)
                last_lose = True
            elif (self.pos < 0 and brk_exit == 1) or (self.pos > 0 and brk_exit == -1):
                # win exit
                self.__change_pos(0, t, last_lose)
                last_lose = False
                
        return pd.DataFrame.from_dict(self.sigs, orient='index')
    
    def __calculate_tradesize(self, sign, n):
        return round((0.01 * self.account_size) / (n * self.dollar_per_point)) * sign
    
    def __change_pos(self, pos, time, last_lose):
        self.pos = pos
        # skip signal if last breakout did not end in loss
        if self.trading_after_lose_only is False or last_lose is True:
            self.sigs[time] = pos
            
    def __calculate_sl(self, curr_day, entry_price):
        if self.pos > 0:
            self.sl = entry_price - curr_day['N'] * 2
        elif self.pos < 0:
            self.sl = entry_price + curr_day['N'] * 2; 
        