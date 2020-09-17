"""
   Misc graphics handy utilitites to be used in interactive analysis
"""
import numpy as np
import pandas as pd
import itertools as it
from typing import List, Tuple, Union

try:
    import matplotlib
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    print("Can't import matplotlib modules in ira charting modlue")

from tools.analysis.tools import isscalar
from tools.charting.mpl_finance import ohlc_plot


def fig(w=16, h=5, dpi=96, facecolor=None, edgecolor=None, num=None):
    """
    Simple helper for creating figure
    """
    return plt.figure(num=num, figsize=(w, h), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)


def subplot(shape, loc, rowspan=1, colspan=1):
    """
    Some handy grid splitting for plots. Example for 2x2:
    
    >>> subplot(22, 1); plt.plot([-1,2,-3])
    >>> subplot(22, 2); plt.plot([1,2,3])
    >>> subplot(22, 3); plt.plot([1,2,3])
    >>> subplot(22, 4); plt.plot([3,-2,1])

    same as following

    >>> subplot((2,2), (0,0)); plt.plot([-1,2,-3])
    >>> subplot((2,2), (0,1)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,0)); plt.plot([1,2,3])
    >>> subplot((2,2), (1,1)); plt.plot([3,-2,1])

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param rowspan: rows spanned
    :param colspan: columns spanned
    """
    if isscalar(shape):
        if 0 < shape < 100:
            shape = (max(shape // 10, 1), max(shape % 10, 1))
        else:
            raise ValueError("Wrong scalar value for shape. It should be in range (1...99)")

    if isscalar(loc):
        nm = max(shape[0], 1) * max(shape[1], 1)
        if 0 < loc <= nm:
            x = (loc - 1) // shape[1]
            y = loc - 1 - shape[1] * x
            loc = (x, y)
        else:
            raise ValueError("Wrong scalar value for location. It should be in range (1...%d)" % nm)

    return plt.subplot2grid(shape, loc=loc, rowspan=rowspan, colspan=colspan)


def sbp(shape, loc, r=1, c=1):
    """
    Just shortcut for subplot(...) function

    :param shape: scalar (like matlab subplot) or tuple
    :param loc: scalar (like matlab subplot) or tuple
    :param r: rows spanned
    :param c: columns spanned
    :return:
    """
    return subplot(shape, loc, rowspan=r, colspan=c)


def plot_trends(trends, uc='w--', dc='c--', lw=0.7, ms=5, fmt='%H:%M'):
    """
    Plot find_movements function output as trend lines on chart

    >>> from ira.analysis import find_movements
    >>>
    >>> tx = pd.Series(np.random.randn(500).cumsum() + 100, index=pd.date_range('2000-01-01', periods=500))
    >>> trends = find_movements(tx, np.inf, use_prev_movement_size_for_percentage=False,
    >>>                    pcntg=0.02,
    >>>                    t_window=np.inf, drop_weekends_crossings=False,
    >>>                    drop_out_of_market=False, result_as_frame=True, silent=True)
    >>> plot_trends(trends)

    :param trends: find_movements output
    :param uc: up trends line spec ('w--')
    :param dc: down trends line spec ('c--')
    :param lw: line weight (0.7)
    :param ms: trends reversals marker size (5)
    :param fmt: time format (default is '%H:%M')
    """
    if not trends.empty:
        u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
        plt.plot([u.index, u.end], [u.start_price, u.end_price], uc, lw=lw, marker='.', markersize=ms);
        plt.plot([d.index, d.end], [d.start_price, d.end_price], dc, lw=lw, marker='.', markersize=ms);

        from matplotlib.dates import num2date
        import datetime
        ax = plt.gca()
        ax.set_xticklabels([datetime.date.strftime(num2date(x), fmt) for x in ax.get_xticks()])
