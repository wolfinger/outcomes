# -*- coding: utf-8 -*-
"""A suite of tools to enable better data analysis for finance.

The outcomes module is the sole module of the outcomes package. The module 
provides a suite of classes and functions to perform more robust data analysis 
for finance -- specifically financial markets.
"""

import numpy as np
import pandas as pd
import copy
import statsmodels.api as sm


# ignore numpy warnings (e.g., when finding no downside vol for a short period)
np.seterr(invalid='ignore')


class Periodicity:
    """Frequency interval of the data set.
    
    Attributes:
        name (str): A human readable label for the periodicity.
        min_days (float): Lower bound (in days) for autodetecting the 
            periodicity.
        max_days (float): Upper bound (in days) for autodetecting the 
            periodicity.
        ann_factor (float): Factor used to annualize data.
    """

    def __init__(self, name, min_days, max_days, ann_factor):
        """Inits Periodicity class using all four passed in attributes."""
        self.name = name
        self.min_days = min_days
        self.max_days = max_days
        self.ann_factor = ann_factor


class TimePeriod:
    """A defined period of time to assist with common time period analysis.

    Attributes:
        name (str): A short label for the time period (e.g., 1w = one week).
        ann_factor (float): Factor used to annualize the time period.
        stat_flag (bool, optional). Boolean indicating if time period requires
            a minimum number of observations to be statistically useful.
            (default is False)
    """

    def __init__(self, name, ann_factor, stat_flag=False):
        """Inits TimePeriod class using passed in attributes."""

        self.name = name
        self.ann_factor = ann_factor
        self.stat_flag = stat_flag

    def __str__(self):
        """Prints TimePeriod class' attributes in CSV-esque string."""

        return (self.name + ', ' + str(self.ann_factor) + ', ' +
                str(self.stat_flag))


class Drawdown:
    """A multi-dimensional assessment of a series' peak-to-trough history.

    Attributes:
        name (str): [desc].
        series (Pandas DataFrame): The series' analyzed for drawdowns.
        count (int): The number of drawdowns occurred.
        avg_size (float): The average drawdown size.
        avg_periods (float): The average length of time for drawdowns.
        med_size (float): The median drawdown size.
        med_periods (float): The median length of time for drawdowns.
        max_size_size (float): The maximum drawdown's size.
        max_size_periods (int): The maximum drawdown's length of time.
        max_size_start_dt (datetime): The start date of the maximum drawdown.
        max_size_end_dt (datetime): The end date of the maximum drawdown.
        max_length_size (float): The longest drawdown's maximum size.
        max_length_periods (int): The longest drawdown's length of time.
        max_length_start_dt (datetime): The start date of the longest drawdown.
        max_length_end_dt (datetime): The end date of the longest drawdown.
        curr_series (Pandas DataFrame): The actaul drawdown series [TODO].
        curr_size (float): The size of current drawdown.
        curr_max_size (float): The max size for current drawdown.
        curr_periods (int): The length of the current drawdown.
        curr_start_dt (datetime): The start date of the current drawdown.
    """

    def __init__(self, name, series, count, avg_size, avg_periods,
                 med_size, med_periods, max_size_size, max_size_periods,
                 max_size_start_dt, max_size_end_dt, max_length_size,
                 max_length_periods, max_length_start_dt, max_length_end_dt,
                 curr_series, curr_size, curr_max_size, curr_periods,
                 curr_start_dt):
        """Inits a Drawdown object with the passed in attributes."""

        self.name = name
        self.series = series
        self.count = count
        self.avg_size = avg_size
        self.avg_periods = avg_periods
        self.med_size = med_size
        self.med_periods = med_periods
        self.max_size_size = max_size_size
        self.max_size_periods = max_size_periods
        self.max_size_start_dt = max_size_start_dt
        self.max_size_end_dt = max_size_end_dt
        self.max_length_size = max_length_size
        self.max_length_periods = max_length_periods
        self.max_length_start_dt = max_length_start_dt
        self.max_length_end_dt = max_length_end_dt
        self.curr_series = curr_series
        self.curr_size = curr_size
        self.curr_max_size = curr_max_size
        self.curr_periods = curr_periods
        self.curr_start_dt = curr_start_dt


class Measures:
    """Suite of measures that outcomes can generate.

    A helper class that lists the names of measures that the outcomes pacakge 
    can generate. The measure names can be overriden at class intantiation 
    to output user desired names.

    Attributes:
        level (str): [desc].
        level_ln (str): log level of the measure.
        chg_rel (str): the relative change, aka the compound return.
        chg_abs (str): the absolute change in the level.
        chg_ln (str): the absolute change in the log level.
        vol_lvl (str): vol of the level of the series.
        vol_lvl_ln (str): vol of the log level of the series.
        vol_cmp (str): vol of relative changes/compound returns.
        vol_ln (str): vol of log level changes/log returns.
        sharpe_ln (str): Sharpe ratio using log measures.
        info_ratio_ln (str): information ratio using log measures.
        vol_downside_ln (str): downside vol using log measures.
        vol_upside_ln (str): upside vol using log measures.
        sortino_ln (str): Sortino ratio using log measures.
        vol_skew_ln (str): vol skew using log measures.
        mod_treynor_ln (str): modified Treynor ratio using log measures.
        corr_lvl (str): correlation of the levels of two measures.
        corr_lvl_ln (str): correlaiton of the log levels of two measures.
        corr_ln (str): correlation of the log changes of two measures.
        r2_ln (str): the R2 of a dependent measure against an 
            independent measure (in log terms).
        beta_ln (str): the beta of a depedent measure against an independent 
            measure (in log terms).
        beta_up_ln (str): the upside beta (in log terms).
        beta_down_ln (str): the downside beta (in log terms).
        jensens_alpha_ln (str): Jensen's alpha (in log terms).
        m2_ln (str): M2 (in log terms).
        m2_excess_ln (str): M2 excess against a reference series (in log terms).
        dd_series_cmp (str): The actual drawdown series (in compound terms).
        dd_count_cmp (str): The count of drawdowns in a series.
        dd_avg_size_cmp (str): The average size of drawdowns in a series (in 
            compound terms).
        dd_avg_periods_cmp (str): The average length of drawdowns in a series.
        dd_med_size_cmp (str): The meidan size of drawdowns in a series (in 
            compound terms).
        dd_med_periods_cmp (str): The median legnth of drawdowns in a series.
        dd_max_size_cmp (str): The maximum drawdown size of a series (in 
            compound terms).
        dd_max_size_periods_cmp (str): The length of the maximum size drawdown.
        dd_max_size_start_cmp (str): The start date of the maximum size 
            drawdown.
        dd_max_size_end_cmp (str): The end date of the maximum size drawdown.
        dd_max_length_size_cmp (str): The size of the longest running drawdown 
            in a series (in compound terms).
        dd_max_length_cmp (str): The length of the longest running drawdown.
        dd_max_length_start_cmp (str): The start date of the longest running 
            drawdown.
        dd_max_length_end_cmp (str): The end date of the longest running 
            drawdown.
        dd_curr_series_cmp (str): The current drawdown series (in compound 
            terms).
        dd_curr_size_cmp (str): The size of the current drawdown (in compound 
            terms).
        dd_curr_max_size_cmp (str): The maximum size of the current drawdown 
            (in compound terms).
        dd_curr_length_cmp (str): The length of the current drawdown.
        dd_curr_start_cmp (str): The start date of the current drawdown.
        dd_cmp (array): The array of drawdown metrics for a series.
    """

    DRAWDOWN = 'drawdown'
    """Constant used to flag the entire drawdown metric suite."""

    def __init__(self, level = 'level', level_ln = 'level_ln', 
                 chg_rel = 'chg_rel', chg_abs = 'chg_abs', chg_ln = 'chg_ln', 
                 vol_lvl = 'vol_lvl', vol_lvl_ln = 'vol_lvl_ln', 
                 vol_cmp = 'vol_cmp', vol_ln = 'vol_ln', 
                 sharpe_ln = 'sharpe_ln', info_ratio_ln = 'info_ratio_ln', 
                 vol_downside_ln = 'vol_downside_ln', 
                 vol_upside_ln = 'vol_upside_ln', sortino_ln = 'sortino_ln', 
                 vol_skew_ln = 'vol_skew_ln', mod_treynor_ln = 'mod_treynor_ln',
                 corr_lvl = 'corr_lvl', corr_lvl_ln = 'corr_lvl_ln', 
                 corr_ln = 'corr_ln', r2_ln = 'r2_ln', beta_ln = 'beta_ln', 
                 beta_up_ln = 'beta_up_ln', beta_down_ln = 'beta_down_ln', 
                 jensens_alpha_ln = 'jensens_alpha_ln', m2_ln = 'm2_ln', 
                 m2_excess_ln = 'm2_excess_ln', dd_series_cmp = 'dd_series_cmp',
                 dd_count_cmp = 'dd_count_cmp', 
                 dd_avg_size_cmp = 'dd_avg_size_cmp', 
                 dd_avg_periods_cmp = 'dd_avg_periods_cmp', 
                 dd_med_size_cmp = 'dd_med_size_cmp', 
                 dd_med_periods_cmp = 'dd_med_periods_cmp', 
                 dd_max_size_cmp = 'dd_max_size_cmp', 
                 dd_max_size_periods_cmp = 'dd_max_size_periods_cmp', 
                 dd_max_size_start_cmp = 'dd_max_size_start_cmp', 
                 dd_max_size_end_cmp = 'dd_max_size_end_cmp', 
                 dd_max_length_size_cmp = 'dd_max_legnth_size_cmp', 
                 dd_max_length_cmp = 'dd_max_length_cmp', 
                 dd_max_length_start_cmp = 'dd_max_length_start_cmp', 
                 dd_max_length_end_cmp = 'dd_max_length_end_cmp', 
                 dd_curr_series_cmp = 'dd_curr_series_cmp', 
                 dd_curr_size_cmp = 'dd_curr_size_cmp', 
                 dd_curr_max_size_cmp = 'dd_curr_max_size_cmp', 
                 dd_curr_length_cmp = 'dd_curr_length_cmp', 
                 dd_curr_start_cmp = 'dd_curr_start_cmp'):
        """Init the Measures class."""

        self.level = level
        self.level_ln = level_ln
        self.chg_rel = chg_rel
        self.chg_abs = chg_abs
        self.chg_ln = chg_ln
        self.vol_lvl = vol_lvl
        self.vol_lvl_ln = vol_lvl_ln
        self.vol_cmp = vol_cmp
        self.vol_ln = vol_ln
        self.sharpe_ln = sharpe_ln
        self.info_ratio_ln = info_ratio_ln
        self.vol_downside_ln = vol_downside_ln
        self.vol_upside_ln = vol_upside_ln
        self.sortino_ln = sortino_ln
        self.vol_skew_ln = vol_skew_ln
        self.mod_treynor_ln = mod_treynor_ln
        self.corr_lvl = corr_lvl
        self.corr_lvl_ln = corr_lvl_ln
        self.corr_ln = corr_ln
        self.r2_ln = r2_ln
        self.beta_ln = beta_ln
        self.beta_up_ln = beta_up_ln
        self.beta_down_ln = beta_down_ln
        self.jensens_alpha_ln = jensens_alpha_ln
        self.m2_ln = m2_ln
        self.m2_excess_ln = m2_excess_ln
        self.dd_series_cmp = dd_series_cmp
        self.dd_count_cmp = dd_count_cmp
        self.dd_avg_size_cmp = dd_avg_size_cmp
        self.dd_avg_periods_cmp = dd_avg_periods_cmp
        self.dd_med_size_cmp = dd_med_size_cmp
        self.dd_med_periods_cmp = dd_med_periods_cmp
        self.dd_max_size_cmp = dd_max_size_cmp
        self.dd_max_size_periods_cmp = dd_max_size_periods_cmp
        self.dd_max_size_start_cmp = dd_max_size_start_cmp
        self.dd_max_size_end_cmp = dd_max_size_end_cmp
        self.dd_max_length_size_cmp = dd_max_length_size_cmp
        self.dd_max_length_cmp = dd_max_length_cmp
        self.dd_max_length_start_cmp = dd_max_length_start_cmp
        self.dd_max_length_end_cmp = dd_max_length_end_cmp
        self.dd_curr_series_cmp = dd_curr_series_cmp
        self.dd_curr_size_cmp = dd_curr_size_cmp
        self.dd_curr_max_size_cmp = dd_curr_max_size_cmp
        self.dd_curr_length_cmp = dd_curr_length_cmp
        self.dd_curr_start_cmp = dd_curr_start_cmp
        self.dd_cmp = [
            self.dd_series_cmp,
            self.dd_count_cmp,
            self.dd_avg_size_cmp,
            self.dd_avg_periods_cmp,
            self.dd_med_size_cmp,
            self.dd_med_periods_cmp,
            self.dd_max_size_cmp,
            self.dd_max_size_periods_cmp,
            self.dd_max_size_start_cmp,
            self.dd_max_size_end_cmp,
            self.dd_max_length_size_cmp,
            self.dd_max_length_cmp,
            self.dd_max_length_start_cmp,
            self.dd_max_length_end_cmp,
            self.dd_curr_series_cmp,
            self.dd_curr_size_cmp,
            self.dd_curr_max_size_cmp,
            self.dd_curr_length_cmp,
            self.dd_curr_start_cmp
        ]

    def __iter__(self):
        for attr in self.__dict__.keys():
            if not attr.startswith('__'):
                yield attr


PDCY_DAILY_CALENDAR = Periodicity('daily_calendar', 0, 1, 365.25)
"""The daily (calendar days) periodicity; annualizes using 365.25 days/yr."""
PDCY_DAILY_TRADING = Periodicity('daily_trading', 1, 4, 252)
"""The daily (trading days) periodicity; annualizes using 252 days/yr."""
PDCY_WEEKLY = Periodicity('weekly', 4, 7, 52)
"""The weekly periodicity; annualizes using 52 weeks/yr."""
PDCY_BI_MONTHLY = Periodicity('bi-monthly', 7, 17, 24)
"""The bi-monthly periodicity; annualizes using 24 bi-weeks/yr."""
PDCY_MONTHLY = Periodicity('monthly', 17, 31, 12)
"""The monthly periodicity; annualizes using 12 months/yr."""
PDCY_SEMI_MONTHLY = Periodicity('semi-monthly', 31, 62, 6)
"""The semi-monthly periodicity; annualizes using 6 semi-months/yr."""
PDCY_QUARTERLY = Periodicity('quarterly', 62, 94, 4)
"""The quarterly periodicity; annualizes using 4 quarters/yr."""
PDCY_SEMI_ANNUALLY = Periodicity('semi-annually', 94, 184, 2)
"""The semi-annual periodicity; annualizes using twice/yr."""
PDCY_ANNUALLY = Periodicity('annually', 184, 366, 1)
"""The annual periodicity; by definition already stated annually."""

TP_SPOT = 'spot'
"""The 'spot' TimePeriod. Used to reference the current one-period value."""
TP_1D = '1d'
"""The 1 day TimePeriod. If the periodicity is daily this matches TP_SPOT."""
TP_1W = '1w'
"""The one week TimePeriod."""
TP_2W = '2w'
"""The two week TimePeriod."""
TP_1M = '1m'
"""The one month TimePeriod."""
TP_6W = '6w'
"""The six week TimePeriod."""
TP_2M = '2m'
"""The two month TimePeriod."""
TP_3M = '3m'
"""The three month TimePeriod."""
TP_6M = '6m'
"""The six month TimePeriod."""
TP_1Y = '1y'
"""The one year TimePeriod."""
TP_2Y = '2y'
"""The two year TimePeriod."""
TP_3Y = '3y'
"""The three year TimePeriod."""
TP_5Y = '5y'
"""The five year TimePeriod."""
TP_10Y = '10y'
"""The ten year TimePeriod."""
TP_WTD = 'wtd'
"""The week-to-date TimePeriod; not implemented yet."""
TP_MTD = 'mtd'
"""The month-to-date TimePeriod; not implemented yet."""
TP_QTD = 'qtd'
"""The quarter-to-date TimePeriod; not implemented yet."""
TP_YTD = 'ytd'
"""The year-to-date TimePeriod; not implemented yet."""
TP_CUM = 'cum'
"""The cumulative TimePeriod; abbreviated as 'cum'."""

MIN_OBS_THRESH = 6
"""min num of observations necessary for certain statistics; set to 6."""


def create_tm_periods(srs_size, periodicity, periods=[]):
    """Creates a standard set of common time periods used for analysis.
    
    Creates a set of valid TimePeriods that can be used with the time series
    data of interest. Will automatically drop TimePeriods that are too short
    for relevance (based on periodicity) and too long based on series length.

    Args:
        srs_size (int): The length/size of the time series data.
        periodicity (float): The periodicity of the time series data.
        periods (list of TimePeriods, optional): List of time periods to 
            generate. Uses all time periods by default.
            (default is empty/all)
    
    Returns:
        list of TimePeriods: A list of TimePeriods based on the series size & 
        periodicity and any passed in TimePeriods requested. Drops nonsensical 
        TimePeriods.
    """

    if periods == []:
        periods = [TP_1D, TP_1W, TP_2W, TP_1M, TP_6W, TP_2M, TP_3M, TP_6M,
                   TP_1Y, TP_2Y, TP_3Y, TP_5Y, TP_10Y, TP_CUM]

    tm_periods = {}
    tm_periods[TP_SPOT] = TimePeriod(TP_SPOT, 1 / periodicity.ann_factor)

    if TP_1D in periods:
        tm_periods[TP_1D] = TimePeriod(TP_1D, 1 / max(periodicity.ann_factor,
            PDCY_DAILY_TRADING.ann_factor))

    if TP_1W in periods:
        tm_periods[TP_1W] = TimePeriod(TP_1W, 1 / 52)

    if TP_2W in periods:
        tm_periods[TP_2W] = TimePeriod(TP_2W, 2 / 52)

    if TP_1M in periods:  # if daily -> 30 calendar, 21 trading
        tm_periods[TP_1M] = TimePeriod(TP_1M, 1 / 12)

    if TP_6W in periods:
        tm_periods[TP_6W] = TimePeriod(TP_6W, 6 / 52)

    if TP_2M in periods:  # if daily -> 61 calendar, 42 trading
        tm_periods[TP_2M] = TimePeriod(TP_2M, 2 / 12)

    if TP_3M in periods:  # if daily -> 91 calendar, 63 trading
        tm_periods[TP_3M] = TimePeriod(TP_3M, 3 / 12)

    if TP_6M in periods:  # if daily -> 183 calendar, 126 trading
        tm_periods[TP_6M] = TimePeriod(TP_6M, 6 / 12)

    if TP_1Y in periods:
        tm_periods[TP_1Y] = TimePeriod(TP_1Y, 1.)

    if TP_2Y in periods:
        tm_periods[TP_2Y] = TimePeriod(TP_2Y, 2.)

    if TP_3Y in periods:
        tm_periods[TP_3Y] = TimePeriod(TP_3Y, 3.)

    if TP_5Y in periods:
        tm_periods[TP_5Y] = TimePeriod(TP_5Y, 5.)

    if TP_10Y in periods:
        tm_periods[TP_10Y] = TimePeriod(TP_10Y, 10.)

    if TP_CUM in periods:
        tm_periods[TP_CUM] = TimePeriod(TP_CUM, 1.)

    # don't include rolling periods longer than input series
    # or short intertemporal periods for monthly series
    del_keys = []
    for key in tm_periods:
        obs = tm_periods[key].ann_factor * periodicity.ann_factor
        if obs >= srs_size or obs < 1 or (1 < obs < 2):
            del_keys.append(key)
        else:
            # flag periods that require a min observation threshold
            # for certain statistics
            if (tm_periods[key].ann_factor * periodicity.ann_factor >=
                    MIN_OBS_THRESH):
                tm_periods[key].stat_flag = True

    for key in del_keys:
        tm_periods.pop(key)

    return tm_periods


def get_periodicity(index):
    """Determines the periodicity of the time series index provided.

    Args:
        index (Pandas DataFrame of datetime): A Pandas datetime index.
    
    Returns:
        Periodicity: A Periodicity object based on the periodicity of the 
        passed index. Returns none if the periodicity cannot be determined.
    """

    periodicity = None

    timedelta = index[1] - index[0]
    days = 0

    if timedelta.seconds == 0:
        # if it's daily and a long enough series has been provided
        # determine if it's calendar days or trading days
        if (timedelta.days == 1) and (index.size > 6):
            days = 1
            for i in range(1, 6):
                if (index[i] - index[i - 1]).days > 1:
                    days = 2
        else:
            days = timedelta.days

    if days > 0:
        if days <= PDCY_DAILY_CALENDAR.max_days:
            periodicity = PDCY_DAILY_CALENDAR
        elif (PDCY_DAILY_TRADING.min_days < days <=
              PDCY_DAILY_TRADING.max_days):
            periodicity = PDCY_DAILY_TRADING
        elif (PDCY_WEEKLY.min_days < days <=
              PDCY_WEEKLY.max_days):
            periodicity = PDCY_WEEKLY
        elif (PDCY_BI_MONTHLY.min_days < days <=
              PDCY_BI_MONTHLY.max_days):
            periodicity = PDCY_BI_MONTHLY
        elif (PDCY_MONTHLY.min_days < days <=
              PDCY_MONTHLY.max_days):
            periodicity = PDCY_MONTHLY
        elif (PDCY_SEMI_MONTHLY.min_days < days <=
              PDCY_SEMI_MONTHLY.max_days):
            periodicity = PDCY_SEMI_MONTHLY
        elif (PDCY_QUARTERLY.min_days < days <=
              PDCY_QUARTERLY.max_days):
            periodicity = PDCY_QUARTERLY
        elif (PDCY_SEMI_ANNUALLY.min_days < days <=
              PDCY_SEMI_ANNUALLY.max_days):
            periodicity = PDCY_SEMI_ANNUALLY
        elif (PDCY_ANNUALLY.min_days < days <=
              PDCY_ANNUALLY.max_days):
            periodicity = PDCY_ANNUALLY
        else:
            periodicity = None  # seriously? longer than a year?
    else:
        periodicity = None  # sorry HFT, you'll need to build this

    return periodicity


def valid_period(period):
    """Checks to see if a period is valid.

    Args:
        period (float): The period to be validated.
    
    Returns:
        bool: Returns true if the period is valid, false otherwise.
    """

    if (period < 1) or (period % 1 != 0):
        return False
    else:
        return True


def return_to_level(returns, lvl_start_val=100, calc_method='cmp'):
    """Takes a series of returns and converts it to levels.

    An index level reflect a series of returns over time. If the returns passed 
    are total returns, a total return level series is generated. This is useful 
    if you want to easily interpret how much an asset's value grows over time.

    Args:
        returns (Pandas DataFrame of float). A return series in compound or log 
            space.
        lvl_start_val (float, optional). The value to start the index level at.
            (default is 100)
        cacl_method (str, optional). Create a compound or log level series.
            (default is compound)
    
    Returns:
        Pandas DataFrame of float: A level series based on the series' returns.
    """

    # create a new starting row for the initial level
    timedelta = returns.index[1] - returns.index[0]
    new_index = returns.index[0] - timedelta
    returns[new_index] = 0

    # convert to log returns before generating levels
    if calc_method == 'cmp':
        returns = np.log(1 + returns)

    # calc cumulative return and convert to compound value
    returns = np.exp(returns.sort_index().cumsum()) - 1

    # convert from cumulative returns to levels
    levels = (1 + returns) * lvl_start_val

    return levels


def excess(topic_srs, ref_srs):
    """Calculates the excess of a topic series vs. a reference series.

    Excess is simply the difference between the topic series and a reference 
    series. If passing returns, log returns should be used since the function 
    simply subtracts to calculate the difference.

    Args:
        topic_srs (Pandas DataFrame of float): The topic series.
        market_srs (Pandas DataFrame of float): The reference series.
    
    Returns:
        Pandas DataFrame of float: The difference between the topic series and 
        the reference series.
    """
    return topic_srs - ref_srs


def rolling_chg(series_data, periods, annual_factor=12, annualize_flag=True,
                cum=False, to_date=None):
    """Calculates a rolling change or returns (if passed log returns).

    Calculates a rolling period's value at any given point in time. Will step 
    through a series and at each observation date a rolling value is calculated.
    If the passed in series is returns data, it will be a rolling return, 
    otherwise it will be a rolling diff.

    Args:
        series_data (Pandas DataFrame of float): [desc].
        periods (int): [desc].
        annual_factor (float): [desc].
        annualize_flag (bool, optional): [desc].
            (default is True)
        cum (bool, optional): [desc].
            (default is False)
        to_date: Not implemented yet.
            (default is None)

    Returns:
        Pandas DataFrame of float: A series of rolling returns/diffs that 
        matches the passed in series.
    """

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    # don't return values if being asked to calculate something shorter
    # than 1 period or not a multiple of the periodicity
    if not valid_period(periods):
        return np.nan

    if (periods < annual_factor) or (annualize_flag is False):
        annualize = 1
    else:
        annualize = periods / annual_factor

    return (series_data.rolling(window=periods,
                                min_periods=periods).sum() / annualize
            if cum is False else series_data.cumsum())


def rolling_ols(y, X, periods, cum=False, to_date=None):
    """Calculates a rolling single variate least squares regression.

    Args:
        y: Dependent variable for the regression.
        X: Independent variable for the regression.
        periods: [desc].
        cum (bool, optional): Boolean to indicate whether to calculate a 
            cumulative series.
            (default is False)
        to_date: For future implementation of 'to-date' calculations.
            (default is None)

    Returns:
        Pandas DataFrame of float: A Pandas DataFrame with the rolling OLS 
        output in two columns: r2 - the r-squared and r2-adj - the adjusted 
        r-squared.
    """
    
    ols = []
    ols_details = {'r2': None, 'r2-adj': None}

    # round periods to nearest int if it comes in as float
    periods = int(np.round(periods)) #.astype(int)

    # don't return values if being asked to calculate something shorter
    # than 1 period or not a multiple of the periodicity
    if not valid_period(periods):
        return pd.DataFrame()

    # add blank first record
    ols.append(copy.copy(ols_details))

    X = sm.add_constant(X)

    for i in range(1, len(y)):

        if cum is False:
            if periods > i:
                start = i
            else:
                start = i - periods + 1
        else:
            # find the start of the data to begin cumulating (expanding)
            if np.isnan(X[:, 2]).all() is False:
                start = np.where(~np.isnan(X[:, 2]))[0][0]
            else:
                start = X[:, 1].size + 1
            periods = start

        if ((i < periods) or (i < start) or ((cum is False) and
                np.isnan(X[start:i + 1]).any())):
            ols_details['r2'] = np.nan
            ols_details['r2-adj'] = np.nan
        else:
            model = sm.OLS(y[start:i + 1], X[start:i + 1],
                           missing='drop').fit()
            ols_details['r2'] = model.rsquared
            ols_details['r2-adj'] = model.rsquared_adj

        ols.append(copy.copy(ols_details))

    ret_df = pd.DataFrame(ols, index=y.index)

    return ret_df


def dd_profile(ret):
    """Creates a summary profile each of the drawdowns.
    """

    # TODO: reivist how you want the start and end periods to show up
    drawdowns = []
    curr_dd = {'size': 0, 'length': 0, 'start': None, 'end': None}
    new_dd_flag = True

    for i in range(1, len(ret)):
        if ret[i] != 0 and new_dd_flag is True:
            new_dd_flag = False
            curr_dd['size'] = ret[i]
            curr_dd['length'] = 1
            curr_dd['start'] = ret.index[max(0, i - 1)]
            curr_dd['end'] = None
        elif ret[i] != 0:
            curr_dd['size'] = min(curr_dd['size'], ret[i])
            curr_dd['length'] += 1
        elif ret[i] == 0 and new_dd_flag is False:
            new_dd_flag = True
            curr_dd['end'] = ret.index[i]
            drawdowns.append(copy.copy(curr_dd))

    if new_dd_flag is False:
        drawdowns.append(copy.copy(curr_dd))

    df = pd.DataFrame(drawdowns)

    return df


def dd_calc_series(ret):
    """Create a compound drawdown series from a log return series.
    """

    dd_series = ret.copy()
    running = 0.0

    for j in range(0, len(ret)):
        running = min(0, running + ret[j])
        dd_series[j] = running

    return (np.exp(dd_series) - 1)


def rolling_dd_series(ret, periods, cum=False):
    """Calculates a rolling drawdown series.
    """

    # TODO: make drawdown functions much more efficient
    # TODO: pretty sure there is a better way to do this with matrix math...

    # cumsum = df.cumsum()
    # total = cumsum[-1:].T
    # dd = cumsum.min()[cumsum.min() < 0]

    drawdowns = []
    dd = {
        'count': 0, 'size_avg': None, 'length_avg': None, 'size_med': None,
        'length_med': None, 'max_size': None, 'max_size_length': None,
        'max_size_start': None, 'max_size_end': None,
        'max_length_size': None, 'max_length': None,
        'max_length_start': None, 'max_length_end': None,
        'curr_size': None, 'curr_max_size': None, 'curr_length': None,
        'curr_start': None
    }

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    if (not valid_period(periods)) or (periods < MIN_OBS_THRESH):
        return pd.DataFrame()

    # add an empty row for the first return series item
    drawdowns.append(copy.copy(dd))

    for i in range(1, len(ret)):

        if cum is False:
            if int(periods) > i:
                start = i
            else:
                start = i - int(periods)
        else:
            start = 1

        sub_series = ret[start:i + 1].copy()
        dd_series = dd_calc_series(sub_series)
        df = dd_profile(dd_series)

        if df.empty:
            dd['count'] = 0
            dd['size_avg'] = None
            dd['length_avg'] = None
            dd['size_med'] = None
            dd['length_med'] = None
            dd['max_size'] = None
            dd['max_size_length'] = None
            dd['max_size_start'] = None
            dd['max_size_end'] = None
            dd['max_length_size'] = None
            dd['max_length'] = None
            dd['max_length_start'] = None
            dd['max_length_end'] = None
            df['curr_size'] = None
            df['curr_max_size'] = None
            df['curr_length'] = None
            df['curr_start'] = None
        else:
            max_dd_size_row = df['size'].idxmin()
            max_dd_length_row = df['length'].idxmax()
            dd['count'] = df['size'].count()
            dd['size_avg'] = df['size'].mean()
            dd['length_avg'] = df['length'].mean()
            dd['size_med'] = df['size'].median()
            dd['length_med'] = df['length'].median()
            dd['max_size'] = df.ix[max_dd_size_row, 'size']
            dd['max_size_length'] = df.ix[max_dd_size_row, 'length']
            dd['max_size_start'] = df.ix[max_dd_size_row, 'start']
            dd['max_size_end'] = df.ix[max_dd_size_row, 'end']
            dd['max_length_size'] = df.ix[max_dd_length_row, 'size']
            dd['max_length'] = df.ix[max_dd_length_row, 'length']
            dd['max_length_start'] = df.ix[max_dd_length_row, 'start']
            dd['max_length_end'] = df.ix[max_dd_length_row, 'end']
            if pd.isnull(df['end'][df.index[-1]]):
                dd['curr_size'] = dd_series[-1]
                dd['curr_max_size'] = df['size'][df.index[-1]]
                dd['curr_length'] = df['length'][df.index[-1]]
                dd['curr_start'] = df['start'][df.index[-1]]
            else:
                dd['curr_size'] = None
                dd['curr_max_size'] = None
                dd['curr_length'] = None
                dd['curr_start'] = None

        drawdowns.append(copy.copy(dd))

    ret_df = pd.DataFrame(drawdowns, index=ret.index)
    return ret_df


def rolling_vol(series_data, periods, annual_factor=12, annualize_flag=True,
                cum=False, semivol=None, semivol_threshold=None):
    """Calculates a volatility over a specified period.

    Volatility is simply the standard deviation over some time frame. The 
    shorter the time period the less statistically useful vol is; however, 
    long periods can also dampen volatility giving a false impression of risk. 
    Most people believe volatility for financial data series is 'nonstationary' 
    meaning volatility changes over time; therefore, analyzing rolling vol is 
    useful to get a better understanding of how the risk changes over time. 
    The function by default annualizes vol to make it easier to interpret. 
    Semivol (i.e., the standard deviation of observations above or below some 
    specific value) can also be calculated by passing in semivol and 
    semivol_threshold values. Semivol can be useful for assessing the downside 
    risk. Vol is just one measure of risk, blah blah blah, save us the lecture.

    Args:
        series_data (Pandas DataFrame of float): The series to calculate the 
            vol of.
        periods (float): The rolling period's length of time.
        annual_factor (float, optional): The factor used to annualize the vol.
            (default is 12)
        annualize_flag (bool, optional): A flag to indicate whether or not vol 
            should be stated annually.
            (default is True)
        cum (bool, optional): Calculates a cumulative volatility instead of a 
            rolling (moving window).
            (default is False)
        semivol (int, optional). Flag indicating whether to calculate a semi 
            vol or not. Defaults to calculate a normal vol, 1 uses series 
            values above the semivol_threshold, any other number uses series 
            values below the semivol_threshold. Similar concept to semibeta.
            (default is None)
        semivol_threshold (float, optional): The threshold value used to 
            determine which series values should be included in the semivol 
            calculation.
            (default is None)

    Returns:
        Pandas DataFrame of float: A series of rolling volatilities (stated 
        annually by default) at each each observation point of the passed in 
        series.
    """

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    if (not valid_period(periods)) or (periods < MIN_OBS_THRESH):
        return np.nan

    if cum is False:
        if semivol is None:
            ret = series_data.rolling(window=periods).std()
        elif semivol == 1:
            ret = np.maximum(series_data,
                             semivol_threshold).rolling(window=periods).std()
        else:
            ret = np.minimum(series_data,
                             semivol_threshold).rolling(window=periods).std()
    else:
        if semivol is None:
            ret = series_data.expanding().std()
        elif semivol == 1:
            ret = np.maximum(series_data, semivol_threshold).expanding().std()
        else:
            ret = np.minimum(series_data, semivol_threshold).expanding().std()

    if annualize_flag is True:
        annualize = np.sqrt(annual_factor)
    else:
        annualize = 1

    return ret * annualize


def _risk_adj_ratio(topic_srs, ref_srs, risk_measure):
    """Helper function to calculate risk adjusted ratios.

    Many 'risk-adjusted' measures are simply the excess between two series 
    divided by some risk measure. For example, the Sharpe ratio is the excess 
    return (return over the risk-free rate) divided by volatility. This helper 
    function simply calcs the arithmetic difference (pass logs for returns!) of 
    the passed in series and divides by the passed in risk measure.

    Args:
        topic_srs (Pandas DataFrame of float): The series of interest.
        ref_srs (Pandas DataFrame of float): The reference/benchmark series.
        risk_measure (Pandas DataFrame of float): The risk measure used to 
            'adjust' the excess.
    
    Returns:
        Pandas DataFrame of float: A risk-adjusted ratio series by calculating 
        the excess divded by the risk measure. 
    """

    return excess(topic_srs, ref_srs) / risk_measure


def sharpe(topic_srs, ref_srs, topic_vol):
    """Calcs the topic's excess return per unit of topic vol.

    Proposed by William Sharpe, the ratio of the topic series' return over the 
    risk-free rate (e.g., short-term cash rate for global macro) dividend by 
    the topic series' volatility.

    Args:
        topic_srs (Pandas DataFrame of float): The topic series of interest.
        ref_srs (Pandas DataFrame of float): The risk-free series.
        topic_vol (Pandas DataFrame of float): The volatility of the topic 
            series.
    
    Returns:
        Panadas DataFrame of float: The Sharpe ratio: excess return divided by 
        volatility.
    """
    
    return _risk_adj_ratio(topic_srs, ref_srs, topic_vol)


def info_ratio(topic_srs, ref_srs, active_vol):
    """Calcs the active return per unit of active vol.

    The information ratio is similar to the Sharpe ratio; however, it uses the 
    active return (i.e., the topic series' excess return v. some benchmark) 
    divided by the active volatility (the standard deviaiton of returns v. that 
    same benchamrk). Values of 1.0 and higher are usually considered top decile.

    Args:
        topic_srs (Pandas DataFrame of float): The topic series of interest.
        ref_srs (Pandas DataFrame of float): A benchmark series.
        active_vol (Pandas DataFrame of float): The volatility of the active 
            return.
    
    Returns:
        Panadas DataFrame of float: The information ratio: active return 
        divided by active volatility.
    """
    
    return _risk_adj_ratio(topic_srs, ref_srs, active_vol)


def sortino(topic_srs, ref_srs, topic_downside_vol):
    """Calcs the excess return per unit of downside vol.

    Investors are often more concerned with downside risk, so the Sortino 
    ratio evaluates the excess or active return compared to only the downside 
    volatility.

    Args:
        topic_srs (Pandas DataFrame of float): The topic series of interest.
        ref_srs (Pandas DataFrame of float): A reference series.
        active_vol (Pandas DataFrame of float): The downside volatility of the 
            excess/active return.
    
    Returns:
        Panadas DataFrame of float: The Sortino ratio: excess/active return 
        divided by downside volatility.
    """
    
    return _risk_adj_ratio(topic_srs, ref_srs, topic_downside_vol)


def vol_skew(vol_upside, vol_downside):
    """Calc the ratio of upside volatility to downside volatility.

    The ratio of upside and downside volatilities is called the vol skew. 
    Uses std dev not variance unlike some defs to calculate.

    Args:
        vol_upside (Pandas DataFrame of float): The upside volatility.
        vol_downside (Pandas DataFrame of float): The downside volatility.
    
    Returns:
        Pandas DataFrame of float: The ratio of the upside to downside vol.
    """
    
    return vol_upside / vol_downside


def mod_treynor(topic_srs, rf_srs, mkt_vol):
    """Modified treynor calcs ratio of excess return to market volatility.
    
    Treynor ratio calculation which is simply the return v. risk-free rate 
    divided by the market volatility. Although we classify it as part of the 
    excess return metric suite, it requires a benchmark return series to 
    define what 'the market' is.

    Args:
        topic_srs (Pandas DataFrame of float): The topic return series.
        rf_srs (Pandas DataFrame of float): The risk-free return series.
        mkt_vol (Pandas DataFrame of float): The market volatility series.
    
    Returns:
        Pandas DataFrame of float: Excess return dividied by market volatility.
    """
    
    return _risk_adj_ratio(topic_srs, rf_srs, mkt_vol)


def rolling_corr(srs_1, srs_2, periods, cum=False):
    """Calculates rolling correlation.

    Correlations are ineherently unstable, so rolling correlation analysis 
    helps provide a more robust analysis.

    Args:
        srs_1 (Pandas DataFrame of float): The first series to correlate.
        srs_2 (Pandas DataFrame of float): The second series to correlate.
        periods (float): The number of periods in the rolling window.
        cum (bool, optional): Boolean to indicate if a cumulative calc should 
            be used.
            (default is False)
    
    Returns:
        Pandas DataFrame of float: A rolling correlation series.
    """

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    # don't return values if being asked to calculate something shorter than
    # 1 period or not a multiple of the periodicity
    if not valid_period(periods) or (periods < MIN_OBS_THRESH):
        return np.nan

    return (srs_1.rolling(window=periods).corr(srs_2)
        if cum is False else srs_1.expanding().corr(srs_2))


def rolling_r2(srs_1, srs_2, periods, cum=False):
    """Calcs rolling R2 which is just corr^2.

    The rolling R2 is the square of the correlation for single variate 
    regression, so this function simply calls the rolling_correlation function.

    Args:
        srs_1 (Pandas DataFrame of float): The first data series to 
            correlate.
        srs_2 (Pandas DataFrame of float): The second data series to 
            correlate.
        periods (float): The number of periods in the rolling window.
        cum (bool, optional): Boolean flag to indicate if the correlation is 
            cumulative or not.
            (default is False)
    
    Returns:
        Pandas DataFrame of float: A series of rolling R2s.
    """

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    # don't return values if being asked to calculate something shorter than
    # 1 period or not a multiple of the periodicity
    if not valid_period(periods) or (periods < MIN_OBS_THRESH):
        return np.nan

    return rolling_corr(srs_1, srs_2, periods, cum) ** 2


def rolling_beta(topic_srs, ref_srs, periods, cum=False, semibeta=None, 
                 semibeta_threshold=None):
    """Calculates the rolling beta of the topic series.

    A beta is the coefficient calculated in a single variable linear 
    regression. In finance, the beta is one way to describe the sensitivity 
    (or risk) of something against a benchmark. For example, if a stock is 
    called 'high beta' it means that it's value changes an order of magnitude 
    larger than the over stock market. Just like correlations, vols, and other 
    measures, betas can move around a lot.

    Args:
        topic_srs (Pandas DataFrame of float): The dependent variable series.
        ref_srs (Pandas DataFrame of float): The independent variable series.
        periods (float): The number of periods in the rolling window.
        cum (bool, optional): Boolean flag to indicate calculating a cumulative 
            beta or not.
            (default is False)
        semibeta (int, optional): Default of none calcs normal beta, 1 calcs 
            the upside beta (beta above a threshold value), any other number 
            calcs the downside beta (beta below a threshold value). Similar 
            concept to semivol.
            (default is None)
        semibeta_threshold (float, optional): The threshold value to use when 
            calculating a semibeta.
            (default is None)
    """

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    # don't return values if being asked to calculate something shorter than
    # 1 period or not a multiple of the periodicity
    if not valid_period(periods) or (periods < MIN_OBS_THRESH):
        return np.nan

    rs_topic = topic_srs.copy()
    rs_mkt = ref_srs.copy()

    if semibeta == 1:
        rs_mkt = np.maximum(rs_mkt, semibeta_threshold)
        cond = rs_mkt == 0
        rs_topic[cond] = rs_mkt[cond]
    elif semibeta == -1:
        rs_mkt = np.minimum(rs_mkt, semibeta_threshold)
        cond = rs_mkt == 0
        rs_topic[cond] = rs_mkt[cond]

    if cum is False:
        covar = rs_topic.rolling(window=periods).cov(rs_mkt)
        var = rs_mkt.rolling(window=periods).var()
    else:
        covar = rs_topic.expanding().cov(rs_mkt)
        var = rs_mkt.expanding().var()

    return covar / var


def jensens_alpha(topic_srs, ref_srs, rf_srs, beta):
    """Calculates jensen's alpha.

    Jensen's alpha calculates a risk-adjusted excess return. Many asset 
    managers will present excess return as alpha; however, the could have 
    achieves excess returns by simply running higher risk v. their benchmark. 
    Jense's alpha is effectively the y-intercept if one were to regress manager 
    returns against their benchmark returns.

    Args:
        topic_srs (Pandas DataFrame of float): The topic series of interest.
        bmk_srs (Pandas DataFrame of float): The benchmark series to calculate 
            alpha / risk-adjusted outperformance against.
        rf_srs (Pandas DataFrame of float): The risk-free rate series.
        beta (Pandas DataFrame of float): The topic series beta (v. the 
            benchmark series).
    
    Returns:
        float: Jensen's alpha.
    """

    return topic_srs - rf_srs - (beta * (ref_srs - rf_srs))


def m2(topic_srs, topic_vol, sharpe, ref_vol, cum=False, annual_factor=1):
    """Calcs m2 return which is a port to mkt vol adjusted return measure.
    
    The Sharpe ratio can be difficult to interpret since it's a ratio, so M2 
    converts a Sharpe to a return number.

    Args:
        topic_srs (Pandas DataFrame of float): The series of interest.
        topic_vol (Pandas DataFrame of float): The volatility of the topic 
            series.
        sharpe (Pandas DataFrame of float): The Sharpe ratio of the topic.
        ref_vol (Pandas DataFrame of float): The reference series' volatility. 
            The M2 return calculated with be comparable to this reference 
            series' return.
        cum (bool, optional): Boolean flag to inidicate calculating a 
            cumulative value.
            (default is False)
        annual_factor (float, optional): The factor used to annualize the M2 
            value.
            (default is 1)
    
    Returns:
        Pantas DataFrame of float: M2 return.
    """

    return (topic_srs + (sharpe * (ref_vol - topic_vol))) * annual_factor


def m2_excess(topic_m2_srs, ref_srs):
    """Calcs the m2 excess return v. the reference.

    Args:
        topic_m2_srs (Panads DataFrame of float): An m2 return or series of returns.
        ret_srs (Pandas DataFrame of float): The benchmark return or series of 
            returns.
    
    Returns:
        Pandas DataFrame of float: An excess return series.
    """
    
    return excess(topic_m2_srs, ref_srs)


def change_analysis(src_df, src_col_topic='topic', src_col_rf=None,
                    src_col_bmk=None, annualize_flag=True, periods=[],
                    measures=[], m_lbls=Measures(), 
                    measures_exclude_flag=False):
    """Generates all of the change analytics data from a Pandas DataFrame.

    [longer desc]

    Args:
        src_df (Pandas DataFrame): The source data set.
        src_col_topic (str, optional): Column name of the topic series.
            (default is 'topic')
        src_col_rf (str, optional): Column name of the risk-free series.
            (default is None)
        src_col_bmk (str, optional): Column name of the benchmark/reference 
            series.
            (default is None)
        annualize_flag (bool, optional): Indicates whether calculations should 
            be stated annually.
            (default is True)
        periods (list of TimePeriod, optional): List of rolling time periods 
            to analyze.
            (default is empty list)
        measures (list of str, optional): A list of measures to be calculated. 
            If no measure list is specified, all supported measures are 
            calculated.
            (default is empty list)
        m_lbls (Measures, optional): A Measures object that defines 
            the labels for each measure. If no object is passed, all measures 
            use their default labels.
            (default is a Measures instance)
        measures_exclude_flag (bool, optional): Flag to indicate whether to 
            exclude the list of measures passed in from being calculated. To 
            exclude the drawdown metrics, add Measures.DRAWDOWN as an item to 
            the measures list.
            (default is False)
    
    Returns:
        Pandas DataFrame: A complete change analysis using the measures and 
        time periods specified (default all).
    """
    
    # create change series data structure
    srs_type_topic = 'topic'
    srs_type_rf = 'rf'
    srs_type_bmk = 'bmk'
    srs_type_excess = 'excess'
    srs_type_active = 'active'
    chg_srs = [
        srs_type_topic,
        srs_type_rf,
        srs_type_bmk,
        srs_type_excess,
        srs_type_active
    ]
 
    # set default measure list to all if none passed in
    if not measures:
        for measure in m_lbls:
            measures.append(measure)
    
    # remove measures if the list passed in meant to exclude
    if measures_exclude_flag == True:
        tmp_measures = []
        for measure in m_lbls:
            if (measure not in measures) and not ((Measures.DRAWDOWN in 
                                                   measures) and 
                                                  ('dd' in measure)):
                tmp_measures.append(measure)
        measures = tmp_measures

    # create time periods windows
    the_periodicity = get_periodicity(src_df.index)
    periodicity = the_periodicity.ann_factor
    tm_periods = create_tm_periods(src_df.index.size, the_periodicity, periods)

    src_chg_streams = {
        srs_type_topic: src_col_topic,
        srs_type_rf: src_col_rf,
        srs_type_bmk: src_col_bmk
    }

    # create change/return time period columns
    # there is probably a better way to check if certain measures should be
    # calc'ed v. using a bunch of if statements, but i'm lazy
    cols = []
    for key in src_chg_streams:
        for measure in [m_lbls.level, m_lbls.level_ln, m_lbls.chg_rel]:
            if measure in measures:
                cols.append([key, measure, tm_periods[TP_SPOT].name])
        
        for key2 in tm_periods:
            for measure in [m_lbls.chg_ln, m_lbls.chg_abs, m_lbls.vol_lvl, 
                            m_lbls.vol_lvl_ln]:
                if measure in measures:
                    cols.append([key, measure, tm_periods[key2].name])
    
    for key in [srs_type_excess, srs_type_active]:
        if m_lbls.chg_rel in measures:
            cols.append([key, m_lbls.chg_rel, tm_periods[TP_SPOT].name])
        for key2 in tm_periods:
            if m_lbls.chg_ln in measures:
                cols.append([key, m_lbls.chg_ln, tm_periods[key2].name])
    
    # create vol time period columns
    for key in chg_srs:
        for key2 in tm_periods:
            if tm_periods[key2].stat_flag is True:
                for measure in [m_lbls.vol_cmp, m_lbls.vol_ln]:
                    if measure in measures:
                        cols.append([key, measure, tm_periods[key2].name])
    
    # create time period columns for other measures
    for key in tm_periods:
        if tm_periods[key].stat_flag is True:
            for measure in [[m_lbls.sharpe_ln, srs_type_excess], 
                            [m_lbls.info_ratio_ln, srs_type_active], 
                            [m_lbls.vol_downside_ln, srs_type_excess], 
                            [m_lbls.vol_upside_ln, srs_type_excess], 
                            [m_lbls.vol_downside_ln, srs_type_active], 
                            [m_lbls.vol_upside_ln, srs_type_active], 
                            [m_lbls.sortino_ln, srs_type_excess], 
                            [m_lbls.sortino_ln, srs_type_active], 
                            [m_lbls.vol_skew_ln, srs_type_excess], 
                            [m_lbls.vol_skew_ln, srs_type_active], 
                            [m_lbls.mod_treynor_ln, srs_type_active], 
                            [m_lbls.corr_lvl, srs_type_active], 
                            [m_lbls.corr_lvl_ln, srs_type_active], 
                            [m_lbls.corr_ln, srs_type_active], 
                            [m_lbls.r2_ln, srs_type_active], 
                            [m_lbls.beta_ln, srs_type_active], 
                            [m_lbls.beta_up_ln, srs_type_active], 
                            [m_lbls.beta_down_ln, srs_type_active], 
                            [m_lbls.jensens_alpha_ln, srs_type_active], 
                            [m_lbls.m2_ln, srs_type_active], 
                            [m_lbls.m2_excess_ln, srs_type_active]]:
                if measure[0] in measures:
                    cols.append([measure[1], measure[0], tm_periods[key].name])
            for key2 in m_lbls.dd_cmp:
                if (key2 in measures) or (Measures.DRAWDOWN in measures):
                    cols.append([srs_type_topic, key2, tm_periods[key].name])
    
    df = pd.DataFrame(index=src_df.index,
                      columns=pd.MultiIndex.from_tuples(
                        cols, names=['srs_type', 'measure', 'period']))

    # calc spot changes
    for key in src_chg_streams:
        if (key, m_lbls.level, tm_periods[TP_SPOT].name) in df.columns:
            df.loc[:, (key, m_lbls.level, tm_periods[TP_SPOT].name)] = \
                src_df[src_chg_streams[key]]
        if (key, m_lbls.level_ln, tm_periods[TP_SPOT].name) in df.columns:
            df.loc[:, (key, m_lbls.level_ln, tm_periods[TP_SPOT].name)] = \
                np.log(df[key][m_lbls.level][tm_periods[TP_SPOT].name])
        if (key, m_lbls.chg_ln, tm_periods[TP_SPOT].name) in df.columns:
            df.loc[:, (key, m_lbls.chg_ln, tm_periods[TP_SPOT].name)] = np.log(
                    1 + df[key][m_lbls.level][tm_periods[TP_SPOT].name].pct_change(1))
        if (key, m_lbls.chg_rel, tm_periods[TP_SPOT].name) in df.columns:
            df.loc[:, (key, m_lbls.chg_rel, tm_periods[TP_SPOT].name)] = np.exp(
                    df[key][m_lbls.chg_ln][tm_periods[TP_SPOT].name]) - 1
        if (key, m_lbls.chg_abs, tm_periods[TP_SPOT].name) in df.columns:
            df.loc[:, (key, m_lbls.chg_abs, tm_periods[TP_SPOT].name)] = \
                df[key][m_lbls.level][tm_periods[TP_SPOT].name].diff(1)

    # calc excess returns
    if (srs_type_excess, m_lbls.chg_ln, tm_periods[TP_SPOT].name) in df.columns:
        df.loc[:, (srs_type_excess, m_lbls.chg_ln, tm_periods[TP_SPOT].name)] = excess(
                df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                df[srs_type_rf][m_lbls.chg_ln][tm_periods[TP_SPOT].name])
    if (srs_type_excess, m_lbls.chg_rel, tm_periods[TP_SPOT].name) in df.columns:
        df.loc[:, (srs_type_excess, m_lbls.chg_rel, tm_periods[TP_SPOT].name)] = \
            np.exp(df[srs_type_excess][m_lbls.chg_ln][tm_periods[TP_SPOT].name]) - 1

    # calc active returns
    if (srs_type_active, m_lbls.chg_ln, tm_periods[TP_SPOT].name) in df.columns:
        df.loc[:, (srs_type_active, m_lbls.chg_ln, tm_periods[TP_SPOT].name)] = excess(
                df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                df[srs_type_bmk][m_lbls.chg_ln][tm_periods[TP_SPOT].name])
    if (srs_type_active, m_lbls.chg_rel, tm_periods[TP_SPOT].name) in df.columns:
        df.loc[:, (srs_type_active, m_lbls.chg_rel, tm_periods[TP_SPOT].name)] = \
            np.exp(df[srs_type_active][m_lbls.chg_ln][tm_periods[TP_SPOT].name]) - 1

    for ret in chg_srs:
        for key in tm_periods:
            # only calc periodic returns if it's not the spot return
            if tm_periods[key] != tm_periods[TP_SPOT]:
                cum_sum = (key == TP_CUM)
                if (ret, m_lbls.chg_ln, tm_periods[key].name) in df.columns:
                    df.loc[:, (ret, m_lbls.chg_ln, tm_periods[key].name)] = rolling_chg(
                        df[ret][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                        tm_periods[key].ann_factor * periodicity,
                        periodicity,
                        annualize_flag,
                        cum_sum)
                # only calc the abs chg & vols for topic, bmk, and rfr streams
                if ret in src_chg_streams:
                    if (ret, m_lbls.chg_abs, tm_periods[key].name) in df.columns:
                        df.loc[:, (ret, m_lbls.chg_abs, tm_periods[key].name)] = rolling_chg(
                            df[ret][m_lbls.chg_abs][tm_periods[TP_SPOT].name],
                            tm_periods[key].ann_factor * periodicity,
                            periodicity,
                            False,
                            cum_sum)
                    if (ret, m_lbls.vol_lvl, tm_periods[key].name) in df.columns:
                        df.loc[:, (ret, m_lbls.vol_lvl, tm_periods[key].name)] = rolling_vol(
                            df[ret][m_lbls.level][tm_periods[TP_SPOT].name],
                            tm_periods[key].ann_factor * periodicity,
                            periodicity,
                            annualize_flag,
                            cum_sum)
                    if (ret, m_lbls.vol_lvl_ln, tm_periods[key].name) in df.columns:
                        df.loc[:, (ret, m_lbls.vol_lvl_ln, tm_periods[key].name)] = rolling_vol(
                            df[ret][m_lbls.level_ln][tm_periods[TP_SPOT].name],
                            tm_periods[key].ann_factor * periodicity,
                            periodicity,
                            annualize_flag,
                            cum_sum)

    # calc vols for compound + log returns
    for ret in chg_srs:
        for key in tm_periods:
            if tm_periods[key].stat_flag is True:
                if key == TP_CUM:
                    cum = True
                else:
                    cum = False
                if (ret, m_lbls.vol_ln, tm_periods[key].name) in df.columns:
                    df.loc[:, (ret, m_lbls.vol_ln, tm_periods[key].name)] = rolling_vol(
                        df[ret][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                        tm_periods[key].ann_factor * periodicity,
                        periodicity,
                        annualize_flag,
                        cum)
                if (ret, m_lbls.vol_cmp, tm_periods[key].name) in df.columns:
                    df.loc[:, (ret, m_lbls.vol_cmp, tm_periods[key].name)] = rolling_vol(
                        df[ret][m_lbls.chg_rel][tm_periods[TP_SPOT].name],
                        tm_periods[key].ann_factor * periodicity,
                        periodicity,
                        annualize_flag,
                        cum)

    for key in tm_periods:
        if tm_periods[key].stat_flag is True:
            if key == TP_CUM:
                annual_factor = np.maximum(
                    df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name].expanding().count(),
                    periodicity) / periodicity
                cum = True
            else:
                if annualize_flag is True:
                    annual_factor = 1
                else:
                    annual_factor = key.ann_factor
                cum = False

            # calc sharpe ratio
            if (srs_type_excess, m_lbls.sharpe_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_excess, m_lbls.sharpe_ln, tm_periods[key].name)] = sharpe(
                    df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_rf][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_topic][m_lbls.vol_ln][tm_periods[key].name])

            # calc info ratio
            if (srs_type_active, m_lbls.info_ratio_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.info_ratio_ln, tm_periods[key].name)] = \
                    info_ratio(df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / 
                            annual_factor, 
                            df[srs_type_bmk][m_lbls.chg_ln][tm_periods[key].name] / 
                            annual_factor,
                            df[srs_type_active][m_lbls.vol_ln][tm_periods[key].name])

            # calc excess up/down vols
            if (srs_type_excess, m_lbls.vol_downside_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_excess, m_lbls.vol_downside_ln, tm_periods[key].name)] = \
                    rolling_vol(df[srs_type_excess][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                tm_periods[key].ann_factor * periodicity,
                                periodicity, annualize_flag, cum, -1, 0)
            if (srs_type_excess, m_lbls.vol_upside_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_excess, m_lbls.vol_upside_ln, tm_periods[key].name)] = \
                    rolling_vol(df[srs_type_excess][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                tm_periods[key].ann_factor * periodicity,
                                periodicity, annualize_flag, cum, 1, 0)

            # calc active up/down vols
            if (srs_type_active, m_lbls.vol_downside_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.vol_downside_ln, tm_periods[key].name)] = \
                    rolling_vol(df[srs_type_active][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                tm_periods[key].ann_factor * periodicity,
                                periodicity, annualize_flag, cum, -1, 0)
            if (srs_type_active, m_lbls.vol_upside_ln, tm_periods[key].name) in df.columns:          
                df.loc[:, (srs_type_active, m_lbls.vol_upside_ln, tm_periods[key].name)] = \
                    rolling_vol(df[srs_type_active][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                tm_periods[key].ann_factor * periodicity,
                                periodicity, annualize_flag, cum, 1, 0)

            # calc sortino ratios
            if (srs_type_excess, m_lbls.sortino_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_excess, m_lbls.sortino_ln, tm_periods[key].name)] = sortino(
                    df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_rf][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_excess][m_lbls.vol_downside_ln][tm_periods[key].name])
            if (srs_type_active, m_lbls.sortino_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.sortino_ln, tm_periods[key].name)] = sortino(
                    df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_bmk][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_active][m_lbls.vol_downside_ln][tm_periods[key].name])

            # calc vol skew ratios
            if (srs_type_excess, m_lbls.vol_skew_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_excess, m_lbls.vol_skew_ln, tm_periods[key].name)] = \
                    vol_skew(df[srs_type_excess][m_lbls.vol_upside_ln][tm_periods[key].name],
                             df[srs_type_excess][m_lbls.vol_downside_ln][tm_periods[key].name])
            if (srs_type_active, m_lbls.vol_skew_ln, tm_periods[key].name) in df.columns:       
                df.loc[:, (srs_type_active, m_lbls.vol_skew_ln, tm_periods[key].name)] = \
                    vol_skew(df[srs_type_active][m_lbls.vol_upside_ln][tm_periods[key].name],
                             df[srs_type_active][m_lbls.vol_downside_ln][tm_periods[key].name])

            # calc modified treynor ratio
            if (srs_type_active, m_lbls.mod_treynor_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.mod_treynor_ln, tm_periods[key].name)] = \
                    mod_treynor(df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / 
                                annual_factor, 
                                df[srs_type_rf][m_lbls.chg_ln][tm_periods[key].name] / 
                                annual_factor,
                                df[srs_type_bmk][m_lbls.vol_ln][tm_periods[key].name])

            # calc correlations & r2
            if (srs_type_active, m_lbls.corr_lvl, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.corr_lvl, tm_periods[key].name)] = \
                    rolling_corr(df[srs_type_topic][m_lbls.level][tm_periods[TP_SPOT].name],
                                 df[srs_type_bmk][m_lbls.level][tm_periods[TP_SPOT].name],
                                 tm_periods[key].ann_factor * periodicity,cum)
            if (srs_type_active, m_lbls.corr_lvl_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.corr_lvl_ln, tm_periods[key].name)] = \
                    rolling_corr(df[srs_type_topic][m_lbls.level_ln][tm_periods[TP_SPOT].name],
                                 df[srs_type_bmk][m_lbls.level_ln][tm_periods[TP_SPOT].name],
                                 tm_periods[key].ann_factor * periodicity, cum)
            if (srs_type_active, m_lbls.corr_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.corr_ln, tm_periods[key].name)] = \
                    rolling_corr(df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 df[srs_type_bmk][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 tm_periods[key].ann_factor * periodicity, cum)
            if (srs_type_active, m_lbls.r2_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.r2_ln, tm_periods[key].name)] = \
                    df[srs_type_active][m_lbls.corr_ln][tm_periods[key].name] ** 2

            # calc betas to benchmark
            if (srs_type_active, m_lbls.beta_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.beta_ln, tm_periods[key].name)] = \
                    rolling_beta(df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 df[srs_type_bmk][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 tm_periods[key].ann_factor * periodicity, cum)
            if (srs_type_active, m_lbls.beta_up_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.beta_up_ln, tm_periods[key].name)] = \
                    rolling_beta(df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 df[srs_type_bmk][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 tm_periods[key].ann_factor * periodicity, cum, 1, 0)
            if (srs_type_active, m_lbls.beta_down_ln, tm_periods[key].name) in df.columns:           
                df.loc[:, (srs_type_active, m_lbls.beta_down_ln, tm_periods[key].name)] = \
                    rolling_beta(df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 df[srs_type_bmk][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                                 tm_periods[key].ann_factor * periodicity, cum, -1, 0)

            # calc jensen's alpha -- real alpha, not just active return b.s.
            if (srs_type_active, m_lbls.jensens_alpha_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.jensens_alpha_ln, tm_periods[key].name)] = \
                    jensens_alpha(df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / 
                                annual_factor,
                                df[srs_type_bmk][m_lbls.chg_ln][tm_periods[key].name] / 
                                annual_factor,
                                df[srs_type_rf][m_lbls.chg_ln][tm_periods[key].name] / 
                                annual_factor,
                                df[srs_type_active][m_lbls.beta_ln][tm_periods[key].name])

            # calc m2 return and excess return
            if (srs_type_active, m_lbls.m2_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.m2_ln, tm_periods[key].name)] = m2(
                    df[srs_type_topic][m_lbls.chg_ln][tm_periods[key].name] / annual_factor,
                    df[srs_type_topic][m_lbls.vol_ln][tm_periods[key].name],
                    df[srs_type_excess][m_lbls.sharpe_ln][tm_periods[key].name],
                    df[srs_type_bmk][m_lbls.vol_ln][tm_periods[key].name], cum, annual_factor)
            if (srs_type_active, m_lbls.m2_excess_ln, tm_periods[key].name) in df.columns:
                df.loc[:, (srs_type_active, m_lbls.m2_excess_ln, tm_periods[key].name)] = \
                    m2_excess(df[srs_type_active][m_lbls.m2_ln][tm_periods[key].name], 
                              df[srs_type_bmk][m_lbls.chg_ln][tm_periods[key].name])
                        
            # calculate rolling drawdowns. algo needs to be rewritten -- thing
            # is insanely slow
            if (srs_type_topic, m_lbls.dd_series_cmp, tm_periods[key].name) in df.columns:
                dd_df = rolling_dd_series(
                    df[srs_type_topic][m_lbls.chg_ln][tm_periods[TP_SPOT].name],
                    tm_periods[key].ann_factor * periodicity, cum)
                if not dd_df.empty:
                    df.loc[:, (srs_type_topic, m_lbls.dd_series_cmp, tm_periods[key].name)] = np.nan
                    df.loc[:, (srs_type_topic, m_lbls.dd_count_cmp, tm_periods[key].name)] = dd_df['count']
                    df.loc[:, (srs_type_topic, m_lbls.dd_avg_size_cmp, tm_periods[key].name)] = \
                        dd_df['size_avg']
                    df.loc[:, (srs_type_topic, m_lbls.dd_avg_periods_cmp, tm_periods[key].name)] = \
                        dd_df['length_avg']
                    df.loc[:, (srs_type_topic, m_lbls.dd_med_size_cmp, tm_periods[key].name)] = \
                        dd_df['size_med']
                    df.loc[:, (srs_type_topic, m_lbls.dd_med_periods_cmp, tm_periods[key].name)] = \
                        dd_df['length_med']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_size_cmp, tm_periods[key].name)] = \
                        dd_df['max_size']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_size_periods_cmp, tm_periods[key].name)] = \
                        dd_df['max_size_length']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_size_start_cmp, tm_periods[key].name)] = \
                        dd_df['max_size_start']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_size_end_cmp, tm_periods[key].name)] = \
                        dd_df['max_size_end']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_length_size_cmp, tm_periods[key].name)] = \
                        dd_df['max_length_size']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_length_cmp, tm_periods[key].name)] = \
                        dd_df['max_length']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_length_start_cmp, tm_periods[key].name)] = \
                        dd_df['max_length_start']
                    df.loc[:, (srs_type_topic, m_lbls.dd_max_length_end_cmp, tm_periods[key].name)] = \
                        dd_df['max_length_end']
                    df.loc[:, (srs_type_topic, m_lbls.dd_curr_series_cmp, tm_periods[key].name)] = np.nan
                    df.loc[:, (srs_type_topic, m_lbls.dd_curr_size_cmp, tm_periods[key].name)] = \
                        dd_df['curr_size']
                    df.loc[:, (srs_type_topic, m_lbls.dd_curr_max_size_cmp, tm_periods[key].name)] = \
                        dd_df['curr_max_size']
                    df.loc[:, (srs_type_topic, m_lbls.dd_curr_length_cmp, tm_periods[key].name)] = \
                        dd_df['curr_length']
                    df.loc[:, (srs_type_topic, m_lbls.dd_curr_start_cmp, tm_periods[key].name)] = \
                        dd_df['curr_start']

    return df


def main(src_file='infile.xlsx', src_sheet='Sheet1', src_col_dt='date',
         src_col_topic='strategy', src_col_rfr='rfr', src_col_bmk='benchmark',
         tgt_file='outfile.xlsx', tgt_sheet='Sheet1'):
    """Sample main function to use outcomes."""

    # read source data
    xlsx = pd.ExcelFile(src_file)
    src_df = pd.read_excel(xlsx, src_sheet)
    src_df.index = src_df[src_col_dt]

    df = change_analysis(src_df, src_col_topic, src_col_rfr, src_col_bmk)

    # write data to excel file
    xlsx_writer = pd.ExcelWriter(tgt_file)
    '''
    df.reorder_levels(('period', 'measure', 'srs_type'), 1).sort_index(axis=1,
                     level=('period', 'measure', 'srs_type')).to_excel(
                             xlsx_writer, tgt_sheet)
    '''
    df.reorder_levels(('measure', 'period', 'srs_type'), 1).sort_index(axis=1,
                     level=('measure', 'period', 'srs_type')).to_excel(
                             xlsx_writer, tgt_sheet)
    xlsx_writer.save()


if __name__ == '__main__':
    #src_file = ('PATH\\returns\\msci_returns.xlsx')
    #src_file = ('PATH\\returns\\spx_returns_trd.xlsx')
    #src_sheet = 'Sheet1'
    #src_col_dt = 'date'
    #src_col_topic = 'strategy'
    #src_col_rf = 'rfr'
    #src_col_bmk = 'benchmark'
    #tgt_file = ('PATH\\returns\\msci_output.xlsx')
    #tgt_file = ('PATH\\returns\\spx_trd_out.xlsx')
    #tgt_sheet = 'Sheet1'
    #main()
    pass