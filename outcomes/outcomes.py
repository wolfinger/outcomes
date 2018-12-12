# -*- coding: utf-8 -*-

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
        min_days (float): Lower bound (in days) for autodetecting the periodicity.
        max_days (float): Upper bound (in days) for autodetecting the periodicity.
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
        name (str): A short label for the time period e.g., 1w = one week.
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
        """Prints TimePeriod class' attributes in CSV-eqsue string."""
        return (self.name + ', ' + str(self.ann_factor) + ', ' +
                str(self.stat_flag))


class Drawdown:
    def __init__(self, name, series, count, avg_size, avg_periods,
                 med_size, med_periods, max_size_size, max_size_periods,
                 max_size_start_dt, max_size_end_dt, max_length_size,
                 max_length_periods, max_length_start_dt, max_length_end_dt,
                 curr_series, curr_size, curr_max_size, curr_periods,
                 curr_start_dt):
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


PDCY_DAILY_CALENDAR = Periodicity('daily_calendar', 0, 1, 365.25)
PDCY_DAILY_TRADING = Periodicity('daily_trading', 1, 4, 252)
PDCY_WEEKLY = Periodicity('weekly', 4, 7, 52)
PDCY_BI_MONTHLY = Periodicity('bi-monthly', 7, 17, 24)
PDCY_MONTHLY = Periodicity('monthly', 17, 31, 12)
PDCY_SEMI_MONTHLY = Periodicity('semi-monthly', 31, 62, 6)
PDCY_QUARTERLY = Periodicity('quarterly', 62, 94, 4)
PDCY_SEMI_ANNUALLY = Periodicity('semi-annually', 94, 184, 2)
PDCY_ANNUALLY = Periodicity('annually', 184, 366, 1)

TP_SPOT = 'spot'
TP_1D = '1d'
TP_1W = '1w'
TP_2W = '2w'
TP_1M = '1m'
TP_6W = '6w'
TP_2M = '2m'
TP_3M = '3m'
TP_6M = '6m'
TP_1Y = '1y'
TP_2Y = '2y'
TP_3Y = '3y'
TP_5Y = '5y'
TP_10Y = '10y'
TP_WTD = 'wtd'  # 'to-date' future release
TP_MTD = 'mtd'  # 'to-date' future release
TP_QTD = 'qtd'  # 'to-date' future release
TP_YTD = 'ytd'  # 'to-date' future release
TP_CUM = 'cum'

MIN_OBS_THRESH = 6  # min num of observations necessary for certain statistics


def create_tm_periods(srs_size, periodicity, periods=[]):
    """Creates a standard set of common time periods used for analysis.
    
    Creates a set of valid TimePeriods that can be used with the time series
    data of interest. Will automatically drop TimePeriods that are too short
    for relevance (based on periodiity) and too long based on series length.

    Args:
        srs_size (int): The length/size of the time series data.
        periodicity (float): The periodicity of the time series data.
        periods (list of TimePeriods, optional): List of time periods to 
            generate.
            (default is empty)
    
    Returns:
        list of TimePeriods: A list of TimePeriods based on the series size & 
        periodicity and any passed in TimePeriods requested. Will drop 
        nonsensical TimePeriods.
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
        period: The period to be validated.
    
    Returns:
        bool: Returns true if the period is valid, false otherwise.
    """

    if (period < 1) or (period % 1 != 0):
        return False
    else:
        return True


def return_to_level(returns, lvl_start_val=100, calc_method='cmp'):
    """Takes a series of returns and converts it to levels.

    [longer desc]

    Args:
        returns (Pandas DataFrame of float). A return series in compound or log 
            space.
        lvl_start_val (float, optional). The value to start the index level at.
            (default is 100)
        cacl_method (str, optional). Create a compound or log level series.
            (default is compound)
    
    Returns:
        Pandas DataFrame of float: A level series based on the returns passed in.
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


def excess_return(topic_ret, market_ret):
    """Calculates excess returns of a topic series vs. a market series.

    Args:
        topic_ret (Pandas DataFrame of float): The topic return series.
        market_ret (Pandas DataFrame of float): The market return series.
    
    Returns:
        Pandas DataFrame of float: The difference between the topic series and 
        the market series.
    """
    return topic_ret - market_ret


def rolling_chg(return_data, periods, annual_factor=12, annualize_flag=True,
                cum=False, to_date=None):
    """Calculates a rolling change or returns (if passed log returns).

    Calculates a rolling period's value at any given point in time. Will step 
    through a series and at each observation date a rolling value is calculated.
    If the passed in series is returns data, it will be a rolling return, 
    otherwise it will be a rolling diff.

    Args:
        return_data (Pandas DataFrame of float): [desc].
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

    return (return_data.rolling(window=periods,
                                min_periods=periods).sum() / annualize
            if cum is False else return_data.cumsum())


def rolling_ols(y, X, periods, cum=False, to_date=None):
    """Calculates a rolling least squares regression.

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
    periods = int(np.round(periods)).astype(int)

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


def rolling_vol(return_data, periods, annual_factor=12, annualize_flag=True,
                cum=False, semivol=None, semivol_threshold=None):
    """Calculates an annualized volatility over a specified period.

    [desc]

    Args:
        return_data (Pandas DataFrame of float): [desc].
        periods (float): The rolling period's length of time.
        annual_factor (float, optional): [desc].
            (default is 12)
        annualize_flag (bool, optional): [desc].
            (default is True)
        cum (bool, optional): [desc].
            (default is False)
        semivol (int, optional). [desc].
            (default is None)
        semivol_threshold (float, optional): [desc].
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
            ret = return_data.rolling(window=periods).std()
        elif semivol == 1:
            ret = np.maximum(return_data,
                             semivol_threshold).rolling(window=periods).std()
        else:
            ret = np.minimum(return_data,
                             semivol_threshold).rolling(window=periods).std()
    else:
        if semivol is None:
            ret = return_data.expanding().std()
        elif semivol == 1:
            ret = np.maximum(return_data, semivol_threshold).expanding().std()
        else:
            ret = np.minimum(return_data, semivol_threshold).expanding().std()

    if annualize_flag is True:
        annualize = np.sqrt(annual_factor)
    else:
        annualize = 1

    return ret * annualize


def _risk_adj_ratio(topic_ret, hurdle_ret, risk_measure):
    """Helper function to calculate risk adjusted return ratios.
    """

    return (topic_ret - hurdle_ret) / risk_measure


def sharpe(topic_ret, bmk_ret, asset_vol):
    """Calcs the excess return per unit of asset vol.
    """
    
    return _risk_adj_ratio(topic_ret, bmk_ret, asset_vol)


def info_ratio(topic_ret, bmk_ret, active_vol):
    """Calcs the active return per unit of active vol.
    """
    
    return _risk_adj_ratio(topic_ret, bmk_ret, active_vol)


def sortino(topic_ret, bmk_ret, topic_downside_vol):
    """Calcs the excess return per unit of downside vol.
    """
    
    return _risk_adj_ratio(topic_ret, bmk_ret, topic_downside_vol)


def vol_skew(vol_upside, vol_downside):
    """Calc the ratio of upside volatility to downside volatility.

    The ratio of upside and downside volatilities is called the vol skew. 
    Uses std dev not variance unlike some defs to calculate.
    """
    
    return vol_upside / vol_downside


def mod_treynor(topic_ret, rf_ret, mkt_vol):
    """Modified treynor calcs ratio of excess return to market volatility.
    
    Treynor ratio calculation which is simply the excess return v. risk-free 
    rate divided by the market volatility. Although we classify it as part of 
    the excess return metric suite, it requires a benchmark return series 
    to define what "the market" is.
    """
    
    return _risk_adj_ratio(topic_ret, rf_ret, mkt_vol)


def rolling_corr(ret_stream_1, ret_stream_2, periods, cum=False):
    """Calculates rolling correlation.

    Correlations are ineherently unstable, so rolling correlation analysis 
    helps build a more robust analysis.

    Args:
        ret_stream_1 (Pandas DataFrame of float): The first series to correlate.
        ret_stream_2 (Pandas DataFrame of float): The second series to 
            correlate.
        periods (float): The rolling period's length of time.
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

    return (ret_stream_1.rolling(window=periods).corr(ret_stream_2)
        if cum is False else ret_stream_1.expanding().corr(ret_stream_2))


def rolling_r2(ret_stream_1, ret_stream_2, periods, cum=False):
    """Calcs rolling R2 which is just corr^2.

    The rolling R2 is simply the square of the correlation. So this function 
    simply calls the rolling_correlation function.

    Args:
        ret_stream_1 (Pandas DataFrame of float): The first data series to 
            correlate.
        ret_stream_2 (Pandas DataFrame of float): The second data series to 
            correlate.
        periods (float): The rolling period's length of time.
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

    return rolling_corr(ret_stream_1, ret_stream_2, periods, cum) ** 2


def rolling_beta(ret_stream_topic, ret_stream_mkt, periods, cum=False,
                 semibeta=None, semibeta_threshold=None):
    """Calculates rolling beta.

    [longer desc]

    Args:
        ret_stream_topic (Pandas DataFrame of float): The dependent variable 
            series.
        ret_stream_mkt (Pandas DataFrame of float): The independent variable 
            series.
        periods (float): The rolling period's length of time.
        cum (bool, optional): Boolean flag to indicate calculating a cumulative 
            beta or not.
            (default is False)
        semibeta (int, optional): [desc].
            (default is None)
        semibeta_threshold (float, optional): [desc].
            (default is None)
    """

    # round periods to nearest int if it comes in as float
    periods = np.round(periods).astype(int)

    # don't return values if being asked to calculate something shorter than
    # 1 period or not a multiple of the periodicity
    if not valid_period(periods) or (periods < MIN_OBS_THRESH):
        return np.nan

    rs_topic = ret_stream_topic.copy()
    rs_mkt = ret_stream_mkt.copy()

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


def jensens_alpha(topic_ret, bmk_ret, rf_ret, beta):
    """Calculates jensen's alpha.

    Jensen's alpha calculates a risk-adjusted excess return. Many asset 
    managers will present excess return as alpha; however, the could have 
    achieves excess returns by simply running higher risk v. their benchmark. 
    Jense's alpha is effectively the y-intercept if one were to regress manager 
    returns against their benchmark returns.

    Args:
        topic_ret (Pandas DataFrame of float): [desc].
        bmk_ret (Pandas DataFrame of float): [desc].
        rf_ret (Pandas DataFrame of float): [desc].
        beta (Pandas DataFrame of float): [desc].
    
    Returns:
        float: Jensen's alpha.
    """

    return topic_ret - rf_ret - (beta * (bmk_ret - rf_ret))


def m2(topic_ret, topic_vol, sharpe, mkt_vol, cum=False, annual_factor=1):
    """Calcs m2 return which is a port to mkt vol adjusted return measure.
    
    The Sharpe ratio can be difficult to interpres, so M2 takes a sharpe and 
    translates it into a return nummber.

    Args:
        topic_ret (Pandas DataFrame of float): [desc].
        topic_vol (Pandas DataFrame of float): [desc].
        sharpe (Pandas DataFrame of float): [desc].
        mkt_vol (Pandas DataFrame of float): [desc].
        cum (bool, optional): Boolean flag to inidicate calculating a cumulative
            value.
            (default is False)
        annual_factor (float, optional): The factor used to annualize the M2 
            value.
            (default is 1)
    
    Returns:
        Pantas DataFrame of float: M2.
    """

    return (topic_ret + (sharpe * (mkt_vol - topic_vol))) * annual_factor


def m2_excess(ret_m2, ret_bmk):
    """Calcs the m2 excess return v. benchmark.

    Args:
        ret_m2 (Panads DataFrame of float): An m2 return or series of returns.
        ret_bmk (Pandas DataFrame of float): The benchmark return or series of 
            returns.
    
    Returns:
        Pandas DataFrame of float: An excess return series.
    """
    
    return ret_m2 - ret_bmk


def change_analysis(src_df, src_col_topic='topic', src_col_rf=None,
                    src_col_bmk=None, annualize_flag=True, tm_periods=None,
                    measures=None):
    """Generates all of the change analytics data from a Pandas DataFrame.
    """
    
    # create change series data structure
    srs_type_topic = 'topic'
    srs_type_rf = 'rf'
    srs_type_bmk = 'bmk'
    srs_type_excess = 'excess'
    srs_type_active = 'active'
    chg_streams = [
        srs_type_topic,
        srs_type_rf,
        srs_type_bmk,
        srs_type_excess,
        srs_type_active
    ]
    level = 'level'
    level_ln = 'level_ln'
    chg_rel = 'chg_rel'  # the relative change, aka the compound return
    chg_abs = 'chg_abs'  # the absolute change in the level
    chg_ln = 'chg_ln'  # log returns, the absolute change in the log level
    vol_lvl = 'vol_lvl'  # vol of the level of the series
    vol_lvl_ln = 'vol_lvl_ln'  # vol of the log level of the series
    vol_cmp = 'vol_cmp'  # vol of relative changes/compound returns
    vol_ln = 'vol_ln'  # vol of log level changes/log returns
    sharpe_ln = 'sharpe_ln'
    info_ratio_ln = 'info_ratio_ln'
    vol_downside_ln = 'vol_downside_ln'
    vol_upside_ln = 'vol_upside_ln'
    sortino_ln = 'sortino_ln'
    vol_skew_ln = 'vol_skew_ln'
    mod_treynor_ln = 'mod_treynor_ln'
    corr_lvl = 'corr_lvl'
    corr_lvl_ln = 'corr_lvl_ln'
    corr_ln = 'corr_ln'
    r2_ln = 'r2_ln'
    beta_ln = 'beta_ln'
    beta_up_ln = 'beta_up_ln'
    beta_down_ln = 'beta_down_ln'
    jensens_alpha_ln = 'jensens_alpha_ln'
    m2_ln = 'm2_ln'
    m2_excess_ln = 'm2_excess_ln'
    dd_series_cmp = 'dd_series_cmp'
    dd_count_cmp = 'dd_count_cmp'
    dd_avg_size_cmp = 'dd_avg_size_cmp'
    dd_avg_periods_cmp = 'dd_avg_periods_cmp'
    dd_med_size_cmp = 'dd_med_size_cmp'
    dd_med_periods_cmp = 'dd_med_periods_cmp'
    dd_max_size_cmp = 'dd_max_size_cmp'
    dd_max_size_periods_cmp = 'dd_max_size_periods_cmp'
    dd_max_size_start_cmp = 'dd_max_size_start_cmp'
    dd_max_size_end_cmp = 'dd_max_size_end_cmp'
    dd_max_length_size_cmp = 'dd_max_legnth_size_ln'
    dd_max_length_cmp = 'dd_max_length_cmp'
    dd_max_length_start_cmp = 'dd_max_length_start_cmp'
    dd_max_length_end_cmp = 'dd_max_length_end_cmp'
    dd_curr_series_cmp = 'dd_curr_series_cmp'
    dd_curr_size_cmp = 'dd_curr_size_cmp'
    dd_curr_max_size_cmp = 'dd_curr_max_size_cmp'
    dd_curr_length_cmp = 'dd_curr_length_cmp'
    dd_curr_start_cmp = 'dd_curr_start_cmp'
    dd_cmp = [
        dd_series_cmp,
        dd_count_cmp,
        dd_avg_size_cmp,
        dd_avg_periods_cmp,
        dd_med_size_cmp,
        dd_med_periods_cmp,
        dd_max_size_cmp,
        dd_max_size_periods_cmp,
        dd_max_size_start_cmp,
        dd_max_size_end_cmp,
        dd_max_length_size_cmp,
        dd_max_length_cmp,
        dd_max_length_start_cmp,
        dd_max_length_end_cmp,
        dd_curr_series_cmp,
        dd_curr_size_cmp,
        dd_curr_max_size_cmp,
        dd_curr_length_cmp,
        dd_curr_start_cmp
    ]

    # create time periods windows
    the_periodicity = get_periodicity(src_df.index)
    periodicity = the_periodicity.ann_factor
    create_tm_periods(src_df.index.size, the_periodicity)

    # spot time period is the lowest level return period possible based
    # on periodicity of the input time series
    tm_period_spot = ['spot', 1 / periodicity]

    tm_period_1d = ['1d', 1 / PDCY_DAILY_CALENDAR.ann_factor]
    if periodicity == PDCY_DAILY_TRADING.ann_factor:
        tm_period_1d[1] = 1 / periodicity
    tm_period_1w = ['1w', 1 / 52]
    tm_period_2w = ['2w', 2 / 52]
    tm_period_1m = ['1m', 1 / 12]  # if daily -> 30 calendar, 21 trading
    tm_period_6w = ['6w', 6 / 52]
    tm_period_2m = ['2m', 2 / 12]  # if daily -> 61 calendar, 42 trading
    tm_period_3m = ['3m', 3 / 12]  # if daily -> 91 calendar, 63 trading
    tm_period_6m = ['6m', 6 / 12]  # if daily -> 183 calendar, 126 trading
    tm_period_1y = ['1y', 1.]
    tm_period_2y = ['2y', 2.]
    tm_period_3y = ['3y', 3.]
    tm_period_5y = ['5y', 5.]
    tm_period_10y = ['10y', 10.]
    # tm_period_wtd = ['wtd', 1 / 52]
    # tm_period_mtd = ['mtd', 1 / 12]
    # tm_period_qtd = ['qtd', 3 / 12]
    # tm_period_ytd = ['ytd', 1.]
    tm_period_cum = ['cum', 1.]
    all_tm_periods = [
        tm_period_spot, tm_period_1d, tm_period_1w,
        tm_period_2w, tm_period_1m, tm_period_6w,
        tm_period_2m, tm_period_3m, tm_period_6m,
        tm_period_1y, tm_period_2y, tm_period_3y,
        tm_period_5y, tm_period_10y,  # tm_period_wtd,
        # tm_period_mtd, tm_period_qtd, tm_period_ytd,
        tm_period_cum
    ]
    tm_periods = {}
    for period in all_tm_periods:
        # don't include rolling periods longer than input series
        # or short intertemporal periods for monthly series
        if ((period[1] * periodicity < src_df.index.size and
             not (1 < period[1] * periodicity < 2)) or
                (period[0] == 'cum')):
            if period[1] * periodicity >= 1:
                # flag periods that require a min observation threshold
                # for certain statistics
                if period[1] * periodicity >= MIN_OBS_THRESH:
                    stat_flag = True
                else:
                    stat_flag = False
                tm_periods[period[0]] = [period[1], stat_flag]

    src_chg_streams = {
        srs_type_topic: src_col_topic,
        srs_type_rf: src_col_rf,
        srs_type_bmk: src_col_bmk
    }

    # create change/return time period columns
    cols = []

    for key in src_chg_streams:
        cols.append([key, level, tm_period_spot[0]])
        cols.append([key, level_ln, tm_period_spot[0]])
        cols.append([key, chg_rel, tm_period_spot[0]])
        for key2 in tm_periods.keys():
            cols.append([key, chg_ln, key2])
            cols.append([key, chg_abs, key2])
            cols.append([key, vol_lvl, key2])
            cols.append([key, vol_lvl_ln, key2])
    for key in [srs_type_excess, srs_type_active]:
        cols.append([key, chg_rel, tm_period_spot[0]])
        for key2 in tm_periods.keys():
            cols.append([key, chg_ln, key2])

    # create vol time period columns
    for key in chg_streams:
        for key2 in tm_periods.keys():
            if tm_periods[key2][1] is True:
                cols.append([key, vol_cmp, key2])
                cols.append([key, vol_ln, key2])

    # create time period columns for other measures
    for key in tm_periods.keys():
        if tm_periods[key][1] is True:
            cols.append([srs_type_topic, sharpe_ln, key])
            cols.append([srs_type_active, info_ratio_ln, key])
            cols.append([srs_type_excess, vol_downside_ln, key])
            cols.append([srs_type_excess, vol_upside_ln, key])
            cols.append([srs_type_active, vol_downside_ln, key])
            cols.append([srs_type_active, vol_upside_ln, key])
            cols.append([srs_type_excess, sortino_ln, key])
            cols.append([srs_type_active, sortino_ln, key])
            cols.append([srs_type_excess, vol_skew_ln, key])
            cols.append([srs_type_active, vol_skew_ln, key])
            cols.append([srs_type_excess, mod_treynor_ln, key])
            cols.append([srs_type_active, corr_lvl, key])
            cols.append([srs_type_active, corr_lvl_ln, key])
            cols.append([srs_type_active, corr_ln, key])
            cols.append([srs_type_active, r2_ln, key])
            cols.append([srs_type_active, beta_ln, key])
            cols.append([srs_type_active, beta_up_ln, key])
            cols.append([srs_type_active, beta_down_ln, key])
            cols.append([srs_type_active, jensens_alpha_ln, key])
            cols.append([srs_type_active, m2_ln, key])
            cols.append([srs_type_active, m2_excess_ln, key])
            for key2 in dd_cmp:
                cols.append([srs_type_topic, key2, key])

    df = pd.DataFrame(index=src_df.index,
                      columns=pd.MultiIndex.from_tuples(
                        cols, names=['srs_type', 'measure', 'period']))

    # calc spot changes
    for key in src_chg_streams:
        df.loc[:, (key, level, tm_period_spot[0])] = \
            src_df[src_chg_streams[key]]
        df.loc[:, (key, level_ln, tm_period_spot[0])] = \
            np.log(df[key][level][tm_period_spot[0]])
        df.loc[:, (key, chg_ln, tm_period_spot[0])] = np.log(
                1 + df[key][level][tm_period_spot[0]].pct_change(1))
        df.loc[:, (key, chg_rel, tm_period_spot[0])] = np.exp(
                df[key][chg_ln][tm_period_spot[0]]) - 1
        df.loc[:, (key, chg_abs, tm_period_spot[0])] = \
            df[key][level][tm_period_spot[0]].diff(1)

    # calc excess returns
    df.loc[:, (srs_type_excess, chg_ln, tm_period_spot[0])] = excess_return(
            df[srs_type_topic][chg_ln][tm_period_spot[0]],
            df[srs_type_rf][chg_ln][tm_period_spot[0]])
    df.loc[:, (srs_type_excess, chg_rel, tm_period_spot[0])] = np.exp(
            df[srs_type_excess][chg_ln][tm_period_spot[0]]) - 1

    # calc active returns
    df.loc[:, (srs_type_active, chg_ln, tm_period_spot[0])] = excess_return(
            df[srs_type_topic][chg_ln][tm_period_spot[0]],
            df[srs_type_bmk][chg_ln][tm_period_spot[0]])
    df.loc[:, (srs_type_active, chg_rel, tm_period_spot[0])] = np.exp(
            df[srs_type_active][chg_ln][tm_period_spot[0]]) - 1

    for ret in chg_streams:
        for key in tm_periods.keys():
            # only calc periodic returns if it's not the spot return
            if key != tm_period_spot[0]:
                cum_sum = (key == tm_period_cum[0])

                df.loc[:, (ret, chg_ln, key)] = rolling_chg(
                    df[ret][chg_ln][tm_period_spot[0]],
                    tm_periods[key][0] * periodicity,
                    periodicity,
                    annualize_flag,
                    cum_sum)
                # only calc the abs chg & vols for topic, bmk, and rfr streams
                if ret in src_chg_streams:
                    df.loc[:, (ret, chg_abs, key)] = rolling_chg(
                        df[ret][chg_abs][tm_period_spot[0]],
                        tm_periods[key][0] * periodicity,
                        periodicity,
                        False,
                        cum_sum)
                    df.loc[:, (ret, vol_lvl, key)] = rolling_vol(
                        df[ret][level][tm_period_spot[0]],
                        tm_periods[key][0] * periodicity,
                        periodicity,
                        annualize_flag,
                        cum_sum)
                    df.loc[:, (ret, vol_lvl_ln, key)] = rolling_vol(
                        df[ret][level_ln][tm_period_spot[0]],
                        tm_periods[key][0] * periodicity,
                        periodicity,
                        annualize_flag,
                        cum_sum)

    # calc vols for compound + log returns
    for ret in chg_streams:
        for key in tm_periods.keys():
            if tm_periods[key][1] is True:
                if key == tm_period_cum[0]:
                    cum = True
                else:
                    cum = False
                df.loc[:, (ret, vol_ln, key)] = rolling_vol(
                    df[ret][chg_ln][tm_period_spot[0]],
                    tm_periods[key][0] * periodicity,
                    periodicity,
                    annualize_flag,
                    cum)
                df.loc[:, (ret, vol_cmp, key)] = rolling_vol(
                    df[ret][chg_rel][tm_period_spot[0]],
                    tm_periods[key][0] * periodicity,
                    periodicity,
                    annualize_flag,
                    cum)

    for key in tm_periods.keys():
        if tm_periods[key][1] is True:
            if key == tm_period_cum[0]:
                annual_factor = np.maximum(
                    df[srs_type_topic][chg_ln][key].expanding().count(),
                    periodicity) / periodicity
                cum = True
            else:
                if annualize_flag is True:
                    annual_factor = 1
                else:
                    annual_factor = tm_periods[key][0]
                cum = False

            # calc sharpe ratio
            df.loc[:, (srs_type_topic, sharpe_ln, key)] = sharpe(
                df[srs_type_topic][chg_ln][key] / annual_factor,
                df[srs_type_rf][chg_ln][key] / annual_factor,
                df[srs_type_topic][vol_ln][key])

            # calc info ratio
            df.loc[:, (srs_type_active, info_ratio_ln, key)] = info_ratio(
                df[srs_type_topic][chg_ln][key] / annual_factor,
                df[srs_type_bmk][chg_ln][key] / annual_factor,
                df[srs_type_active][vol_ln][key])

            # calc excess up/down vols
            df.loc[:, (srs_type_excess, vol_downside_ln, key)] = rolling_vol(
                df[srs_type_excess][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                periodicity, annualize_flag, cum, -1, 0)
            df.loc[:, (srs_type_excess, vol_upside_ln, key)] = rolling_vol(
                df[srs_type_excess][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                periodicity, annualize_flag, cum, 1, 0)

            # calc acive up/down vols
            df.loc[:, (srs_type_active, vol_downside_ln, key)] = rolling_vol(
                df[srs_type_active][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                periodicity, annualize_flag, cum, -1, 0)
            df.loc[:, (srs_type_active, vol_upside_ln, key)] = rolling_vol(
                df[srs_type_active][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                periodicity, annualize_flag, cum, 1, 0)

            # calc sortino ratios
            df.loc[:, (srs_type_excess, sortino_ln, key)] = sortino(
                df[srs_type_topic][chg_ln][key] / annual_factor,
                df[srs_type_rf][chg_ln][key] / annual_factor,
                df[srs_type_excess][vol_downside_ln][key])
            df.loc[:, (srs_type_active, sortino_ln, key)] = sortino(
                df[srs_type_topic][chg_ln][key] / annual_factor,
                df[srs_type_bmk][chg_ln][key] / annual_factor,
                df[srs_type_active][vol_downside_ln][key])

            # calc vol skew ratios
            df.loc[:, (srs_type_excess, vol_skew_ln, key)] = vol_skew(
                df[srs_type_excess][vol_upside_ln][key],
                df[srs_type_excess][vol_downside_ln][key])
            df.loc[:, (srs_type_active, vol_skew_ln, key)] = vol_skew(
                df[srs_type_active][vol_upside_ln][key],
                df[srs_type_active][vol_downside_ln][key])

            # calc modified treynor ratio
            df.loc[:, (srs_type_excess, mod_treynor_ln, key)] = mod_treynor(
                df[srs_type_topic][chg_ln][key] / annual_factor,
                df[srs_type_rf][chg_ln][key] / annual_factor,
                df[srs_type_bmk][vol_ln][key])

            # calc correlations & r2
            df.loc[:, (srs_type_active, corr_lvl, key)] = rolling_corr(
                df[srs_type_topic][level][tm_period_spot[0]],
                df[srs_type_bmk][level][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum)
            df.loc[:, (srs_type_active, corr_lvl_ln, key)] = rolling_corr(
                df[srs_type_topic][level_ln][tm_period_spot[0]],
                df[srs_type_bmk][level_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum)
            df.loc[:, (srs_type_active, corr_ln, key)] = rolling_corr(
                df[srs_type_topic][chg_ln][tm_period_spot[0]],
                df[srs_type_bmk][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum)
            df.loc[:, (srs_type_active, r2_ln, key)] = \
                df[srs_type_active][corr_ln][key] ** 2  # don't bother using r2

            # calc betas to benchmark
            df.loc[:, (srs_type_active, beta_ln, key)] = rolling_beta(
                df[srs_type_topic][chg_ln][tm_period_spot[0]],
                df[srs_type_bmk][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum)
            df.loc[:, (srs_type_active, beta_up_ln, key)] = rolling_beta(
                df[srs_type_topic][chg_ln][tm_period_spot[0]],
                df[srs_type_bmk][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum, 1, 0)
            df.loc[:, (srs_type_active, beta_down_ln, key)] = rolling_beta(
                df[srs_type_topic][chg_ln][tm_period_spot[0]],
                df[srs_type_bmk][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum, -1, 0)

            # calc jensen's alpha -- real alpha, not just active return b.s.
            df.loc[:, (srs_type_active, jensens_alpha_ln, key)] = \
                jensens_alpha(df[srs_type_topic][chg_ln][key] / annual_factor,
                              df[srs_type_bmk][chg_ln][key] / annual_factor,
                              df[srs_type_rf][chg_ln][key] / annual_factor,
                              df[srs_type_active][beta_ln][key])

            # calc m2 return and excess return
            df.loc[:, (srs_type_active, m2_ln, key)] = m2(
                df[srs_type_topic][chg_ln][key] / annual_factor,
                df[srs_type_topic][vol_ln][key],
                df[srs_type_topic][sharpe_ln][key],
                df[srs_type_bmk][vol_ln][key], cum, annual_factor)
            df.loc[:, (srs_type_active, m2_excess_ln, key)] = m2_excess(
                df[srs_type_active][m2_ln][key], df[srs_type_bmk][chg_ln][key])

            """
            # calculate rolling drawdowns. algo needs to be rewritten -- thing
            # is insanely slow
            dd_df = rolling_dd_series(
                df[srs_type_topic][chg_ln][tm_period_spot[0]],
                tm_periods[key][0] * periodicity,
                cum)
            if not dd_df.empty:
                df.loc[:, (srs_type_topic, dd_series_cmp, key)] = np.nan
                df.loc[:, (srs_type_topic, dd_count_cmp, key)] = dd_df['count']
                df.loc[:, (srs_type_topic, dd_avg_size_cmp, key)] = \
                    dd_df['size_avg']
                df.loc[:, (srs_type_topic, dd_avg_periods_cmp, key)] = \
                    dd_df['length_avg']
                df.loc[:, (srs_type_topic, dd_med_size_cmp, key)] = \
                    dd_df['size_med']
                df.loc[:, (srs_type_topic, dd_med_periods_cmp, key)] = \
                    dd_df['length_med']
                df.loc[:, (srs_type_topic, dd_max_size_cmp, key)] = \
                    dd_df['max_size']
                df.loc[:, (srs_type_topic, dd_max_size_periods_cmp, key)] = \
                    dd_df['max_size_length']
                df.loc[:, (srs_type_topic, dd_max_size_start_cmp, key)] = \
                    dd_df['max_size_start']
                df.loc[:, (srs_type_topic, dd_max_size_end_cmp, key)] = \
                    dd_df['max_size_end']
                df.loc[:, (srs_type_topic, dd_max_length_size_cmp, key)] = \
                    dd_df['max_length_size']
                df.loc[:, (srs_type_topic, dd_max_length_cmp, key)] = \
                    dd_df['max_length']
                df.loc[:, (srs_type_topic, dd_max_length_start_cmp, key)] = \
                    dd_df['max_length_start']
                df.loc[:, (srs_type_topic, dd_max_length_end_cmp, key)] = \
                    dd_df['max_length_end']
                df.loc[:, (srs_type_topic, dd_curr_series_cmp, key)] = np.nan
                df.loc[:, (srs_type_topic, dd_curr_size_cmp, key)] = \
                    dd_df['curr_size']
                df.loc[:, (srs_type_topic, dd_curr_max_size_cmp, key)] = \
                    dd_df['curr_max_size']
                df.loc[:, (srs_type_topic, dd_curr_length_cmp, key)] = \
                    dd_df['curr_length']
                df.loc[:, (srs_type_topic, dd_curr_start_cmp, key)] = \
                    dd_df['curr_start']
            """

    return df

def main(src_file='infile.xlsx', src_sheet='Sheet1', src_col_dt='date',
         src_col_topic='strategy', src_col_rfr='rfr', src_col_bmk='benchmark',
         tgt_file='outfile.xlsx', tgt_sheet='Sheet1'):
    
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
    main()