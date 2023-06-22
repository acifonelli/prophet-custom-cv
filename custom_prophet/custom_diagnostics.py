import pandas as pd

from prophet.diagnostics import concurrent, logger, single_cutoff_forecast, tqdm
from typing import Union



def cross_validation(
    model,
    horizon: int,
    freq ='d',
    period: Union[int, None] =None,
    initial: Union[pd.DateOffset, pd.Timestamp, None]=None,
    parallel=None,
    cutoffs=None,
    disable_tqdm=False):
    """Cross-Validation for time series.

    Adaptation of original 'prophet.diagnotics.cross_validation()'.
    
    Contrary to the repo version, this function preserves the 'model.history' 
    frequency while generating the cross validation Dataframe.
    
    Check the original version for further details.

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model.
    horizon: Expressed as number of time points.
    period: Expressed as number of 'jumps'. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will include at least this much data. If not provided,
        3 * horizon is used.
    cutoffs: list of pd.Timestamp specifying cutoffs to be used during
        cross validation. If not provided, they are generated automatically.
    parallel : {None, 'processes', 'threads', 'dask', object}

        Original implementation legacy.
    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None

        Original implementation legacy. For more information on parallelism, check the original function.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    
    if model.history is None:
        raise Exception('Model has not been fit. Fitting the model provides contextual parameters for cross validation.')
    
    df = model.history.copy().reset_index(drop=True)

    predict_columns = ['ds', 'yhat']
    if model.uncertainty_samples:
        predict_columns.extend(['yhat_lower', 'yhat_upper'])
        
    # Identify largest seasonality period
    period_max = 0.
    for s in model.seasonalities.values():
        period_max = max(period_max, s['period'])
    seasonality_dt = pd.Timedelta(str(period_max) + ' days')
    min_date = df['ds'].min()
    max_date = df['ds'].max()
    date_index = pd.date_range(
        start = min_date,
        end = max_date,
        freq = pd.infer_freq(df['ds'])
    )
    df_freq = date_index.freq
    initial_td = max(3 * horizon, period_max)
    end_date_minus_horizon = max_date - horizon * df_freq
    horizon_td = pd.Timedelta(max_date - end_date_minus_horizon)

    if cutoffs is None:
        # Set period
        _period = 0.5 * horizon if period is None else period

        # Set initial
        _initial = min_date + initial_td * df_freq
        if initial is not None:
          init_type = type(initial)
          if init_type is pd.DateOffset:
            _initial = min_date + initial
          elif init_type is pd.Timestamp:
            _initial = initial

        # Compute Cutoffs
        date_index = pd.date_range(
            start = _initial,
            end = max_date - horizon * df_freq,
            freq = "%d%s" % (_period, freq) if type(freq) is str else _period * freq
        )
        cutoffs = list(filter(lambda e: e >= _initial, date_index))
        if len(cutoffs) == 0:
            raise ValueError(
                'Less data than horizon after initial window. '
                'Make horizon or initial shorter.'
            )
        initial_td = cutoffs[0] - min_date
        logger.info('Making {} forecasts with cutoffs between {} and {}'.format(
            len(cutoffs), cutoffs[0], cutoffs[-1]
        ))
    else:
        # add validation of the cutoff to make sure that the min cutoff is strictly greater than the min date in the history
        if min(cutoffs) <= min_date: 
            raise ValueError("Minimum cutoff value is not strictly greater than min date in history")
        # max value of cutoffs is <= (end date minus horizon)
        if max(cutoffs) > end_date_minus_horizon: 
            raise ValueError("Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining")
        initial_td = cutoffs[0] - min_date
        
    # Check if the initial window 
    # (that is, the amount of time between the start of the history and the first cutoff)
    # is less than the maximum seasonality period
    if initial_td < seasonality_dt:
            msg = 'Seasonality has period of {} days '.format(period_max)
            msg += 'which is larger than initial window. '
            msg += 'Consider increasing initial.'
            logger.warning(msg)

    if parallel:
        valid = {"threads", "processes", "dask"}

        if parallel == "threads":
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == "processes":
            pool = concurrent.futures.ProcessPoolExecutor()
        elif parallel == "dask":
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requires the optional "
                                  "dependency dask.") from e
            pool = get_client()
            # delay df and model to avoid large objects in task graph.
            df, model = pool.scatter([df, model])
        elif hasattr(parallel, "map"):
            pool = parallel
        else:
            msg = ("'parallel' should be one of {} for an instance with a "
                   "'map' method".format(', '.join(valid)))
            raise ValueError(msg)

        iterables = ((df, model, cutoff, horizon_td, predict_columns)
                     for cutoff in cutoffs)
        iterables = zip(*iterables)

        logger.info("Applying in parallel with %s", pool)
        predicts = pool.map(single_cutoff_forecast, *iterables)
        if parallel == "dask":
            # convert Futures to DataFrames
            predicts = pool.gather(predicts)

    else:
        predicts = [
            single_cutoff_forecast(df, model, cutoff, horizon_td, predict_columns) 
            for cutoff in (tqdm(cutoffs) if not disable_tqdm else cutoffs)
        ]

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return pd.concat(predicts, axis=0).reset_index(drop=True)