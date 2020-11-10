
from tqdm import tqdm
from collections import namedtuple
from pymc3.model import modelcontext
from scipy.special import logsumexp
import numpy as np
import warnings
import pymc3 as pm
from pymc3.backends import tracetab as ttab
import pandas as pd

def _log_post_trace(trace, model=None, progressbar=False):
    """Calculate the elementwise log-posterior for the sampled trace.

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    logp : array of shape (n_samples, n_observations)
        The contribution of the observations to the logp of the whole model.
    """
    model = modelcontext(model)
    cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

    def logp_vals_point(pt):
        if len(model.observed_RVs) == 0:
            return floatX(np.array([], dtype='d'))

        logp_vals = []
        for var, logp in cached:
            logp = logp(pt)
            if var.missing_values:
                logp = logp[~var.observations.mask]
            logp_vals.append(logp.ravel())

        return np.concatenate(logp_vals)

    try:
        points = trace.points()
    except AttributeError:
        points = trace

    points = tqdm(points) if progressbar else points

    try:
        logp = (logp_vals_point(pt) for pt in points)
        return np.stack(logp)
    finally:
        if progressbar:
            points.close()



WAIC_r_pointwise = namedtuple('WAIC_r_pointwise', 'WAIC, WAIC_se, p_WAIC, var_warn, WAIC_i,lppd_i,vars_lpd')
WAIC_r = namedtuple('WAIC_r', 'WAIC, WAIC_se, p_WAIC, var_warn')
def waic(trace, model=None, pointwise=False, progressbar=False):
    """Calculate the widely available information criterion, its standard error
    and the effective number of parameters of the samples in trace from model.
    Read more theory here - in a paper by some of the leading authorities on
    model selection - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    namedtuple with the following elements:
    waic: widely available information criterion
    waic_se: standard error of waic
    p_waic: effective number parameters
    var_warn: 1 if posterior variance of the log predictive
         densities exceeds 0.4
    waic_i: and array of the pointwise predictive accuracy, only if pointwise True
    """
    model = modelcontext(model)

    log_py = _log_post_trace(trace, model, progressbar=progressbar)
    if log_py.size == 0:
        raise ValueError('The model does not contain observed values.')

    lppd_i = logsumexp(log_py, axis=0, b=1.0 / log_py.shape[0])

    vars_lpd = np.var(log_py, axis=0)
    warn_mg = 0
    if np.any(vars_lpd > 0.4):
        warnings.warn("""For one or more samples the posterior variance of the
        log predictive densities exceeds 0.4. This could be indication of
        WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
        """)
        warn_mg = 1

    waic_i = - 2 * (lppd_i - vars_lpd)

    waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

    waic = np.sum(waic_i)

    p_waic = np.sum(vars_lpd)

    if pointwise:
        if np.equal(waic, waic_i).all():
            warnings.warn("""The point-wise WAIC is the same with the sum WAIC,
            please double check the Observed RV in your model to make sure it
            returns element-wise logp.
            """)
        return WAIC_r_pointwise(waic, waic_se, p_waic, warn_mg, waic_i,lppd_i,vars_lpd)
    else:
        return WAIC_r(waic, waic_se, p_waic, warn_mg)


LOO_r_pointwise = namedtuple('LOO_r_pointwise', 'LOO, LOO_se, p_LOO, shape_warn, LOO_i,ks')
LOO_r = namedtuple('LOO_r', 'LOO, LOO_se, p_LOO, shape_warn')
def loo(trace, model=None, pointwise=False, reff=None, progressbar=False):
    """Calculates leave-one-out (LOO) cross-validation for out of sample
    predictive model fit, following Vehtari et al. (2015). Cross-validation is
    computed using Pareto-smoothed importance sampling (PSIS).

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False
    reff : float
        relative MCMC efficiency, `effective_n / n` i.e. number of effective
        samples divided by the number of actual samples. Computed from trace by
        default.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    namedtuple with the following elements:
    loo: approximated Leave-one-out cross-validation
    loo_se: standard error of loo
    p_loo: effective number of parameters
    shape_warn: 1 if the estimated shape parameter of
        Pareto distribution is greater than 0.7 for one or more samples
    loo_i: array of pointwise predictive accuracy, only if pointwise True
    """
    model = modelcontext(model)

    if reff is None:
        if trace.nchains == 1:
            reff = 1.
        else:
            eff = pm.effective_n(trace)
            eff_ave = pm.stats.dict2pd(eff, 'eff').mean()
            samples = len(trace) * trace.nchains
            reff = eff_ave / samples

    log_py = _log_post_trace(trace, model, progressbar=progressbar)
    if log_py.size == 0:
        raise ValueError('The model does not contain observed values.')

    lw, ks = _psislw(-log_py, reff)
    lw += log_py

    warn_mg = 0
    if np.any(ks > 0.7):
        warnings.warn("""Estimated shape parameter of Pareto distribution is
        greater than 0.7 for one or more samples.
        You should consider using a more robust model, this is because
        importance sampling is less likely to work well if the marginal
        posterior and LOO posterior are very different. This is more likely to
        happen with a non-robust model and highly influential observations.""")
        warn_mg = 1

    loo_lppd_i = - 2 * logsumexp(lw, axis=0)
    loo_lppd = loo_lppd_i.sum()
    loo_lppd_se = (len(loo_lppd_i) * np.var(loo_lppd_i)) ** 0.5
    lppd = np.sum(logsumexp(log_py, axis=0, b=1. / log_py.shape[0]))
    p_loo = lppd + (0.5 * loo_lppd)

    if pointwise:
        if np.equal(loo_lppd, loo_lppd_i).all():
            warnings.warn("""The point-wise LOO is the same with the sum LOO,
            please double check the Observed RV in your model to make sure it
            returns element-wise logp.
            """)
        return LOO_r_pointwise(loo_lppd, loo_lppd_se, p_loo, warn_mg, loo_lppd_i,ks)
    else:
        return LOO_r(loo_lppd, loo_lppd_se, p_loo, warn_mg)


def _psislw(lw, reff):
    """Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    lw : array
        Array of size (n_samples, n_observations)
    reff : float
        relative MCMC efficiency, `effective_n / n`

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : array
        Pareto tail indices
    """
    n, m = lw.shape

    lw_out = np.copy(lw, order='F')
    kss = np.empty(m)

    # precalculate constants
    cutoff_ind = - int(np.ceil(min(n / 5., 3 * (n / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)
    k_min = 1. / 3

    # loop over sets of log weights
    for i, x in enumerate(lw_out.T):
        # improve numerical accuracy
        x -= np.max(x)
        # sort the array
        x_sort_ind = np.argsort(x)
        # divide log weights into body and right tail
        xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

        expxcutoff = np.exp(xcutoff)
        tailinds, = np.where(x > xcutoff)
        x2 = x[tailinds]
        n2 = len(x2)
        if n2 <= 4:
            # not enough tail samples for gpdfit
            k = np.inf
        else:
            # order of tail samples
            x2si = np.argsort(x2)
            # fit generalized Pareto distribution to the right tail samples
            x2 = np.exp(x2) - expxcutoff
            k, sigma = _gpdfit(x2[x2si])

        if k >= k_min and not np.isinf(k):
            # no smoothing if short tail or GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, n2) / n2
            qq = _gpinv(sti, k, sigma)
            qq = np.log(qq + expxcutoff)
            # place the smoothed tail into the output array
            x[tailinds[x2si]] = qq
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
        # renormalize weights
        x -= logsumexp(x)
        # store tail index k
        kss[i] = k

    return lw_out, kss


def _gpdfit(x):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD)

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    x : array
        sorted 1D data array

    Returns
    -------
    k : float
        estimated shape parameter
    sigma : float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(x)
    m = 30 + int(n**0.5)

    bs = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    bs /= prior_bs * x[int(n/4 + 0.5) - 1]
    bs += 1 / x[-1]

    ks = np.log1p(-bs[:, None] * x).mean(axis=1)
    L = n * (np.log(-(bs / ks)) - ks - 1)
    w = 1 / np.exp(L - L[:, None]).sum(axis=1)

    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()

    # posterior mean for b
    b = np.sum(bs * w)
    # estimate for k
    k = np.log1p(- b * x).mean()
    # add prior for k
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    sigma = - k / b

    return k, sigma


def _gpinv(p, k, sigma):
    """Inverse Generalized Pareto distribution function"""
    x = np.full_like(p, np.nan)
    if sigma <= 0:
        return x
    ok = (p > 0) & (p < 1)
    if np.all(ok):
        if np.abs(k) < np.finfo(float).eps:
            x = - np.log1p(-p)
        else:
            x = np.expm1(-k * np.log1p(-p)) / k
        x *= sigma
    else:
        if np.abs(k) < np.finfo(float).eps:
            x[ok] = - np.log1p(-p[ok])
        else:
            x[ok] = np.expm1(-k * np.log1p(-p[ok])) / k
        x *= sigma
        x[p == 0] = 0
        if k >= 0:
            x[p == 1] = np.inf
        else:
            x[p == 1] = - sigma / k

    return x


def summary(trace, varnames=None, transform=lambda x: x, stat_funcs=None,
               extend=False, include_transformed=False,
               alpha=0.05, start=0, batches=None):
    R"""Create a data frame with summary statistics.

    Parameters
    ----------
    trace : MultiTrace instance
    varnames : list
        Names of variables to include in summary
    transform : callable
        Function to transform data (defaults to identity)
    stat_funcs : None or list
        A list of functions used to calculate statistics. By default,
        the mean, standard deviation, simulation standard error, and
        highest posterior density intervals are included.

        The functions will be given one argument, the samples for a
        variable as a 2 dimensional array, where the first axis
        corresponds to sampling iterations and the second axis
        represents the flattened variable (e.g., x__0, x__1,...). Each
        function should return either

        1) A `pandas.Series` instance containing the result of
           calculating the statistic along the first axis. The name
           attribute will be taken as the name of the statistic.
        2) A `pandas.DataFrame` where each column contains the
           result of calculating the statistic along the first axis.
           The column names will be taken as the names of the
           statistics.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in
        addition to, rather than in place of, the default statistics.
        This is only meaningful when `stat_funcs` is not None.
    include_transformed : bool
        Flag for reporting automatically transformed variables in addition
        to original variables (defaults to False).
    alpha : float
        The alpha level for generating posterior intervals. Defaults
        to 0.05. This is only meaningful when `stat_funcs` is None.
    start : int
        The starting index from which to summarize (each) chain. Defaults
        to zero.
    batches : None or int
        Batch size for calculating standard deviation for non-independent
        samples. Defaults to the smaller of 100 or the number of samples.
        This is only meaningful when `stat_funcs` is None.

    Returns
    -------
    `pandas.DataFrame` with summary statistics for each variable Defaults one
    are: `mean`, `sd`, `mc_error`, `hpd_2.5`, `hpd_97.5`, `n_eff` and `Rhat`.
    Last two are only computed for traces with 2 or more chains.

    Examples
    --------
    .. code:: ipython

        >>> import pymc3 as pm
        >>> trace.mu.shape
        (1000, 2)
        >>> pm.summary(trace, ['mu'])
                   mean        sd  mc_error     hpd_5    hpd_95
        mu__0  0.106897  0.066473  0.001818 -0.020612  0.231626
        mu__1 -0.046597  0.067513  0.002048 -0.174753  0.081924

                  n_eff      Rhat
        mu__0     487.0   1.00001
        mu__1     379.0   1.00203

    Other statistics can be calculated by passing a list of functions.

    .. code:: ipython

        >>> import pandas as pd
        >>> def trace_sd(x):
        ...     return pd.Series(np.std(x, 0), name='sd')
        ...
        >>> def trace_quantiles(x):
        ...     return pd.DataFrame(pm.quantiles(x, [5, 50, 95]))
        ...
        >>> pm.summary(trace, ['mu'], stat_funcs=[trace_sd, trace_quantiles])
                     sd         5        50        95
        mu__0  0.066473  0.000312  0.105039  0.214242
        mu__1  0.067513 -0.159097 -0.045637  0.062912
    """

    if varnames is None:
        varnames = get_default_varnames(trace.varnames,
                       include_transformed=include_transformed)

    if batches is None:
        batches = min([100, len(trace)])

    funcs = [lambda x: pd.Series(np.mean(x, 0), name='mean'),
             lambda x: pd.Series(np.std(x, 0), name='sd'),
             lambda x: pd.Series(mc_error(x, batches), name='mc_error'),
             lambda x: _hpd_df(x, alpha)]

    if stat_funcs is not None:
        if extend:
            funcs = funcs + stat_funcs
        else:
            funcs = stat_funcs

    var_dfs = []
    for var in varnames:
        vals = transform(trace.get_values(var, burn=start, combine=True))
        flat_vals = vals.reshape(vals.shape[0], -1)
        var_df = pd.concat([f(flat_vals) for f in funcs], axis=1)
        var_df.index = ttab.create_flat_names(var, vals.shape[1:])
        var_dfs.append(var_df)
    dforg = pd.concat(var_dfs, axis=0)

    if (stat_funcs is not None) and (not extend):
        return dforg
    elif trace.nchains < 2:
        return dforg
    else:
        n_eff = pm.effective_n(trace,
                               varnames=varnames,
                               include_transformed=include_transformed)
        n_eff_pd = dict2pd(n_eff, 'n_eff')
        rhat = pm.gelman_rubin(trace,
                               varnames=varnames,
                               include_transformed=include_transformed)
        rhat_pd = dict2pd(rhat, 'Rhat')
        #import pdb; pdb.set_trace()
        # return pd.concat([dforg, n_eff_pd, rhat_pd],
        #                  axis=1, join_axes=[dforg.index])
        return pd.concat([dforg, n_eff_pd, rhat_pd],axis=1).reindex(dforg.index)


def statfunc(f):
    """
    Decorator for statistical utility function to automatically
    extract the trace array from whatever object is passed.
    """

    def wrapped_f(pymc3_obj, *args, **kwargs):
        try:
            vars = kwargs.pop('vars',  pymc3_obj.varnames)
            chains = kwargs.pop('chains', pymc3_obj.chains)
        except AttributeError:
            # If fails, assume that raw data was passed.
            return f(pymc3_obj, *args, **kwargs)

        burn = kwargs.pop('burn', 0)
        thin = kwargs.pop('thin', 1)
        combine = kwargs.pop('combine', False)
        # Remove outer level chain keys if only one chain)
        squeeze = kwargs.pop('squeeze', True)

        results = {chain: {} for chain in chains}
        for var in vars:
            samples = pymc3_obj.get_values(var, chains=chains, burn=burn,
                                           thin=thin, combine=combine,
                                           squeeze=False)
            for chain, data in zip(chains, samples):
                results[chain][var] = f(np.squeeze(data), *args, **kwargs)

        if squeeze and (len(chains) == 1 or combine):
            results = results[chains[0]]
        return results

    wrapped_f.__doc__ = f.__doc__
    wrapped_f.__name__ = f.__name__

    return wrapped_f


def _hpd_df(x, alpha):
    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]
    return pd.DataFrame(hpd(x, alpha), columns=cnames)

@statfunc
def hpd(x, alpha=0.05, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    This function assumes the posterior distribution is unimodal:
    it always returns one interval per variable.

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)
      transform : callable
          Function to transform data (defaults to identity)

    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))


def make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices

def dict2pd(statdict, labelname):
    """Small helper function to transform a diagnostics output dict into a
    pandas Series.
    """
    var_dfs = []
    for key, value in statdict.items():
        var_df = pd.Series(value.flatten())
        var_df.index = ttab.create_flat_names(key, value.shape)
        var_dfs.append(var_df)
    statpd = pd.concat(var_dfs, axis=0)
    statpd = statpd.rename(labelname)
    return statpd


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max



@statfunc
def mc_error(x, batches=5):
    R"""Calculates the simulation standard error, accounting for non-independent
        samples. The trace is divided into batches, and the standard deviation of
        the batch means is calculated.

    Parameters
    ----------
    x : Numpy array
              An array containing MCMC samples
    batches : integer
              Number of batches

    Returns
    -------
    `float` representing the error
    """
    if x.ndim > 1:

        dims = np.shape(x)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        trace = np.transpose([t.ravel() for t in x])

        return np.reshape([mc_error(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1:
            return np.std(x) / np.sqrt(len(x))

        try:
            batched_traces = np.resize(x, (batches, int(len(x) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(x) % batches
            new_shape = (batches, (len(x) - resid) / batches)
            batched_traces = np.resize(x[:-resid], new_shape)

        means = np.mean(batched_traces, 1)

        return np.std(means) / np.sqrt(batches)
