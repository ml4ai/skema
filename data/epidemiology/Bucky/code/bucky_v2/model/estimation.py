"""Submodule that manages some of the calculations for estimating params from the historical data."""

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END
from ..util.fractional_slice import frac_last_n_vals

# TODO there's alot of repeated code between estimate_chr/cfr, they should be generalized
# @sync_numerical_libs
# def gamma_delayed_ratio(mean_delay, k, numer, denom)

# TODO refactor crr to be more like the other estimates (use the gamma delay)

# TODO lots of misnamed variables (namely anything with 'rolling' in the name...)


###CTM @sync_numerical_libs
def estimate_crr(g_data, case_to_death_lag, ifr, days_back=14, case_lag=None, min_deaths=100.0, S_dist=1.0):
    """Estimate the case reporting rate based off observed vs. expected CFR."""
    # TODO rename vars to be more clear about what is cfr/ifr

    if case_lag is None:
        adm0_cfr_by_age = xp.sum(S_dist * ifr * g_data.Nij, axis=1) / xp.sum(g_data.Nj, axis=0)
        adm0_cfr_total = xp.sum(
            xp.sum(S_dist * ifr * g_data.Nij, axis=1) / xp.sum(g_data.Nj, axis=0),
            axis=0,
        )
        case_lag = xp.sum(case_to_death_lag * adm0_cfr_by_age / adm0_cfr_total, axis=0)

    case_lag_int = int(case_lag)
    recent_cum_cases = g_data.csse_data.cumulative_cases - g_data.csse_data.cumulative_cases[0]
    recent_cum_deaths = g_data.csse_data.cumulative_deaths - g_data.csse_data.cumulative_deaths[0]
    case_lag_frac = case_lag % 1  # TODO replace with util function for the indexing
    cases_lagged = frac_last_n_vals(recent_cum_cases, days_back + case_lag_frac, offset=case_lag_int)
    if case_lag_frac:
        cases_lagged = cases_lagged[0] + cases_lagged[1:]

    # adm0
    adm0_cfr_param = xp.sum(xp.sum(S_dist * ifr * g_data.Nij, axis=1) / xp.sum(g_data.Nj, axis=0), axis=0)
    adm0_cfr_reported = xp.sum(recent_cum_deaths[-days_back:], axis=1) / xp.sum(cases_lagged, axis=1)
    adm0_case_report = adm0_cfr_param / adm0_cfr_reported

    # if self.debug:
    #    logging.debug("Adm0 case reporting rate: " + pformat(adm0_case_report))
    # if xp.any(~xp.isfinite(adm0_case_report)):
    #    if self.debug:
    #        logging.debug("adm0 case report not finite")
    #        logging.debug(adm0_cfr_param)
    #        logging.debug(self.adm0_cfr_reported)
    #    raise SimulationException

    case_report = xp.repeat(adm0_case_report[:, None], cases_lagged.shape[-1], axis=1)

    # adm1
    adm1_cfr_param = xp.zeros((g_data.max_adm1 + 1,), dtype=float)
    adm1_totpop = g_data.adm1_Nj  # xp.zeros((self.g_data.max_adm1 + 1,), dtype=float)

    tmp_adm1_cfr = xp.sum(S_dist * ifr * g_data.Nij, axis=0)

    xp.scatter_add(adm1_cfr_param, g_data.adm1_id, tmp_adm1_cfr)
    # xp.scatter_add(adm1_totpop, self.g_data.adm1_id, self.Nj)
    adm1_cfr_param /= adm1_totpop

    # adm1_cfr_reported is const, only calc it once and cache it
    adm1_deaths_reported = g_data.sum_adm1(recent_cum_deaths[-days_back:].T, cache=True)
    adm1_lagged_cases = g_data.sum_adm1(cases_lagged.T)
    # adm1_deaths_reported = xp.zeros((g_data.max_adm1 + 1, days_back), dtype=float)
    # adm1_lagged_cases = xp.zeros((g_data.max_adm1 + 1, days_back), dtype=float)

    # TODO use sum_adm1...
    # xp.scatter_add(
    #    adm1_deaths_reported,
    #    g_data.adm1_id,
    #    recent_cum_deaths[-days_back:].T,
    # )
    # xp.scatter_add(adm1_lagged_cases, g_data.adm1_id, cases_lagged.T)

    adm1_cfr_reported = adm1_deaths_reported / adm1_lagged_cases

    adm1_case_report = (adm1_cfr_param[:, None] / adm1_cfr_reported)[g_data.adm1_id].T

    valid_mask = (adm1_deaths_reported > min_deaths)[g_data.adm1_id].T & xp.isfinite(adm1_case_report)
    case_report[valid_mask] = adm1_case_report[valid_mask]

    # adm2
    adm2_cfr_param = xp.sum(S_dist * ifr * (g_data.Nij / g_data.Nj), axis=0)

    adm2_cfr_reported = recent_cum_deaths[-days_back:] / cases_lagged
    adm2_case_report = adm2_cfr_param / adm2_cfr_reported

    valid_adm2_cr = xp.isfinite(adm2_case_report) & (recent_cum_deaths[-days_back:] > min_deaths)
    case_report[valid_adm2_cr] = adm2_case_report[valid_adm2_cr]

    # from IPython import embed
    # embed()
    case_report = 2.0 / (1.0 / adm0_case_report[..., None] + 1.0 / case_report)

    return case_report


###CTM @sync_numerical_libs
def estimate_chr(
    g_data,
    base_CHR,
    I_to_H_time,
    Rh_gamma_k,
    S_age_dist,
    days_back=7,
):
    """Estimate CHR from recent case data."""

    mean = I_to_H_time

    adm2_mean = xp.sum(S_age_dist * mean[..., None], axis=0)
    k = Rh_gamma_k

    rolling_case_hist = g_data.csse_data.incident_cases
    rolling_hosp_hist = g_data.hhs_data.incident_hospitalizations

    t_max = rolling_case_hist.shape[0]
    x = xp.arange(0.0, t_max)

    # adm0
    adm0_inc_cases = xp.sum(rolling_case_hist, axis=1)
    adm0_inc_hosp = xp.sum(rolling_hosp_hist, axis=1)

    adm0_theta = xp.sum(adm2_mean * g_data.Nj / g_data.N) / k

    w = 1.0 / (xp.special.gamma(k) * adm0_theta**k) * x ** (k - 1) * xp.exp(-x / adm0_theta)
    w = w / (1.0 - w)

    w = w / xp.sum(w)
    w = w[::-1]

    chr_ = xp.empty((days_back,))
    for i in range(days_back):
        d = i + 1
        chr_[i] = adm0_inc_hosp[-d] / (xp.sum(w[d:] * adm0_inc_cases[:-d], axis=0))

    adm0_chr = 1.0 / xp.nanmean(1.0 / chr_, axis=0)

    # adm1
    adm1_inc_cases = g_data.sum_adm1(rolling_case_hist.T).T
    adm1_inc_hosp = rolling_hosp_hist

    adm1_theta = g_data.sum_adm1(adm2_mean * g_data.Nj) / g_data.adm1_Nj / k

    x = xp.tile(x, (adm1_theta.shape[0], 1)).T
    w = 1.0 / (xp.special.gamma(k) * adm1_theta**k) * x ** (k - 1) * xp.exp(-x / adm1_theta)
    w = w / (1.0 - w)
    w = w / xp.sum(w, axis=0)
    w = w[::-1]
    chr_ = xp.empty((days_back, adm1_theta.shape[0]))
    for i in range(days_back):
        d = i + 1
        chr_[i] = adm1_inc_hosp[-d] / (xp.sum(w[d:] * adm1_inc_cases[:-d], axis=0))

    adm1_chr = 1.0 / xp.nanmean(1.0 / chr_, axis=0)

    baseline_adm1_chr = g_data.sum_adm1(xp.sum(base_CHR * S_age_dist, axis=0) * g_data.Nj) / g_data.adm1_Nj

    chr_fac = (adm1_chr / baseline_adm1_chr)[g_data.adm1_id]

    baseline_adm0_chr = xp.sum(xp.sum(base_CHR * S_age_dist, axis=0) * g_data.Nj) / g_data.N
    adm0_chr_fac = adm0_chr / baseline_adm0_chr
    valid = xp.isfinite(chr_fac) & (chr_fac > 0.002) & (xp.mean(adm1_inc_hosp[-7:]) > 4.0)
    chr_fac[~valid] = adm0_chr_fac

    return xp.clip(base_CHR * chr_fac, 0.0, 1.0)


###CTM @sync_numerical_libs
def estimate_cfr(
    g_data,
    base_CFR,
    case_to_death_time,
    Rh_gamma_k,
    S_age_dist,
    days_back=7,
):
    """Estimate CFR from recent case data."""

    mean = case_to_death_time  # params["H_TIME"] + params["I_TO_H_TIME"] #+ params["D_REPORT_TIME"]
    adm2_mean = xp.sum(S_age_dist * mean[..., None], axis=0)
    k = Rh_gamma_k

    rolling_case_hist = g_data.csse_data.incident_cases
    rolling_death_hist = g_data.csse_data.incident_deaths

    t_max = rolling_case_hist.shape[0]
    x = xp.arange(0.0, t_max)

    # adm0
    adm0_inc_cases = xp.sum(rolling_case_hist, axis=1)
    adm0_inc_deaths = xp.sum(rolling_death_hist, axis=1)

    adm0_theta = xp.sum(adm2_mean * g_data.Nj / g_data.N) / k

    w = 1.0 / (xp.special.gamma(k) * adm0_theta**k) * x ** (k - 1) * xp.exp(-x / adm0_theta)
    w = w / (1.0 - w)
    w = w / xp.sum(w)
    w = w[::-1]

    # n_loc = rolling_case_hist.shape[1]
    cfr = xp.empty((days_back,))
    for i in range(days_back):
        d = i + 1
        cfr[i] = adm0_inc_deaths[-d] / (xp.sum(w[d:] * adm0_inc_cases[:-d], axis=0))

    adm0_cfr = 1.0 / xp.nanmean(1.0 / cfr, axis=0)

    # adm1
    adm1_inc_cases = g_data.sum_adm1(rolling_case_hist.T).T
    adm1_inc_deaths = g_data.sum_adm1(rolling_death_hist.T).T

    adm1_theta = g_data.sum_adm1(adm2_mean * g_data.Nj) / g_data.adm1_Nj / k

    x = xp.tile(x, (adm1_theta.shape[0], 1)).T
    w = 1.0 / (xp.special.gamma(k) * adm1_theta**k) * x ** (k - 1) * xp.exp(-x / adm1_theta)
    w = w / (1.0 - w)
    w = w / xp.sum(w, axis=0)
    w = w[::-1]
    cfr = xp.empty((days_back, adm1_theta.shape[0]))
    for i in range(days_back):
        d = i + 1
        cfr[i] = adm1_inc_deaths[-d] / (xp.sum(w[d:] * adm1_inc_cases[:-d], axis=0))

    adm1_cfr = 1.0 / xp.nanmean(1.0 / cfr, axis=0)

    baseline_adm1_cfr = g_data.sum_adm1(xp.sum(base_CFR * S_age_dist, axis=0) * g_data.Nj) / g_data.adm1_Nj

    cfr_fac = (adm1_cfr / baseline_adm1_cfr)[g_data.adm1_id]

    baseline_adm0_cfr = xp.sum(xp.sum(base_CFR * S_age_dist, axis=0) * g_data.Nj) / g_data.N
    adm0_cfr_fac = adm0_cfr / baseline_adm0_cfr
    valid = xp.isfinite(cfr_fac) & (cfr_fac > 0.002) & (xp.mean(adm1_inc_deaths[-days_back:]) > 4.0)
    cfr_fac[~valid] = adm0_cfr_fac

    # cfr_fac = 2.0 / (1.0 / cfr_fac + 1.0 / adm0_cfr_fac[..., None])
    # cfr_fac = xp.sqrt(cfr_fac * adm0_cfr_fac[..., None])

    return xp.clip(base_CFR * cfr_fac, 0.0, 1.0)


###CTM @sync_numerical_libs
def estimate_Rt(
    g_data,
    generation_interval,
    E_gamma_k,
    days_back=7,
    case_reporting=None,
    # use_geo_mean=False,
):
    """Estimate R_t from the recent case data."""

    rolling_case_hist = g_data.csse_data.incident_cases[-case_reporting.shape[0] :] / case_reporting

    rolling_case_hist = xp.clip(rolling_case_hist, a_min=0.0, a_max=None)

    tot_case_hist = (g_data.Aij.A.T @ rolling_case_hist.T).T + 1.0  # to avoid weirdness with small numbers

    t_max = rolling_case_hist.shape[0]
    k = E_gamma_k

    mean = generation_interval
    theta = mean / k
    x = xp.arange(0.0, t_max)

    w = 1.0 / (xp.special.gamma(k) * theta**k) * x ** (k - 1) * xp.exp(-x / theta)
    w = w / (1.0 - w)
    w = w / xp.sum(w)
    w = w[::-1]
    # adm0
    rolling_case_hist_adm0 = xp.nansum(rolling_case_hist, axis=1)[:, None]
    tot_case_hist_adm0 = xp.nansum(tot_case_hist, axis=1)[:, None]

    n_loc = rolling_case_hist_adm0.shape[1]
    Rt = xp.empty((days_back, n_loc))
    for i in range(days_back):  # TODO we can vectorize by convolving w over case hist
        d = i + 1
        Rt[i] = rolling_case_hist_adm0[-d] / (xp.sum(w[d:, None] * tot_case_hist_adm0[:-d], axis=0))

    # Take harmonic mean
    Rt[~(Rt > 0.0)] = xp.nan
    Rt = 1.0 / xp.nanmean(1.0 / Rt, axis=0)

    Rt_out = xp.full((rolling_case_hist.shape[1],), Rt)

    # adm1
    tot_case_hist_adm1 = g_data.sum_adm1(tot_case_hist.T).T
    rolling_case_hist_adm1 = g_data.sum_adm1(rolling_case_hist.T).T

    n_loc = rolling_case_hist_adm1.shape[1]
    Rt = xp.empty((days_back, n_loc))
    for i in range(days_back):
        d = i + 1
        Rt[i] = rolling_case_hist_adm1[-d] / (xp.sum(w[d:, None] * tot_case_hist_adm1[:-d], axis=0))

    # take harmonic mean
    Rt[~(Rt > 0.0)] = xp.nan
    Rt = 1.0 / xp.nanmean(1.0 / Rt, axis=0)

    # TODO we should mask this before projecting it to adm2...
    Rt = Rt[g_data.adm1_id]
    Rt_adm1 = Rt[g_data.adm1_id]
    valid_mask = xp.isfinite(Rt) & (xp.mean(rolling_case_hist_adm1[-7:], axis=0)[g_data.adm1_id] > 50)
    Rt_out[valid_mask] = Rt[valid_mask]

    # adm2
    n_loc = rolling_case_hist.shape[1]
    Rt = xp.empty((days_back, n_loc))
    for i in range(days_back):
        d = i + 1
        Rt[i] = rolling_case_hist[-d] / (xp.sum(w[d:, None] * tot_case_hist[:-d], axis=0))

    Rt[~(Rt > 0.0)] = xp.nan

    # rt_geo = xp.exp(xp.nanmean(xp.log(Rt), axis=0))
    # rt_mean = xp.nanmean(Rt, axis=0)
    # rt_med = xp.nanmedian(Rt, axis=0)
    rt_harm = 1.0 / xp.nanmean(1.0 / Rt, axis=0)

    Rt = rt_harm  # (rt_geo + rt_med) /2.
    # TODO make this max value a param
    valid_mask = xp.isfinite(Rt) & (xp.mean(rolling_case_hist[-7:], axis=0) > 50) & (Rt > 0.1) & (Rt < 5)
    Rt_out[valid_mask] = Rt[valid_mask]
    Rt_out = 2.0 / (1.0 / Rt_adm1 + 1.0 / Rt_out)
    return Rt_out
