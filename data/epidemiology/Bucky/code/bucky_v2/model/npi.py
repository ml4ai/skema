"""Module to parse npi csv files."""
import datetime
import logging

#import numpy as np TF: numpy import currently breaks generation
import pandas as pd

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END


###CTM @sync_numerical_libs
def get_npi_params(g_data, first_date, t_max, npi_file=None, disable_npi=False):
    """Read an npi scenario file or if none is provided provide correctly shaped 'no future changes' npi_params."""

    n_nodes = g_data.Nij.shape[-1]
    if npi_file is not None:
        logging.info(f"Using NPI from: {npi_file}")
        npi_params = read_npi_file(
            npi_file,
            first_date,
            t_max,
            g_data.adm2_id,
            disable_npi,
        )
        for k in npi_params:
            npi_params[k] = xp.array(npi_params[k])
            if k == "contact_weights":
                npi_params[k] = xp.broadcast_to(npi_params[k], (t_max + 1, n_nodes, 4))
            else:
                npi_params[k] = xp.broadcast_to(npi_params[k], (t_max + 1, n_nodes))
        npi_params["npi_active"] = True
    else:
        npi_params = {
            "npi_active": False,
            "r0_reduct": xp.broadcast_to(xp.ones(1), (t_max + 1, n_nodes)),
            "contact_weights": xp.broadcast_to(xp.ones(1), (t_max + 1, n_nodes, 4)),
            "mobility_reduct": xp.broadcast_to(xp.ones(1), (t_max + 1, n_nodes)),
        }
    return npi_params


def read_npi_file(fname, start_date, end_t, adm2_map, disable_npi=False):
    """TODO Description.

    Parameters
    ----------
    fname : str
        Filename of NPI file
    start_date : str
        Start date to use
    end_t : int
        Number of days after start date
    adm2_map : ndarray
        Array of adm2 IDs
    disable_npi : bool, optional
        Bool indicating whether NPIs should be disabled after being initialized for the first day

    Returns
    -------
    npi_params : dict
        TODO
    """

    # filter by overlap with simulation date range
    df = pd.read_csv(fname)
    df["date"] = pd.to_datetime(df.date)  # force a parse in case it's an odd format
    # rename adm2 column b/c people keep using different names
    df = df.rename(columns={"admin2": "adm2", "FIPS": "adm2"})
    end_date = start_date + datetime.timedelta(days=end_t)
    mask = (df["date"] >= str(start_date)) & (df["date"] <= str(end_date))
    # If npi file isn't up to date just use last known value
    if np.all(~mask):
        max_npi_date = df["date"].max()
        mask = df["date"] == max_npi_date

    df = df.loc[mask]

    npi_params = {}
    r0_reductions = []
    mobility_reductions = []
    contact_weights = []

    # 1st dimension is date, 2nd is admin2 code
    for _, group in df.sort_values(by=["date"]).groupby("date"):
        # convert adm2 id to int
        group["admin2"] = group.adm2.astype(int)
        date_group = group.set_index("adm2").reindex(xp.to_cpu(adm2_map))
        r0_reduction = np.array(date_group[["r0_reduction"]])
        mobility_reduction = np.array(date_group[["mobility_reduction"]])
        contact_weight = np.array(date_group[["home", "other_locations", "school", "work"]])
        r0_reductions.append(r0_reduction)
        mobility_reductions.append(mobility_reduction)
        contact_weights.append(contact_weight)

    npi_params["r0_reduct"] = np.array(r0_reductions)
    npi_params["mobility_reduct"] = np.array(mobility_reductions)
    npi_params["contact_weights"] = np.array(contact_weights)

    for key, value in npi_params.items():
        logging.debug(str(key) + str(value.shape))

        # forward fill with last defined date
        tmp = np.repeat(value[-1][None, ...], end_t + 1 - value.shape[0], axis=0)
        npi_params[key] = np.squeeze(np.concatenate((value, tmp), axis=0))

    if disable_npi:
        npi_params["mobility_reduct"].fill(1.0)
        npi_params["contact_weights"].fill(1.0)
        npi_params["r0_reduct"] = 1.0 / npi_params["r0_reduct"]
    else:
        # rescale the r0 scaling such that it's 1 on the first day because the doubling time is set
        # to match case history @ that date (i.e, it's not unmitigated, it's 'currently mitigated')
        # This doesn't need to happen for Cij or Aij
        npi_params["r0_reduct"] /= npi_params["r0_reduct"][0]

    # Fill any missing values with 1. (in case we don't have all the adm2 in the file)
    for k in npi_params:
        npi_params[k] = np.nan_to_num(npi_params[k], nan=1.0)

    return npi_params
