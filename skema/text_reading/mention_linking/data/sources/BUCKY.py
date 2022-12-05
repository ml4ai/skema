## ----------------------------------------------------------------------------
## 2022-10-01

# NOTE: The following is a modified version of the Bucky code base.
#   This version combines the Bucky implementations of functions and
#   declarations into a single file. This version removes a number
#   of idioms that are not yet supported by the SKEMA Code2FN python
#   pipeline (listed next); due to these changes, this code does NOT
#   replicate the functionality of the Bucky system; it is instead
#   intended to preserve the majority of the idioms that are currently
#   supported and serves as a first step towards ingesting the
#   full Bucky framework.

# ----- Gromet idioms not yet handled:
# default_argument_values
# unpack_seq: *seq
# unpack_dict: **dict
# class_inheritance
# class_superclass_call
# classes with additional Field definitions in fns other than __init__ constructor
# class method decorators
#     @staticmethod
#     @property : https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work-in-python
# dunder functions:
#     __call__ in a class  <-- this does not occur in Bucky, but is something to consider
# general decorators:
#     @lru_cache(maxsize=None)
# exception_handling : try... except... raise...
# ellipsis: ...  : (line 592)
#   https://www.geeksforgeeks.org/what-is-three-dots-or-ellipsis-in-python3/
#   https://stackoverflow.com/questions/42190783/what-does-three-dots-in-python-mean-when-indexing-what-looks-like-a-number
#   https://www.youtube.com/watch?v=65_-6kEAq58
# with_clause

# ----- Maybe
# assignment_operator
# multiple_value_assignment
# compound_if_condition
# string_concatenation: "blee" + "blah"
# structured_literals: list, tuple dict
# multiple_iterator in for loop
# comprehension_list
# comprehension_dict

## ----------------------------------------------------------------------------

## imports - native
import datetime
import logging
import os
import pickle
import queue
import random
import sys
import threading
import warnings

## imports - native - read_config.py
import pathlib

## imports - native - arg_parser_model.py
import glob
import argparse
import importlib

## imports - native - util.py
import copy

## imports - native - parameters.py
from pprint import pformat
import yaml

## imports - native - state.py
import inspect

## imports - other
import networkx as nx
import pandas as pd

## imports - other - util.py
import tqdm

## imports - other - distributions.py
import numpy as np
import scipy.special as sc

## imports - other - numerical_libs.py
import contextlib
import numpy as xp
import scipy.integrate._ivp.ivp as ivp  # noqa: F401  # pylint: disable=unused-import
import scipy.sparse as sparse  # noqa: F401  # pylint: disable=unused-import


# supress pandas warning caused by pyarrow
warnings.simplefilter(action="ignore", category=FutureWarning)
# TODO we do alot of allowing div by 0 and then checking for nans later, we should probably refactor that
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# @lru_cache(maxsize=None)  ## suppressing for bucky_simplified_v1
def get_runid():  # TODO move to util and rename to timeid or something
    dt_now = datetime.datetime.now()
    return str(dt_now).replace(" ", "__").replace(":", "_").split(".")[0]


# -----------------------------------------------------------------------------
# util.py
# -----------------------------------------------------------------------------

## xp.scatter_add = xp.add.at
## xp.optimize_kernels = contextlib.nullcontext
## xp.to_cpu = lambda x, **kwargs: x  # one arg noop


def _banner():
    print(r" ____             _          ")  # noqa: T001
    print(r"| __ ) _   _  ___| | ___   _ ")  # noqa: T001
    print(r"|  _ \| | | |/ __| |/ / | | |")  # noqa: T001
    print(r"| |_) | |_| | (__|   <| |_| |")  # noqa: T001
    print(r"|____/ \__,_|\___|_|\_\\__, |")  # noqa: T001
    print(r"                       |___/ ")  # noqa: T001
    print(r"                             ")  # noqa: T001


# https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
## class TqdmLoggingHandler(logging.Handler):  ## TODO class_inheritance
class TqdmLoggingHandler:
    def __init__(self, level=logging.NOTSET):  # pylint: disable=useless-super-delegation
        ## super().__init__(level)  ## TODO class_superclass_call
        pass

    def emit(self, record):
        ## replacement:  ## TODO
        msg = "NON-FUNCTIONAL"  ## self.format(record)
        tqdm.tqdm.write(msg)
        ## self.flush()
        ## replaced: ## TODO exception_handling
        # try:
        #     msg = self.format(record)
        #     tqdm.tqdm.write(msg)
        #     self.flush()
        # except (KeyboardInterrupt, SystemExit):  # pylint: disable=try-except-raise
        #     raise
        # except Exception:  # pylint: disable=broad-except
        #     self.handleError(record)


## replacement ## TODO
class dotdict():
    # """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, d):
        pass

    def __deepcopy__(self, memo=None):
        ## return dotdict({key: copy.deepcopy(value) for key, value in self.items()})  ## TODO comprehension_dict , multiple_iterator
        _dict = dict()
        for _iter in self.items():
            key = _iter[0]
            value = _iter[1]
            _dict[key] = copy.deepcopy(value)
        return dotdict(_dict)

## TODO class_inheritance
## replaced:
# class dotdict(dict):
#     """dot.notation access to dictionary attributes."""
#
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
#
#     def __deepcopy__(self, memo=None):
#         return dotdict({key: copy.deepcopy(value) for key, value in self.items()})


def remove_chars(seq):
    seq_type = type(seq)
    if seq_type != str:
        return seq

    return seq_type().join(filter(seq_type.isdigit, seq))


# -----------------------------------------------------------------------------
# distributions.py
# -----------------------------------------------------------------------------

# TODO only works on cpu atm
# we'd need to implement betaincinv ourselves in cupy
def mPERT_sample(mu, a=0.0, b=1.0, gamma=4.0, var=None):
    # """Provides a vectorized Modified PERT distribution.
    
    #Parameters
    #----------
    #mu : float, array_like
    #    Mean value for the PERT distribution.
    # a : float, array_like
    #    Lower bound for the distribution.
    # b : float, array_like
    #    Upper bound for the distribution.
    # gamma : float, array_like
    #    Shape paramter.
    # var : float, array_like, None
    #    Variance of the distribution. If var != None,
    #    gamma will be calcuated to meet the desired variance.

    # Returns
    # -------
    # out : float, array_like
    #    Samples drawn from the specified mPERT distribution.
    #    Shape is the broadcasted shape of the the input parameters.

    # """
    mu, a, b = np.atleast_1d(mu, a, b)  ## TODO multiple_value_assignment
    if var is not None:
        gamma = (mu - a) * (b - mu) / var - 3.0
    alp1 = 1.0 + gamma * ((mu - a) / (b - a))
    alp2 = 1.0 + gamma * ((b - mu) / (b - a))
    u = np.random.random_sample(mu.shape)
    alp3 = sc.betaincinv(alp1, alp2, u)
    return (b - a) * alp3 + a


def truncnorm(xp, loc=0.0, scale=1.0, size=1, a_min=None, a_max=None):
    #"""Provides a vectorized truncnorm implementation that is compatible with cupy.
    #
    #The output is calculated by using the numpy/cupy random.normal() and
    #truncted via rejection sampling. The interface is intended to mirror
    #the scipy implementation of truncnorm.
    #
    #Parameters
    #----------
    #xp : module
    #
    #
    #Returns
    #-------
    #
    #"""
    ret = xp.random.normal(loc, scale, size)
    if a_min is None:
        a_min = -xp.inf
    if a_max is None:
        a_max = xp.inf

    while True:
        valid = (ret > a_min) & (ret < a_max)
        if valid.all():
            return ret
        ret[~valid] = xp.random.normal(loc, scale, ret[~valid].shape)


# -----------------------------------------------------------------------------
# parameters.py
# -----------------------------------------------------------------------------

def calc_Te(Tg, Ts, n, f):
    num = 2.0 * n * f / (n + 1.0) * Tg - Ts
    den = 2.0 * n * f / (n + 1.0) - 1.0
    return num / den


def calc_Reff(m, n, Tg, Te, r):
    num = 2.0 * n * r / (n + 1.0) * (Tg - Te) * (1.0 + r * Te / m) ** m
    den = 1.0 - (1.0 + 2.0 * r / (n + 1.0) * (Tg - Te)) ** (-n)
    return num / den


def calc_Ti(Te, Tg, n):
    return (Tg - Te) * 2.0 * n / (n + 1.0)


def CI_to_std(CI):
    lower, upper = CI
    std95 = np.sqrt(1.0 / 0.05)
    return (upper + lower) / 2.0, (upper - lower) / std95 / 2.0


class buckyParams:
    def __init__(self, par_file=None):

        self.par_file = par_file
        if par_file is not None:
            self.base_params = self.read_yml(par_file)
            self.consts = dotdict(self.base_params["consts"])
        else:
            self.base_params = None

    ## @staticmethod  ## TODO class_staticmethod
    def read_yml(self, par_file):
        # TODO check file exists

        ## with open(par_file, "rb") as f:  ## TODO with_clause open
        f = open(par_file, "rb")
        loaded_yaml = yaml.load(f, yaml.SafeLoader)  # nosec
        f.close()
        return loaded_yaml

    def generate_params(self, var=0.2):
        if var is None:
            var = 0.0
        while True:  # WTB python do-while...
            params = self.reroll_params(self.base_params, var)
            params = self.calc_derived_params(params)
            if (params.Te > 1.0 and params.Tg > params.Te and params.Ti > 1.0) or var == 0.0:  ## TODO compound_if_condition
                return params
            logging.debug("Rejected params: " + pformat(params))

    def reroll_params(self, base_params, var):
        params = dotdict({})
        for p in base_params:
            # Scalars
            if "gamma" in base_params[p]:
                mu = copy.deepcopy(base_params[p]["mean"])
                params[p] = mPERT_sample(np.array([mu]), gamma=base_params[p]["gamma"])

            elif "mean" in base_params[p]:
                if "CI" in base_params[p]:
                    if var:
                        ## params[p] = truncnorm(np, *CI_to_std(base_params[p]["CI"]), a_min=1e-6)  ## TODO unpack_seq
                        pass
                    else:  # just use mean if we set var to 0
                        params[p] = copy.deepcopy(base_params[p]["mean"])
                else:
                    params[p] = copy.deepcopy(base_params[p]["mean"])
                    params[p] *= truncnorm(np, loc=1.0, scale=var, a_min=1e-6)  ## TODO assignment_operator

            # age-based vectors
            elif "values" in base_params[p]:
                params[p] = np.array(base_params[p]["values"])
                ## params[p] *= truncnorm(np, 1.0, var, size=params[p].shape, a_min=1e-6)  ## TODO assignment_operator
                params[p] = params[p] * truncnorm(np, 1.0, var, size=params[p].shape, a_min=1e-6)  ## TODO
                # interp to our age bins
                if base_params[p]["age_bins"] != base_params["consts"]["age_bins"]:
                    params[p] = self.age_interp(
                        base_params["consts"]["age_bins"],
                        base_params[p]["age_bins"],
                        params[p],
                    )

            # fixed values (noop)
            else:
                params[p] = copy.deepcopy(base_params[p])

            # clip values
            if "clip" in base_params[p]:
                clip_range = base_params[p]["clip"]
                params[p] = np.clip(params[p], clip_range[0], clip_range[1])

        return params

    ## @staticmethod  ## TODO class_staticmethod
    def age_interp(self, x_bins_new, x_bins, y):  # TODO we should probably account for population for the 65+ type bins...
        x_mean_new = np.mean(np.array(x_bins_new), axis=1)
        x_mean = np.mean(np.array(x_bins), axis=1)
        return np.interp(x_mean_new, x_mean, y)

    ## @staticmethod  ## TODO class_staticmethod
    def rescale_doubling_rate(self, D, params, xp, A_diag=None):
        # TODO rename D to Td everwhere for consistency
        r = xp.log(2.0) / D
        params["R0"] = calc_Reff(
            params["consts"]["Im"],
            params["consts"]["En"],
            params["Tg"],
            params["Te"],
            r,
        )
        params["BETA"] = params["R0"] * params["GAMMA"]
        if A_diag is not None:
            # params['BETA'] /= xp.sum(A,axis=1)
            ## params["BETA"] /= A_diag  ## TODO assignment_operator
            params["BETA"] = params["BETA"] / A_diag
        return params

    ## @staticmethod  ## TODO class_staticmethod
    def calc_derived_params(self, params):
        params["Te"] = calc_Te(
            params["Tg"],
            params["Ts"],
            params["consts"]["En"],
            params["frac_trans_before_sym"],
        )
        params["Ti"] = calc_Ti(params["Te"], params["Tg"], params["consts"]["En"])
        r = np.log(2.0) / params["D"]
        params["R0"] = calc_Reff(
            params["consts"]["Im"],
            params["consts"]["En"],
            params["Tg"],
            params["Te"],
            r,
        )

        params["SIGMA"] = 1.0 / params["Te"]
        params["GAMMA"] = 1.0 / params["Ti"]
        params["BETA"] = params["R0"] * params["GAMMA"]
        params["SYM_FRAC"] = 1.0 - params["ASYM_FRAC"]
        params["THETA"] = 1.0 / params["H_TIME"]
        params["GAMMA_H"] = 1.0 / params["I_TO_H_TIME"]
        return params


# -----------------------------------------------------------------------------
# arg_parser_model.py
# -----------------------------------------------------------------------------

def get_arg_parser():

    ## TODO inserted from read_config.py
    f = open("config.yml", "r")  ## TODO with_clause
    bucky_cfg = yaml.load(f, yaml.SafeLoader)
    f.close()
    bucky_cfg["base_dir"] = str(pathlib.Path.cwd())

    # TODO this logic should be in numerical_libs so we can apply it everywhere
    cupy_spec = importlib.util.find_spec("cupy")
    cupy_found = cupy_spec is not None

    if bool(os.getenv("BUCKY_CPU")) or False:
        logging.info("BUCKY_CPU found, forcing cpu usage")
        cupy_found = False

    most_recent_graph = max(
        glob.glob(bucky_cfg["data_dir"] + "/input_graphs/*.p"),
        key=os.path.getctime,
        default="Most recently created graph in <data_dir>/input_graphs",
    )

    parser = argparse.ArgumentParser(description="Bucky Model")

    parser.add_argument(
        "--graph",
        "-g",
        dest="graph_file",
        default=most_recent_graph,
        type=str,
        help="Pickle file containing the graph to run",
    )
    parser.add_argument(
        "par_file",
        default=bucky_cfg["base_dir"] + "/par/scenario_5.yml",
        nargs="?",
        type=str,
        help="File containing paramters",
    )
    parser.add_argument("--n_mc", "-n", default=100, type=int, help="Number of runs to do for Monte Carlo")
    parser.add_argument("--days", "-d", default=40, type=int, help="Length of the runs in days")
    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="Initial seed to generate PRNG seeds from (doesn't need to be high entropy)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="verbose output (repeat for increased verbosity; defaults to WARN, -v is INFO, -vv is DEBUG)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const=-1,
        default=0,
        dest="verbosity",
        help="quiet output (only show ERROR and higher)",
    )
    parser.add_argument(
        "-c",
        "--cache",
        action="store_true",
        help="Cache python files/par file/graph pickle for the run",
    )
    parser.add_argument(
        "-nmc",
        "--no_mc",
        action="store_true",
        help="Just do one run with the mean param values",
    )  # TODO rename to --mean or something

    # TODO this doesnt do anything other than let you throw and error if there's no cupy...
    parser.add_argument(
        "-gpu",
        "--gpu",
        action="store_true",
        default=cupy_found,
        help="Use cupy instead of numpy",
    )

    parser.add_argument(
        "-den",
        "--dense",
        action="store_true",
        help="Don't store the adj matrix as a sparse matrix. \
        This will be faster with a small number of regions or a very dense adj matrix.",
    )

    parser.add_argument(
        "-opt",
        "--opt",
        action="store_true",
        help="Enable cupy kernel optimizations. Do this for large runs using the gpu (n > 100).",
    )

    # TODO this should be able to take in the rejection factor thats hardcoded
    parser.add_argument(
        "-r",
        "--reject_runs",
        action="store_true",
        help="Reject Monte Carlo runs with incidence rates that don't align with historical data",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=bucky_cfg["raw_output_dir"],
        type=str,
        help="Dir to put the output files",
    )

    parser.add_argument("--npi_file", default=None, nargs="?", type=str, help="File containing NPIs")
    parser.add_argument(
        "--disable-npi",
        action="store_true",
        help="Disable all active NPI from the npi_file at the start of the run",
    )
    return parser


# -----------------------------------------------------------------------------
# npi.py
# -----------------------------------------------------------------------------

def read_npi_file(fname, start_date, end_t, adm2_map, disable_npi=False):
    # """TODO Description.

    #Parameters
    # ----------
    #fname : string
    #    Filename of NPI file
    #start_date : string
    #    Start date to use
    #end_t : int
    #    Number of days after start date
    #adm2_map : NumPy array
    #    Array of adm2 IDs
    #disable_npi : bool (default: False)
    #    Bool indicating whether NPIs should be disabled
    #Returns
    # -------
    #npi_params : dict
    #    TODO
    #"""
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
    ## for _, group in df.sort_values(by=["date"]).groupby("date"):  ## TODO multiple_iterator
    for _iter in df.sort_values(by=["date"]).groupby("date"):  ## TODO
        group = _iter[1]
        # convert adm2 id to int
        group["adm2"] = group.adm2.apply(remove_chars).astype(int)
        date_group = group.set_index("adm2").reindex(adm2_map)
        r0_reduction = np.array(date_group[["r0_reduction"]])
        mobility_reduction = np.array(date_group[["mobility_reduction"]])
        contact_weight = np.array(date_group[["home", "other_locations", "school", "work"]])
        r0_reductions.append(r0_reduction)
        mobility_reductions.append(mobility_reduction)
        contact_weights.append(contact_weight)

    npi_params["r0_reduct"] = np.array(r0_reductions)
    npi_params["mobility_reduct"] = np.array(mobility_reductions)
    npi_params["contact_weights"] = np.array(contact_weights)

    ## for key, value in npi_params.items():  ## TODO multiple_iterator
    for _iter in npi_params.items():  ## TODO
        key = _iter[0]
        value = _iter[1]
        logging.debug(str(key) + str(value.shape))

        # forward fill with last defined date
        tmp = np.repeat(value[-1][None, ...], end_t + 1 - value.shape[0], axis=0)  ## TODO ellipsis NOT_DONE
        npi_params[key] = np.squeeze(np.concatenate((value, tmp), axis=0))

    if disable_npi:
        npi_params["mobility_reduct"].fill(1.0)
        npi_params["contact_weights"].fill(1.0)
        npi_params["r0_reduct"] = 1.0 / npi_params["r0_reduct"]
    else:
        # rescale the r0 scaling such that it's 1 on the first day because the doubling time is set
        # to match case history @ that date (i.e, it's not unmitigated, it's 'currently mitigated')
        # This doesn't need to happen for Cij or Aij
        ## npi_params["r0_reduct"] /= npi_params["r0_reduct"][0]  ## TODO assignment_operator
        npi_params["r0_reduct"] = npi_params["r0_reduct"] / npi_params["r0_reduct"][0]

    # Fill any missing values with 1. (in case we don't have all the adm2 in the file)
    for k in npi_params:
        npi_params[k] = np.nan_to_num(npi_params[k], nan=1.0)

    return npi_params


# -----------------------------------------------------------------------------
# state.py
# -----------------------------------------------------------------------------

class buckyState:  # pylint: disable=too-many-instance-attributes
    def __init__(self, consts, Nij, state=None):

        ## removing
        # # use xp from the calling module
        # global xp
        # if xp is None:
        #     xp = inspect.currentframe().f_back.f_globals["xp"]

        self.En = consts["En"]  # TODO rename these to like gamma shape or something
        self.Im = consts["Im"]
        self.Rhn = consts["Rhn"]
        self.consts = consts

        indices = {"S": 0}
        indices["E"] = slice(1, self.En + 1)
        indices["I"] = slice(indices["E"].stop, indices["E"].stop + self.Im)
        indices["Ic"] = slice(indices["I"].stop, indices["I"].stop + self.Im)
        indices["Ia"] = slice(indices["Ic"].stop, indices["Ic"].stop + self.Im)
        indices["R"] = slice(indices["Ia"].stop, indices["Ia"].stop + 1)
        indices["Rh"] = slice(indices["R"].stop, indices["R"].stop + self.Rhn)
        indices["D"] = slice(indices["Rh"].stop, indices["Rh"].stop + 1)

        ## indices["Itot"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if k in ("I", "Ia", "Ic")])  ## TODO
        ## replacement
        indices_temp = list()
        for _iter in indices.items():
            k = _iter[0]
            v = _iter[1]
            if k in ("I", "Ia", "Ic"):
                indices_temp.append(xp.r_[v])
        indices["Itot"] = xp.concatenate(indices_temp)

        ## indices["H"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if k in ("Ic", "Rh")])  ## TODO
        ## replacement
        indices_temp = list()
        for _iter in indices.items():
            k = _iter[0]
            v = _iter[1]
            if k in ("Ic", "Rh"):
                indices_temp.append(xp.r_[v])
        indices["H"] = xp.concatenate(indices_temp)

        indices["incH"] = slice(indices["D"].stop, indices["D"].stop + 1)
        indices["incC"] = slice(indices["incH"].stop, indices["incH"].stop + 1)

        indices["N"] = slice(0, indices["D"].stop)

        self.indices = indices

        self.n_compartments = xp.to_cpu(indices["incC"].stop)

        # self.Nij = Nij
        self.n_age_grps, self.n_nodes = Nij.shape

        if state is None:
            self.state = xp.zeros(self.state_shape)
        else:
            self.state = state

    def __getattribute__(self, attr):

        ## replacement  ## TODO
        if attr in super().__getattribute__("indices"):  ## TODO class_superclass_call NOT_DONE
            out = self.state[self.indices[attr]]
            if out.shape[0] == 1:
                out = xp.squeeze(out, axis=0)
            return out
        ## replaced  ## TODO
        # try:
        #     if attr in super().__getattribute__("indices"):
        #         out = self.state[self.indices[attr]]
        #         if out.shape[0] == 1:
        #             out = xp.squeeze(out, axis=0)
        #         return out
        # except AttributeError:
        #     pass
        # return super().__getattribute__(attr)

    def __setattr__(self, attr, x):

        ## replacement  ## TODO
        if attr in super().__getattribute__("indices"):  ## TODO class_superclass_call NOT_DONE
            # TODO check that its a slice otherwise this wont work so we should warn
            self.state[self.indices[attr]] = x
        else:
            # super().__setattr__(attr, x)
            pass

        ## replaced  ## TODO exception_handling
        # try:
        #     if attr in super().__getattribute__("indices"):
        #         # TODO check that its a slice otherwise this wont work so we should warn
        #         self.state[self.indices[attr]] = x
        #     else:
        #         super().__setattr__(attr, x)
        # except AttributeError:
        #     super().__setattr__(attr, x)

    ## @property  ## TODO class_property
    def state_shape(self):
        return (self.n_compartments, self.n_age_grps, self.n_nodes)

    state_shape = property(state_shape)  ## TODO

    def init_S(self):
        self.S = 1.0 - xp.sum(self.state, axis=0)


# -----------------------------------------------------------------------------
# SEIR_covid
# -----------------------------------------------------------------------------

class SEIR_covid:
    def __init__(
            self,
            seed=None,
            randomize_params_on_reset=True,
            debug=False,
            sparse_aij=False,
            t_max=None,
            graph_file=None,
            par_file=None,
            npi_file=None,
            disable_npi=False,
            reject_runs=False,
    ):
        self.rseed = seed  # TODO drop this and only set seed in reset
        self.randomize = randomize_params_on_reset
        self.debug = debug
        self.sparse = sparse_aij  # we can default to none and autodetect
        # w/ override (maybe when #adm2 > 5k and some sparsity critera?)
        # TODO we could make a adj mat class that provides a standard api (stuff like .multiply,
        # overloaded __mul__, etc) so that we dont need to constantly check 'if self.sparse'.
        # It could also store that diag info and provide the row norm...

        # Integrator params
        self.t = 0.0
        self.dt = 1.0  # time step for model output (the internal step is adaptive...)
        self.t_max = t_max
        self.done = False
        self.run_id = get_runid()
        logging.info(f"Run ID: {self.run_id}")

        self.G = None
        self.graph_file = graph_file

        self.npi_file = npi_file
        self.disable_npi = disable_npi
        self.reject_runs = reject_runs

        self.output_dates = None

        # save files to cache
        # if args.cache:
        #    logging.warn("Cacheing is currently unsupported and probably doesnt work after the refactor")
        #    files = glob.glob("*.py") + [self.graph_file, args.par_file]
        #    logging.info(f"Cacheing: {files}")
        #    cache_files(files, self.run_id)

        # disease params
        self.bucky_params = buckyParams(par_file)
        self.consts = self.bucky_params.consts

    def reset(self, seed=None, params=None):

        # if you set a seed using the constructor, you're stuck using it forever
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            xp.random.seed(seed)
            self.rseed = seed

        #
        # Init graph
        #

        self.t = 0.0
        self.iter = 0
        self.done = False

        if self.G is None:
            # TODO break this out into functions like read_Nij, etc
            # maybe it belongs in a class

            logging.info("loading graph")

            ## with open(self.graph_file, "rb") as f:
            f = open(self.graph_file, "rb")
            G = pickle.load(f)  # nosec

            # Get case history from graph
            cum_case_hist = xp.vstack(list(nx.get_node_attributes(G, "case_hist").values())).T

            self.cum_case_hist = cum_case_hist.astype(float)
            self.inc_case_hist = xp.diff(cum_case_hist, axis=0).astype(float)
            self.inc_case_hist[self.inc_case_hist < 0.0] = 0.0

            # Get death history from graph
            cum_death_hist = xp.vstack(list(nx.get_node_attributes(G, "death_hist").values())).T

            self.cum_death_hist = cum_death_hist.astype(float)
            self.inc_death_hist = xp.diff(cum_death_hist, axis=0).astype(float)
            self.inc_death_hist[self.inc_death_hist < 0.0] = 0.0

            # TODO we should just remove these variables
            self.init_cum_cases = self.cum_case_hist[-1]
            self.init_cum_cases[self.init_cum_cases < 0.0] = 0.0
            self.init_deaths = self.cum_death_hist[-1]

            if "IFR" in G.nodes[list(G.nodes.keys())[0]]:
                logging.info("Using ifr from graph")
                self.use_G_ifr = True
                node_IFR = nx.get_node_attributes(G, "IFR")
                self.ifr = xp.asarray((np.vstack(list(node_IFR.values()))).T)
            else:
                self.use_G_ifr = False

            # grab the geo id's for later
            ## TODO comprehension_list
            # self.adm2_id = np.fromiter(
            #     [remove_chars(x) for x in nx.get_node_attributes(G, G.graph["adm2_key"]).values()], dtype=int
            # )
            _list = list()
            for x in nx.get_node_attributes(G, G.graph["adm2_key"]).values():
                _list.append(remove_chars(x))
            self.adm2_id = np.fromiter(
                _list, dtype=int
            )

            # Mapping from index to adm1
            ## TODO comprehension_list
            # self.adm1_id = np.fromiter(
            #     [remove_chars(x) for x in nx.get_node_attributes(G, G.graph["adm1_key"]).values()], dtype=int
            # )
            _list = list()
            for x in nx.get_node_attributes(G, G.graph["adm1_key"]).values():
                _list.append(remove_chars(x))
            self.adm1_id = np.fromiter(
                _list, dtype=int
            )

            self.adm1_id = xp.asarray(self.adm1_id, dtype=np.int32)
            self.adm1_max = xp.to_cpu(self.adm1_id.max())

            # Make contact mats sym and normalized
            self.contact_mats = G.graph["contact_mats"]
            if self.debug:
                logging.debug(f"graph contact mats: {G.graph['contact_mats'].keys()}")
            for mat in self.contact_mats:
                c_mat = xp.array(self.contact_mats[mat])
                c_mat = (c_mat + c_mat.T) / 2.0
                self.contact_mats[mat] = c_mat
            # remove all_locations so we can sum over the them ourselves
            if "all_locations" in self.contact_mats:
                del self.contact_mats["all_locations"]

            # TODO tmp to remove unused contact mats in como comparison graph
            # print(self.contact_mats.keys())
            valid_contact_mats = ["home", "work", "other_locations", "school"]

            ## self.contact_mats = {k: v for k, v in self.contact_mats.items() if k in valid_contact_mats}  ## TODO comprehension_dict , multiple_iterator
            _dict = dict()
            for _iter in self.contact_mats.items():
                k = _iter[0]
                v = _iter[1]
                if k in valid_contact_mats:
                    _dict[k] = v
            self.contact_mats = _dict

            ## self.Cij = xp.vstack([self.contact_mats[k][None, ...] for k in sorted(self.contact_mats)])  ## TODO ellipsis , comprehension_list
            _list = list()
            for k in sorted(self.contact_mats):
                _list.append(self.contact_mats[k][None, ...])  ## TODO ellipsis NOT_DONE
            self.Cij = xp.vstack(_list)

            # Get stratified population (and total)
            N_age_init = nx.get_node_attributes(G, "N_age_init")
            self.Nij = xp.asarray((np.vstack(list(N_age_init.values())) + 0.0001).T)
            self.Nj = xp.asarray(np.sum(self.Nij, axis=0))
            self.n_age_grps = self.Nij.shape[0]

            self.use_vuln = True
            if "vulnerable_frac" in G.nodes[0]:
                self.vulnerability_factor = 1.5
                self.use_vuln = True
                self.vulnerability_frac = xp.array(list(nx.get_node_attributes(G, "vulnerable_frac").values()))[
                    :, None
                ].T
            else:
                self.vulnerability_frac = xp.full(self.adm2_id.shape, 0.0)

            self.G = G
            n_nodes = self.Nij.shape[-1]  # len(self.G.nodes())

            self.first_date = datetime.date.fromisoformat(G.graph["start_date"])

            if self.npi_file is not None:
                logging.info(f"Using NPI from: {self.npi_file}")
                self.npi_params = read_npi_file(
                    self.npi_file,
                    self.first_date,
                    self.t_max,
                    self.adm2_id,
                    self.disable_npi,
                )
                for k in self.npi_params:
                    self.npi_params[k] = xp.array(self.npi_params[k])
                    if k == "contact_weights":
                        self.npi_params[k] = xp.broadcast_to(self.npi_params[k], (self.t_max + 1, n_nodes, 4))
                    else:
                        self.npi_params[k] = xp.broadcast_to(self.npi_params[k], (self.t_max + 1, n_nodes))
            else:
                self.npi_params = {
                    "r0_reduct": xp.broadcast_to(xp.ones(1), (self.t_max + 1, n_nodes)),
                    "contact_weights": xp.broadcast_to(xp.ones(1), (self.t_max + 1, n_nodes, 4)),
                    "mobility_reduct": xp.broadcast_to(xp.ones(1), (self.t_max + 1, n_nodes)),
                }

            self.Cij = xp.broadcast_to(self.Cij, (n_nodes,) + self.Cij.shape)
            self.npi_params["contact_weights"] = self.npi_params["contact_weights"][..., None, None]  ## TODO ellipsis NOT_DONE

            # Build adj mat for the RHS
            G = nx.convert_node_labels_to_integers(G)

            edges = xp.array(list(G.edges(data="weight"))).T

            A = sparse.coo_matrix((edges[2], (edges[0].astype(int), edges[1].astype(int))))
            A = A.tocsr()  # just b/c it will do this for almost every op on the array anyway...
            # TODO threshold low values to zero to make it even more sparse?
            if not self.sparse:
                A = A.toarray()
            A_diag = edges[2][edges[0] == edges[1]]

            A_norm = 1.0 / A.sum(axis=0)
            A_norm = xp.array(A_norm)  # bring it back to an ndarray
            if self.sparse:
                self.baseline_A = A.multiply(A_norm)
            else:
                self.baseline_A = A * A_norm
            self.baseline_A_diag = xp.squeeze(xp.multiply(A_diag, A_norm))

            self.adm0_cfr_reported = None
            self.adm1_cfr_reported = None
            self.adm2_cfr_reported = None

            if "covid_tracking_data" in G.graph:
                self.rescale_chr = True
                ct_data = G.graph["covid_tracking_data"].reset_index()
                hosp_data = ct_data.loc[ct_data.date == str(self.first_date)][["adm1", "hospitalizedCurrently"]]
                hosp_data_adm1 = hosp_data["adm1"].to_numpy()
                hosp_data_count = hosp_data["hospitalizedCurrently"].to_numpy()
                self.adm1_current_hosp = xp.zeros((self.adm1_max + 1,), dtype=float)
                self.adm1_current_hosp[hosp_data_adm1] = hosp_data_count
                if self.debug:
                    logging.debug("Current hosp: " + pformat(self.adm1_current_hosp))
                df = G.graph["covid_tracking_data"]
                self.adm1_current_cfr = xp.zeros((self.adm1_max + 1,), dtype=float)
                cfr_delay = 20  # TODO this should be calced from D_REPORT_TIME*Nij
                # TODO make a function that will take a 'floating point index' and return
                # the fractional part of the non int (we do this multiple other places while
                # reading over historical data, e.g. case_hist[-Ti:] during init)

                ## for adm1, g in df.groupby("adm1"):  ## TODO multiple_iterator
                for _iter in df.groupby("adm1"):
                    adm1 = _iter[0]
                    g = _iter[1]
                    g_df = g.reset_index().set_index("date").sort_index()
                    g_df = g_df.rolling(7).mean().dropna(how="all")
                    g_df = g_df.clip(lower=0.0)
                    g_df = g_df.rolling(7).sum()
                    new_deaths = g_df.deathIncrease.to_numpy()
                    new_cases = g_df.positiveIncrease.to_numpy()
                    new_deaths = np.clip(new_deaths, a_min=0.0, a_max=None)
                    new_cases = np.clip(new_cases, a_min=0.0, a_max=None)
                    hist_cfr = new_deaths[cfr_delay:] / new_cases[:-cfr_delay]
                    cfr = np.nanmean(hist_cfr[-7:])
                    self.adm1_current_cfr[adm1] = cfr

                if self.debug:
                    logging.debug("Current CFR: " + pformat(self.adm1_current_cfr))

            else:
                self.rescale_chr = False

            if True:
                # Hack the graph data together to get it in the same format as the covid_tracking data
                death_df = (
                    pd.DataFrame(self.inc_death_hist, columns=xp.to_cpu(self.adm1_id))
                    .stack()
                    .groupby(level=[0, 1])
                    .sum()
                    .reset_index()
                )
                death_df.columns = ["date", "adm1", "deathIncrease"]
                case_df = (
                    pd.DataFrame(self.inc_case_hist, columns=xp.to_cpu(self.adm1_id))
                    .stack()
                    .groupby(level=[0, 1])
                    .sum()
                    .reset_index()
                )
                case_df.columns = ["date", "adm1", "positiveIncrease"]

                df = (
                    death_df.set_index(["date", "adm1"])
                    .merge(case_df.set_index(["date", "adm1"]), left_index=True, right_index=True)
                    .reset_index()
                )

                self.adm1_current_cfr = xp.zeros((self.adm1_max + 1,), dtype=float)
                cfr_delay = 20

                ## for adm1, g in df.groupby("adm1"):  ## TODO multiple_iterator
                for _iter in df.groupby("adm1"):
                    adm1 = _iter[0]
                    g = _iter[1]
                    g_df = g.set_index("date").sort_index().rolling(7).mean().dropna(how="all")
                    g_df.clip(lower=0.0, inplace=True)
                    g_df = g_df.rolling(7).sum()
                    new_deaths = g_df.deathIncrease.to_numpy()
                    new_cases = g_df.positiveIncrease.to_numpy()
                    new_deaths = np.clip(new_deaths, a_min=0.0, a_max=None)
                    new_cases = np.clip(new_cases, a_min=0.0, a_max=None)
                    hist_cfr = new_deaths[cfr_delay:] / new_cases[:-cfr_delay]
                    cfr = np.nanmean(hist_cfr[-7:])
                    self.adm1_current_cfr[adm1] = cfr
                logging.debug("Current CFR: " + pformat(self.adm1_current_cfr))

            else:
                self.rescale_chr = False

        # make sure we always reset to baseline
        self.A = self.baseline_A
        self.A_diag = self.baseline_A_diag

        # randomize model params if we're doing that kind of thing
        if self.randomize:
            self.reset_A(self.consts.reroll_variance)
            self.params = self.bucky_params.generate_params(self.consts.reroll_variance)

        else:
            self.params = self.bucky_params.generate_params(None)

        if params is not None:
            self.params = copy.deepcopy(params)

        if self.debug:
            logging.debug("params: " + pformat(self.params, width=120))

        for k in self.params:
            if type(self.params[k]).__module__ == np.__name__:
                self.params[k] = xp.asarray(self.params[k])

        if self.use_vuln:
            self.params.H = (
                self.params.H[:, None] * (1 - self.vulnerability_frac)
                + self.vulnerability_factor * self.params.H[:, None] * self.vulnerability_frac
            )

            self.params.F = (
                self.params.F[:, None] * (1 - self.vulnerability_frac)
                + self.vulnerability_factor * self.params.F[:, None] * self.vulnerability_frac
            )
        else:
            self.params.H = xp.broadcast_to(self.params.H[:, None], self.Nij.shape)
            self.params.F = xp.broadcast_to(self.params.F[:, None], self.Nij.shape)

        # if False:
        #     # self.ifr[xp.isnan(self.ifr)] = 0.0
        #     # self.params.F = self.ifr / self.params["SYM_FRAC"]
        #     # adm0_ifr = xp.sum(self.ifr * self.Nij) / xp.sum(self.Nj)
        #     # ifr_scale = (
        #     #    0.0065 / adm0_ifr
        #     # )  # TODO this should be in par file (its from planning scenario5)
        #     # self.params.F = xp.clip(self.params.F * ifr_scale, 0.0, 1.0)
        #     # self.params.F_old = self.params.F.copy()
        #
        #     # TODO this needs to be cleaned up BAD
        #     # should add a util function to do the rollups to adm1 (it shows up in case_reporting/doubling t calc too)
        #     adm1_Fi = xp.zeros((self.adm1_max + 1, self.n_age_grps))
        #     xp.scatter_add(adm1_Fi, self.adm1_id, (self.params.F * self.Nij).T)
        #     adm1_Ni = xp.zeros((self.adm1_max + 1, self.n_age_grps))
        #     xp.scatter_add(adm1_Ni, self.adm1_id, self.Nij.T)
        #     adm1_Fi = adm1_Fi / adm1_Ni
        #     adm1_F = xp.mean(adm1_Fi, axis=1)
        #
        #     adm1_F_fac = self.adm1_current_cfr / adm1_F
        #     adm1_F_fac[xp.isnan(adm1_F_fac)] = 1.0
        #
        #     # F_RR_fac = truncnorm(xp, 1.0, self.consts.reroll_variance, size=adm1_F_fac.size, a_min=1e-6)
        #     adm1_F_fac = adm1_F_fac  # * F_RR_fac
        #     adm1_F_fac = xp.clip(adm1_F_fac, a_min=0.1, a_max=10.0)  # prevent extreme values
        #     if self.debug:
        #         logging.debug("adm1 cfr rescaling factor: " + pformat(adm1_F_fac))
        #     self.params.F = self.params.F * adm1_F_fac[self.adm1_id]
        #     self.params.F = xp.clip(self.params.F, a_min=1.0e-10, a_max=1.0)
        #     self.params.H = xp.clip(self.params.H, a_min=self.params.F, a_max=1.0)

        # crr_days_needed = max( #TODO this depends on all the Td params, and D_REPORT_TIME...
        case_reporting = xp.to_cpu(
            self.estimate_reporting(cfr=self.params.F, days_back=22, min_deaths=self.consts.case_reporting_min_deaths),
        )
        self.case_reporting = xp.array(
            mPERT_sample(  # TODO these facs should go in param file
                mu=xp.clip(case_reporting, a_min=0.01, a_max=1.0),
                a=xp.clip(0.8 * case_reporting, a_min=0.01, a_max=None),
                b=xp.clip(1.2 * case_reporting, a_min=None, a_max=1.0),
                gamma=500.0,
            ),
        )

        self.doubling_t = self.estimate_doubling_time_WHO(
            doubling_time_window=self.consts.doubling_t_window,
            mean_time_window=self.consts.doubling_t_N_historical_days,
        )

        if xp.any(~xp.isfinite(self.doubling_t)):
            logging.info("non finite doubling times, is there enough case data?")
            if self.debug:
                logging.debug(self.doubling_t)
                logging.debug(self.adm1_id[~xp.isfinite(self.doubling_t)])
            raise SimulationException

        if self.consts.reroll_variance > 0.0:
            ## self.doubling_t *= truncnorm(xp, 1.0, self.consts.reroll_variance, size=self.doubling_t.shape, a_min=1e-6) ## TODO
            self.doubling_t = self.doubling_t * truncnorm(xp, 1.0, self.consts.reroll_variance, size=self.doubling_t.shape, a_min=1e-6)
            self.doubling_t = xp.clip(self.doubling_t, 1.0, None) / 2.0

        self.params = self.bucky_params.rescale_doubling_rate(self.doubling_t, self.params, xp, self.A_diag)

        n_nodes = self.Nij.shape[-1]  # len(self.G.nodes())  # TODO this should be refactored out...

        mean_case_reporting = xp.mean(self.case_reporting[-self.consts.case_reporting_N_historical_days :], axis=0)

        self.params["CASE_REPORT"] = mean_case_reporting
        self.params["THETA"] = xp.broadcast_to(
            self.params["THETA"][:, None], self.Nij.shape
        )  # TODO move all the broadcast_to's to one place, they're all over reset()
        self.params["GAMMA_H"] = xp.broadcast_to(self.params["GAMMA_H"][:, None], self.Nij.shape)
        self.params["F_eff"] = xp.clip(self.consts.scaling_F * self.params["F"] / self.params["H"], 0.0, 1.0)

        # init state vector (self.y)
        yy = buckyState(self.consts, self.Nij)

        if self.debug:
            logging.debug("case init")
        Ti = self.params.Ti
        current_I = (
            xp.sum(self.inc_case_hist[-int(Ti) :], axis=0) + (Ti % 1) * self.inc_case_hist[-int(Ti + 1)]
        )  # TODO replace with util func
        current_I[xp.isnan(current_I)] = 0.0
        current_I[current_I < 0.0] = 0.0
        ## current_I *= 1.0 / (self.params["CASE_REPORT"])  ## TODO
        current_I = current_I * 1.0 / (self.params["CASE_REPORT"])

        # TODO should be in param file
        R_fac = xp.array(mPERT_sample(mu=0.25, a=0.2, b=0.3, gamma=50.0))
        E_fac = xp.array(mPERT_sample(mu=1.5, a=1.25, b=1.75, gamma=50.0))
        H_fac = xp.array(mPERT_sample(mu=1.0, a=0.9, b=1.1, gamma=100.0))

        I_init = current_I[None, :] / self.Nij / self.n_age_grps
        D_init = self.init_deaths[None, :] / self.Nij / self.n_age_grps
        recovered_init = ((self.init_cum_cases) / self.params["SYM_FRAC"] / (self.params["CASE_REPORT"])) * R_fac
        R_init = (
            (recovered_init) / self.Nij / self.n_age_grps - D_init - I_init / self.params["SYM_FRAC"]
        )  # rhi handled later

        self.params.H = self.params.H * H_fac

        # ic_frac = 1.0 / (1.0 + self.params.THETA / self.params.GAMMA_H)
        # hosp_frac = 1.0 / (1.0 + self.params.GAMMA_H / self.params.THETA)

        # print(ic_frac + hosp_frac)
        exp_frac = (
            E_fac
            * xp.ones(I_init.shape[-1])
            # * np.diag(self.A)
            # * np.sum(self.A, axis=1)
            * (self.params.R0)  # @ self.A)
            * self.params.GAMMA
            / self.params.SIGMA
        )

        yy.I = (1.0 - self.params.H) * I_init / yy.Im
        # for bucky the CASE_REPORT is low due to estimating it based on expected CFR and historical CFR
        # thus adding CASE_REPORT here might lower Ic and Rh too much
        yy.Ic = self.params.H * I_init / yy.Im  # self.params.CASE_REPORT *
        yy.Rh = (
            self.consts.scaling_F * self.params.H * I_init * self.params.GAMMA_H / self.params.THETA / yy.Rhn
        )  # self.params.CASE_REPORT *

        if self.rescale_chr:
            adm1_hosp = xp.zeros((self.adm1_max + 1,), dtype=float)
            xp.scatter_add(adm1_hosp, self.adm1_id, xp.sum(yy.H * self.Nij, axis=(0, 1)))
            adm2_hosp_frac = (self.adm1_current_hosp / adm1_hosp)[self.adm1_id]
            adm0_hosp_frac = xp.nansum(self.adm1_current_hosp) / xp.nansum(adm1_hosp)
            # print(adm0_hosp_frac)
            adm2_hosp_frac[xp.isnan(adm2_hosp_frac)] = adm0_hosp_frac
            self.params.H = xp.clip(H_fac * self.params.H * adm2_hosp_frac[None, :], self.params.F, 1.0)

            # TODO this .85 should be in param file...
            self.params["F_eff"] = xp.clip(self.params.scaling_F * self.params["F"] / self.params["H"], 0.0, 1.0)

            yy.I = (1.0 - self.params.H) * I_init / yy.Im
            # y[Ici] = ic_frac * self.params.H * I_init / (len(Ici))
            # y[Rhi] = hosp_frac * self.params.H * I_init / (Rhn)
            yy.Ic = self.params.CASE_REPORT * self.params.H * I_init / yy.Im
            yy.Rh = (
                self.params.scaling_F
                * self.params.CASE_REPORT
                * self.params.H
                * I_init
                * self.params.GAMMA_H
                / self.params.THETA
                / yy.Rhn
            )

        R_init -= xp.sum(yy.Rh, axis=0)

        yy.Ia = self.params.ASYM_FRAC / self.params.SYM_FRAC * I_init / yy.Im
        yy.E = exp_frac[None, :] * I_init / yy.En
        yy.R = R_init
        yy.D = D_init

        yy.init_S()
        # init the bin we're using to track incident cases (it's filled with cumulatives until we diff it later)
        yy.incC = self.cum_case_hist[-1][None, :] / self.Nij / self.n_age_grps
        self.y = yy

        # TODO assert this is 1. (need to take mean and around b/c fp err)
        # if xp.sum(self.y, axis=0)

        if xp.any(~xp.isfinite(self.y.state)):
            logging.info("nonfinite values in the state vector, something is wrong with init")
            ## raise SimulationException

        if self.debug:
            logging.debug("done reset()")

        # return y

    def reset_A(self, var):
        # TODO rename the R0_frac stuff...
        # new_R0_fracij = truncnorm(xp, 1.0, var, size=self.A.shape, a_min=1e-6)
        # new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
        A = self.baseline_A  # * new_R0_fracij
        A_norm = 1.0  # / new_R0_fracij.sum(axis=0)
        A_norm = xp.array(A_norm)  # Make sure we're an ndarray and not a matrix
        if self.sparse:
            self.A = A.multiply(A_norm)  # / 2. + xp.identity(self.A.shape[-1])/2.
        else:
            self.A = A * A_norm
        self.A_diag = xp.squeeze(self.baseline_A_diag * A_norm)

    # TODO this needs to be cleaned up
    def estimate_doubling_time_WHO(
            self, days_back=14, doubling_time_window=7, mean_time_window=None, min_doubling_t=1.0
    ):

        cases = xp.array(self.G.graph["data_WHO"]["#affected+infected+confirmed+total"])[-days_back:]
        cases_old = xp.array(self.G.graph["data_WHO"]["#affected+infected+confirmed+total"])[
                    -days_back - doubling_time_window: -doubling_time_window
                    ]
        adm0_doubling_t = doubling_time_window * xp.log(2.0) / xp.log(cases / cases_old)
        doubling_t = xp.repeat(adm0_doubling_t[:, None], self.cum_case_hist.shape[-1], axis=1)
        if mean_time_window is not None:
            hist_doubling_t = xp.nanmean(doubling_t[-mean_time_window:], axis=0)
        return hist_doubling_t

    # TODO these rollups to higher adm levels should be a util (it might make sense as a decorator)
    # it shows up here, the CRR, the CHR rescaling, and in postprocess...

    def estimate_doubling_time(
            self,
            days_back=7,  # TODO rename, its the number days calc the rolling Td
            doubling_time_window=7,
            mean_time_window=None,
            min_doubling_t=1.0,
    ):
        if mean_time_window is not None:
            days_back = mean_time_window

        cases = self.cum_case_hist[-days_back:] / self.case_reporting[-days_back:]
        cases_old = (
                self.cum_case_hist[-days_back - doubling_time_window: -doubling_time_window]
                / self.case_reporting[-days_back - doubling_time_window: -doubling_time_window]
        )

        # adm0
        adm0_doubling_t = doubling_time_window / xp.log2(xp.nansum(cases, axis=1) / xp.nansum(cases_old, axis=1))

        if self.debug:
            logging.debug("Adm0 doubling time: " + str(adm0_doubling_t))
        if xp.any(~xp.isfinite(adm0_doubling_t)):
            if self.debug:
                logging.debug(xp.nansum(cases, axis=1))
                logging.debug(xp.nansum(cases_old, axis=1))
            ## raise SimulationException

        doubling_t = xp.repeat(adm0_doubling_t[:, None], cases.shape[-1], axis=1)

        # adm1
        cases_adm1 = xp.zeros((self.adm1_max + 1, days_back), dtype=float)
        cases_old_adm1 = xp.zeros((self.adm1_max + 1, days_back), dtype=float)

        xp.scatter_add(cases_adm1, self.adm1_id, cases.T)
        xp.scatter_add(cases_old_adm1, self.adm1_id, cases_old.T)

        adm1_doubling_t = doubling_time_window / xp.log2(cases_adm1 / cases_old_adm1)

        tmp_doubling_t = adm1_doubling_t[self.adm1_id].T
        valid_mask = xp.isfinite(tmp_doubling_t) & (tmp_doubling_t > min_doubling_t)

        doubling_t[valid_mask] = tmp_doubling_t[valid_mask]

        # adm2
        adm2_doubling_t = doubling_time_window / xp.log2(cases / cases_old)

        valid_adm2_dt = xp.isfinite(adm2_doubling_t) & (adm2_doubling_t > min_doubling_t)
        doubling_t[valid_adm2_dt] = adm2_doubling_t[valid_adm2_dt]

        # hist_weights = xp.arange(1., days_back + 1.0, 1.0)
        # hist_doubling_t = xp.sum(doubling_t * hist_weights[:, None], axis=0) / xp.sum(
        #    hist_weights
        # )

        # Take mean of most recent values
        if mean_time_window is not None:
            ret = xp.nanmean(doubling_t[-mean_time_window:], axis=0)
        else:
            ret = doubling_t

        return ret

    def estimate_reporting(self, cfr, days_back=14, case_lag=None, min_deaths=100.0):

        if case_lag is None:
            adm0_cfr_by_age = xp.sum(cfr * self.Nij, axis=1) / xp.sum(self.Nj, axis=0)
            adm0_cfr_total = xp.sum(
                xp.sum(cfr * self.Nij, axis=1) / xp.sum(self.Nj, axis=0),
                axis=0,
            )
            case_lag = xp.sum(self.params["D_REPORT_TIME"] * adm0_cfr_by_age / adm0_cfr_total, axis=0)

        case_lag_int = int(case_lag)
        case_lag_frac = case_lag % 1  # TODO replace with util function for the indexing
        recent_cum_cases = self.cum_case_hist - self.cum_case_hist[-90]
        recent_cum_deaths = self.cum_death_hist - self.cum_death_hist[-90]
        cases_lagged = (
            recent_cum_cases[-case_lag_int - days_back : -case_lag_int]
            + case_lag_frac * recent_cum_cases[-case_lag_int - 1 - days_back : -case_lag_int - 1]
        )

        # adm0
        adm0_cfr_param = xp.sum(xp.sum(cfr * self.Nij, axis=1) / xp.sum(self.Nj, axis=0), axis=0)
        if self.adm0_cfr_reported is None:
            self.adm0_cfr_reported = xp.sum(recent_cum_deaths[-days_back:], axis=1) / xp.sum(cases_lagged, axis=1)
        adm0_case_report = adm0_cfr_param / self.adm0_cfr_reported

        if self.debug:
            logging.debug("Adm0 case reporting rate: " + pformat(adm0_case_report))
        if xp.any(~xp.isfinite(adm0_case_report)):
            if self.debug:
                logging.debug("adm0 case report not finite")
                logging.debug(adm0_cfr_param)
                logging.debug(self.adm0_cfr_reported)
            ## raise SimulationException

        case_report = xp.repeat(adm0_case_report[:, None], cases_lagged.shape[-1], axis=1)

        # adm1
        adm1_cfr_param = xp.zeros((self.adm1_max + 1,), dtype=float)
        adm1_totpop = xp.zeros((self.adm1_max + 1,), dtype=float)

        tmp_adm1_cfr = xp.sum(cfr * self.Nij, axis=0)

        xp.scatter_add(adm1_cfr_param, self.adm1_id, tmp_adm1_cfr)
        xp.scatter_add(adm1_totpop, self.adm1_id, self.Nj)
        adm1_cfr_param /= adm1_totpop

        # adm1_cfr_reported is const, only calc it once and cache it
        if self.adm1_cfr_reported is None:
            self.adm1_deaths_reported = xp.zeros((self.adm1_max + 1, days_back), dtype=float)
            adm1_lagged_cases = xp.zeros((self.adm1_max + 1, days_back), dtype=float)

            xp.scatter_add(
                self.adm1_deaths_reported,
                self.adm1_id,
                recent_cum_deaths[-days_back:].T,
            )
            xp.scatter_add(adm1_lagged_cases, self.adm1_id, cases_lagged.T)

            self.adm1_cfr_reported = self.adm1_deaths_reported / adm1_lagged_cases

        adm1_case_report = (adm1_cfr_param[:, None] / self.adm1_cfr_reported)[self.adm1_id].T

        valid_mask = (self.adm1_deaths_reported > min_deaths)[self.adm1_id].T & xp.isfinite(adm1_case_report)
        case_report[valid_mask] = adm1_case_report[valid_mask]

        # adm2
        adm2_cfr_param = xp.sum(cfr * (self.Nij / self.Nj), axis=0)

        if self.adm2_cfr_reported is None:
            self.adm2_cfr_reported = recent_cum_deaths[-days_back:] / cases_lagged
        adm2_case_report = adm2_cfr_param / self.adm2_cfr_reported

        valid_adm2_cr = xp.isfinite(adm2_case_report) & (recent_cum_deaths[-days_back:] > min_deaths)
        case_report[valid_adm2_cr] = adm2_case_report[valid_adm2_cr]

        return case_report

    ## @staticmethod  ## TODO
    def RHS_func(self, t, y_flat, Nij, contact_mats, Aij, par, npi, aij_sparse, y):
        # constraint on values
        lower, upper = (0.0, 1.0)  # bounds for state vars  ## TODO multiple_value_asignment

        # grab index of OOB values so we can zero derivatives (stability...)
        too_low = y_flat <= lower
        too_high = y_flat >= upper

        # TODO we're passing in y.state just to overwrite it, we probably need another class
        # reshape to the usual state tensor (compartment, age, node)
        y.state = y_flat.reshape(y.state_shape)

        # Clip state to be in bounds (except allocs b/c thats a counter)
        xp.clip(y.state, a_min=lower, a_max=upper, out=y.state)

        # init d(state)/dt
        dy = buckyState(y.consts, Nij)  # TODO make a pseudo copy operator w/ zeros

        # effective params after damping w/ allocated stuff
        t_index = min(int(t), npi["r0_reduct"].shape[0] - 1)  # prevent OOB error when integrator overshoots
        BETA_eff = npi["r0_reduct"][t_index] * par["BETA"]
        F_eff = par["F_eff"]
        HOSP = par["H"]
        THETA = y.Rhn * par["THETA"]
        GAMMA = y.Im * par["GAMMA"]
        GAMMA_H = y.Im * par["GAMMA_H"]
        SIGMA = y.En * par["SIGMA"]
        SYM_FRAC = par["SYM_FRAC"]
        # ASYM_FRAC = par["ASYM_FRAC"]
        CASE_REPORT = par["CASE_REPORT"]

        Cij = npi["contact_weights"][t_index] * contact_mats
        Cij = xp.sum(Cij, axis=1)
        Cij /= xp.sum(Cij, axis=2, keepdims=True)

        if aij_sparse:
            Aij_eff = Aij.multiply(npi["mobility_reduct"][t_index])
        else:
            Aij_eff = npi["mobility_reduct"][t_index] * Aij

        # perturb Aij
        # new_R0_fracij = truncnorm(xp, 1.0, .1, size=Aij.shape, a_min=1e-6)
        # new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
        # A = Aij * new_R0_fracij
        # Aij_eff = A / xp.sum(A, axis=0)

        # Infectivity matrix (I made this name up, idk what its really called)
        I_tot = xp.sum(Nij * y.Itot, axis=0) - (1.0 - par["rel_inf_asym"]) * xp.sum(Nij * y.Ia, axis=0)

        # I_tmp = (Aij.T @ I_tot.T).T
        if aij_sparse:
            I_tmp = (Aij_eff.T * I_tot.T).T
        else:
            I_tmp = I_tot @ Aij  # using identity (A@B).T = B.T @ A.T

        beta_mat = y.S * xp.squeeze((Cij @ I_tmp.T[..., None]), axis=-1).T  ## TODO ellipsis NOT_DONE
        ## beta_mat /= Nij  ## TODO
        beta_mat = beta_mat / Nij

        # dS/dt
        dy.S = -BETA_eff * (beta_mat)
        # dE/dt
        dy.E[0] = BETA_eff * (beta_mat) - SIGMA * y.E[0]
        dy.E[1:] = SIGMA * (y.E[:-1] - y.E[1:])

        # dI/dt
        dy.Ia[0] = (1.0 - SYM_FRAC) * SIGMA * y.E[-1] - GAMMA * y.Ia[0]
        dy.Ia[1:] = GAMMA * (y.Ia[:-1] - y.Ia[1:])

        # dIa/dt
        dy.I[0] = SYM_FRAC * (1.0 - HOSP) * SIGMA * y.E[-1] - GAMMA * y.I[0]
        dy.I[1:] = GAMMA * (y.I[:-1] - y.I[1:])

        # dIc/dt
        dy.Ic[0] = SYM_FRAC * HOSP * SIGMA * y.E[-1] - GAMMA_H * y.Ic[0]
        dy.Ic[1:] = GAMMA_H * (y.Ic[:-1] - y.Ic[1:])

        # dRhi/dt
        dy.Rh[0] = GAMMA_H * y.Ic[-1] - THETA * y.Rh[0]
        dy.Rh[1:] = THETA * (y.Rh[:-1] - y.Rh[1:])

        # dR/dt
        dy.R = GAMMA * (y.I[-1] + y.Ia[-1]) + (1.0 - F_eff) * THETA * y.Rh[-1]

        # dD/dt
        dy.D = F_eff * THETA * y.Rh[-1]

        dy.incH = SYM_FRAC * CASE_REPORT * HOSP * SIGMA * y.E[-1]
        dy.incC = SYM_FRAC * CASE_REPORT * SIGMA * y.E[-1]

        # bring back to 1d for the ODE api
        dy_flat = dy.state.ravel()

        # zero derivatives for things we had to clip if they are going further out of bounds
        dy_flat = xp.where(too_low & (dy_flat < 0.0), 0.0, dy_flat)
        dy_flat = xp.where(too_high & (dy_flat > 0.0), 0.0, dy_flat)

        return dy_flat

    def run_once(self, seed=None, outdir="raw_output/", output=True, output_queue=None):

        # reset everything
        logging.debug("Resetting state")
        self.reset(seed=seed)
        logging.debug("Done reset")

        # TODO should output the IC here

        # do integration
        logging.debug("Starting integration")
        t_eval = np.arange(0, self.t_max + self.dt, self.dt)
        sol = ivp.solve_ivp(
            self.RHS_func,
            method="RK23",
            t_span=(0.0, self.t_max),
            y0=self.y.state.ravel(),
            t_eval=t_eval,
            args=(self.Nij, self.Cij, self.A, self.params, self.npi_params, self.sparse, self.y),
        )
        logging.debug("Done integration")

        y = sol.y.reshape(self.y.state_shape + (len(t_eval),))

        out = buckyState(self.consts, self.Nij)
        out.state = self.Nij[None, ..., None] * y  ## TODO ellipsis NOT_DONE

        # collapse age groups
        out.state = xp.sum(out.state, axis=1)

        population_conserved = (xp.diff(xp.around(xp.sum(out.N, axis=(0, 1)), 1)) == 0.0).all()
        if not population_conserved:
            pass  # TODO we're getting small fp errors here
            # print(xp.sum(xp.diff(xp.around(xp.sum(out[:incH], axis=(0, 1)), 1))))
            # logging.error("Population not conserved!")
            # print(xp.sum(xp.sum(y[:incH],axis=0)-1.))
            # raise SimulationException

        adm2_ids = np.broadcast_to(self.adm2_id[:, None], out.state.shape[1:])

        if self.output_dates is None:
            t_output = xp.to_cpu(sol.t)

            ## dates = [pd.Timestamp(self.first_date + datetime.timedelta(days=np.round(t))) for t in t_output]  ## TODO comprehension_list
            _list = list()
            for t in t_output:
                _list.append(pd.Timestamp(self.first_date + datetime.timedelta(days=np.round(t))))
            dates = _list

            self.output_dates = np.broadcast_to(dates, out.state.shape[1:])

        dates = self.output_dates

        icu = self.Nij[..., None] * self.params["ICU_FRAC"][:, None, None] * xp.sum(y[out.indices["H"]], axis=0)  ## TODO ellipsis NOT_DONE
        vent = self.params.ICU_VENT_FRAC[:, None, None] * icu

        # prepend the min cumulative cases over the last 2 days in case in the decreased
        prepend_deaths = xp.minimum(self.cum_death_hist[-2], self.cum_death_hist[-1])
        daily_deaths = xp.diff(out.D, prepend=prepend_deaths[:, None], axis=-1)

        init_inc_death_mean = xp.mean(xp.sum(daily_deaths[:, 1:4], axis=0))
        hist_inc_death_mean = xp.mean(xp.sum(self.inc_death_hist[-7:], axis=-1))

        inc_death_rejection_fac = 2.0  # TODO These should come from the cli arg -r
        if (
                (init_inc_death_mean > inc_death_rejection_fac * hist_inc_death_mean)
                or (inc_death_rejection_fac * init_inc_death_mean < hist_inc_death_mean)
        ) and self.reject_runs:
            logging.info("Inconsistent inc deaths, rejecting run")
            ## raise SimulationException

        # prepend the min cumulative cases over the last 2 days in case in the decreased
        prepend_cases = xp.minimum(self.cum_case_hist[-2], self.cum_case_hist[-1])
        daily_cases_reported = xp.diff(out.incC, axis=-1, prepend=prepend_cases[:, None])
        cum_cases_reported = out.incC

        init_inc_case_mean = xp.mean(xp.sum(daily_cases_reported[:, 1:4], axis=0))
        hist_inc_case_mean = xp.mean(xp.sum(self.inc_case_hist[-7:], axis=-1))

        inc_case_rejection_fac = 1.5  # TODO These should come from the cli arg -r
        if (
                (init_inc_case_mean > inc_case_rejection_fac * hist_inc_case_mean)
                or (inc_case_rejection_fac * init_inc_case_mean < hist_inc_case_mean)
        ) and self.reject_runs:
            logging.info("Inconsistent inc cases, rejecting run")
            ## raise SimulationException

        daily_cases_total = daily_cases_reported / self.params.CASE_REPORT[:, None]
        cum_cases_total = cum_cases_reported / self.params.CASE_REPORT[:, None]

        out.incH[:, 0] = out.incH[:, 1]
        daily_hosp = xp.diff(out.incH, axis=-1, prepend=out.incH[:, 0][..., None])  ## TODO ellipsis NOT_DONE
        # if (daily_cases < 0)[..., 1:].any():
        #    logging.error('Negative daily cases')
        #    raise SimulationException
        N = xp.broadcast_to(self.Nj[..., None], out.state.shape[1:])  ## TODO ellipsis NOT_DONE

        hosps = xp.sum(out.Ic, axis=0) + xp.sum(out.Rh, axis=0)  # why not just using .H?

        out.state = out.state.reshape(y.shape[0], -1)

        # Grab pretty much everything interesting
        df_data = {
            "adm2_id": adm2_ids.ravel(),
            "date": dates.ravel(),
            "rid": np.broadcast_to(seed, out.state.shape[-1]).ravel(),
            "total_population": N.ravel(),
            "current_hospitalizations": hosps.ravel(),
            "active_asymptomatic_cases": out.Ia,  # TODO remove?
            "cumulative_deaths": out.D,
            "daily_hospitalizations": daily_hosp.ravel(),
            "daily_cases": daily_cases_total.ravel(),
            "daily_reported_cases": daily_cases_reported.ravel(),
            "daily_deaths": daily_deaths.ravel(),
            "cumulative_cases": cum_cases_total.ravel(),
            "cumulative_reported_cases": cum_cases_reported.ravel(),
            "current_icu_usage": xp.sum(icu, axis=0).ravel(),
            "current_vent_usage": xp.sum(vent, axis=0).ravel(),
            "case_reporting_rate": np.broadcast_to(self.params.CASE_REPORT[:, None], adm2_ids.shape).ravel(),
            "R_eff": (
                    self.npi_params["r0_reduct"].T
                    * np.broadcast_to((self.params.R0 * self.A_diag)[:, None], adm2_ids.shape)
            ).ravel(),
            "doubling_t": np.broadcast_to(self.doubling_t[:, None], adm2_ids.shape).ravel(),
        }

        # Collapse the gamma-distributed compartments and move everything to cpu
        negative_values = False
        for k in df_data:
            if df_data[k].ndim == 2:
                df_data[k] = xp.sum(df_data[k], axis=0)

            # df_data[k] = xp.to_cpu(df_data[k])

            if k != "date" and xp.any(xp.around(df_data[k], 2) < 0.0):
                logging.info("Negative values present in " + k)
                negative_values = True

        if negative_values and self.reject_runs:
            logging.info("Rejecting run b/c of negative values in output")
            ## raise SimulationException

        # Append data to the hdf5 file
        output_folder = os.path.join(outdir, self.run_id)

        if output:
            os.makedirs(output_folder, exist_ok=True)
            output_queue.put((os.path.join(output_folder, str(seed)), df_data))
        # TODO we should output the per monte carlo param rolls, this got lost when we switched from hdf5


# -----------------------------------------------------------------------------
# SEIR_covid
# -----------------------------------------------------------------------------

def main(args=None):

    parser = get_arg_parser()

    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args=args)

    ## suppressing for bucky_simplified_v1
    # if args.gpu:
    #     use_cupy(optimize=args.opt)

    ## global xp, ivp, sparse  # pylint: disable=global-variable-not-assigned
    ## from .numerical_libs import xp, ivp, sparse  # noqa: E402  # pylint: disable=import-outside-toplevel  # isort:skip

    warnings.simplefilter(action="ignore", category=xp.ExperimentalWarning)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    loglevel = 30 - 10 * min(args.verbosity, 2)
    runid = get_runid()
    if not os.path.exists(args.output_dir + "/" + runid):
        os.mkdir(args.output_dir + "/" + runid)
    fh = logging.FileHandler(args.output_dir + "/" + runid + "/stdout")
    fh.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[TqdmLoggingHandler()],
    )
    debug_mode = loglevel < 20

    # TODO we should output the logs to output_dir too...
    _banner()

    to_write = queue.Queue(maxsize=100)

    def writer():
        # Call to_write.get() until it returns None
        stream = xp.cuda.Stream() if args.gpu else None


        ## for base_fname, df_data in iter(to_write.get, None):  ## TODO multiple_iterator
        for _iter in iter(to_write.get, None):
            base_fname = _iter[0]
            df_data = _iter[1]

            ## cpu_data = {k: xp.to_cpu(v, stream=stream) for k, v in df_data.items()}  ## comprehension_dict , TODO multiple_iterator
            _dict = dict()
            for _iter in df_data.items():
                k = _iter[0]
                v = _iter[1]
                _dict[k] = xp.to_cpu(v, stream=stream)
            cpu_data = _dict

            if stream is not None:
                stream.synchronize()
            df = pd.DataFrame(cpu_data)
            for date, date_df in df.groupby("date", as_index=False):
                fname = base_fname + "_" + str(date.date()) + ".feather"
                date_df.reset_index().to_feather(fname)

    write_thread = threading.Thread(target=writer, daemon=True)
    write_thread.start()

    if args.gpu:
        logging.info("Using GPU backend")

    logging.info(f"command line args: {args}")
    if args.no_mc:  # TODO can we just remove this already?
        ## raise NotImplementedError  ## TODO exception_handling
        pass
        # env = SEIR_covid(randomize_params_on_reset=False)
        # n_mc = 1
    else:
        env = SEIR_covid(
            randomize_params_on_reset=True,
            debug=debug_mode,
            sparse_aij=(not args.dense),
            t_max=args.days,
            graph_file=args.graph_file,
            par_file=args.par_file,
            npi_file=args.npi_file,
            disable_npi=args.disable_npi,
            reject_runs=args.reject_runs,
        )
        n_mc = args.n_mc

    seed_seq = np.random.SeedSequence(args.seed)

    total_start = datetime.datetime.now()
    success = 0
    n_runs = 0
    times = []
    pbar = tqdm.tqdm(total=n_mc, desc="Performing Monte Carlos", dynamic_ncols=True)

    ## replacement:  ## TODO
    while success < n_mc:
        start_time = datetime.datetime.now()
        mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]  # inc spawn key then grab next seed
        pbar.set_postfix_str(
            "seed="
            + str(mc_seed)
            + ", rej%="  # TODO disable rej% if not -r
            + str(np.around(float(n_runs - success) / (n_runs + 0.00001) * 100, 1)),
            refresh=True,
        )
        n_runs += 1

        env.run_once(seed=mc_seed, outdir=args.output_dir, output_queue=to_write)

        success += 1
        pbar.update(1)
        run_time = (datetime.datetime.now() - start_time).total_seconds()
        times.append(run_time)

        logging.info(f"{mc_seed}: {datetime.datetime.now() - start_time}")

    to_write.put(None)
    write_thread.join()
    pbar.close()
    logging.info(f"Total runtime: {datetime.datetime.now() - total_start}")

    ## replaced:  ## TODO exception_handling
    # try:
    #     while success < n_mc:
    #         start_time = datetime.datetime.now()
    #         mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]  # inc spawn key then grab next seed
    #         pbar.set_postfix_str(
    #             "seed="
    #             + str(mc_seed)
    #             + ", rej%="  # TODO disable rej% if not -r
    #             + str(np.around(float(n_runs - success) / (n_runs + 0.00001) * 100, 1)),
    #             refresh=True,
    #         )
    #         try:
    #             n_runs += 1
    #             with xp.optimize_kernels():
    #                 env.run_once(seed=mc_seed, outdir=args.output_dir, output_queue=to_write)
    #             success += 1
    #             pbar.update(1)
    #         except SimulationException:
    #             pass
    #         run_time = (datetime.datetime.now() - start_time).total_seconds()
    #         times.append(run_time)
    #
    #         logging.info(f"{mc_seed}: {datetime.datetime.now() - start_time}")
    # except (KeyboardInterrupt, SystemExit):
    #     logging.warning("Caught SIGINT, cleaning up")
    #     to_write.put(None)
    #     write_thread.join()
    # finally:
    #     to_write.put(None)
    #     write_thread.join()
    #     pbar.close()
    #     logging.info(f"Total runtime: {datetime.datetime.now() - total_start}")


if __name__ == "__main__":
    main()
