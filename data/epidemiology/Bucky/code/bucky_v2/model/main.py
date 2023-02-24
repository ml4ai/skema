"""The main module handling the simulation."""
import datetime
import random

import numpy as np
import tqdm
from loguru import logger
###CTM from tqdm.contrib.logging import logging_redirect_tqdm

###CTM from ..numerical_libs import enable_cupy, reimport_numerical_libs, xp, xp_ivp
###CTM_START
from ..numerical_libs import xp, xp_ivp
###CTM_END
from ..util.distributions import approx_mPERT
from ..util.fractional_slice import frac_last_n_vals
from ..util.util import _banner
from .data import buckyData
from .derived_epi_params import add_derived_params
from .estimation import estimate_cfr, estimate_chr, estimate_crr, estimate_Rt
from .exceptions import SimulationException
from .io import BuckyOutputWriter
from .mc_instance import buckyMCInstance
from .npi import get_npi_params
from .rhs import RHS_func
from .state import buckyState
from .vacc import buckyVaccAlloc

###CTM_START
from ..config import BuckyConfig
###CTM_END

# TODO rename g_data (needs a better name than 'data' though...)


class buckyModelCovid:
    """Class that handles one full simulation (both time integration and managing MC states)."""

    def __init__(
        self,
        cfg,
        npi_file=None,
        disable_npi=False,
        reject_runs=False,
        output_dir=None,
    ):
        """Initialize the class, do some bookkeeping and read in the input graph."""
        self.cfg = cfg
        self.flags = cfg["model.flags"]

        self.debug = cfg["runtime.debug"]

        # Integrator params
        self.t_max = cfg["runtime.t_max"]
        self.run_id = cfg["runtime.run_id"]
        logger.info("Run ID: {}", self.run_id)

        self.npi_file = npi_file
        self.disable_npi = disable_npi
        self.reject_runs = reject_runs

        if cfg["runtime.start_date"] is not None:
            start_date = datetime.datetime.strptime(cfg["runtime.start_date"], "%Y-%m-%d")
        else:
            start_date = None

        self.g_data = self.load_data(
            data_dir=cfg["system.data_dir"],
            fit_cfg=cfg["model.fitting"],
            force_diag_Aij=cfg["model.flags.identity_Aij"],
            force_start_date=start_date,
        )

        self.writer = BuckyOutputWriter(cfg["system.raw_output_dir"], self.run_id)
        self.writer.write_metadata(
            self.g_data.adm_mapping,
            self.projected_dates,
            {"csse_fitted_timeseries": self.g_data.csse_data, "hhs_fitted_timeseries": self.g_data.hhs_data},
        )

    '''
    def update_params(self, update_dict):
        """Update the params based of a dict of new values."""
        self.bucky_params.update_params(update_dict)
        self.consts = self.bucky_params.consts
    '''

    def load_data(self, data_dir, fit_cfg, force_diag_Aij, force_start_date):
        """Load the historical data and calculate all the variables that are static across MC runs."""
        # TODO refactor to just have this return g_data?

        # Load data from input files
        # TODO we should go through an replace lots of math using self.g_data.* with function IN buckyData
        # TODO rename g_data
        g_data = buckyData(
            data_dir=data_dir,
            fit_cfg=fit_cfg,
            force_diag_Aij=force_diag_Aij,
            force_start_date=force_start_date,
        )

        self.sim_start_date = g_data.csse_data.end_date
        self.projected_dates = [
            str(self.sim_start_date + datetime.timedelta(days=int(np.round(t)))) for t in range(self.t_max + 1)
        ]

        # Load and stack contact matrices
        self.contact_mats = g_data.Cij
        # remove all_locations so we can sum over the them ourselves
        if "all" in self.contact_mats:
            del self.contact_mats["all"]

        # Remove unknown contact mats
        valid_contact_mats = ["home", "work", "others", "school"]
        self.contact_mats = {k: v for k, v in self.contact_mats.items() if k in valid_contact_mats}

        self.Cij = xp.vstack([self.contact_mats[k][None, ...] for k in sorted(self.contact_mats)])

        # Pull some frequently used vars into this scope (stratified population, etc)
        self.Nij = g_data.Nij
        self.Nj = g_data.Nj

        self.base_mc_instance = buckyMCInstance(self.sim_start_date, self.t_max, self.Nij, self.Cij)

        # fill in npi_params either from file or as ones
        self.npi_params = get_npi_params(g_data, self.sim_start_date, self.t_max, self.npi_file, self.disable_npi)

        if self.npi_params["npi_active"]:
            self.base_mc_instance.add_npi(self.npi_params)

        if self.flags["vaccines"]:
            self.vacc_data = buckyVaccAlloc(g_data, self.cfg, self.sim_start_date)
            self.base_mc_instance.add_vacc(self.vacc_data)
        return g_data

    # TODO static?
    ###CTM @staticmethod
    def calc_lagged_rate(var1, var2, lag, mean_days, rollup_func=None):  # pylint: disable=unused-argument
        """WIP."""

        var1_lagged = frac_last_n_vals(var1, mean_days, axis=0, offset=lag)
        var1_lagged = var1_lagged - frac_last_n_vals(var1, mean_days, axis=0, offset=lag + mean_days + 1)
        var1_var2_ratio = var2 / var1_lagged
        ret = xp.mean(var1_var2_ratio, axis=0)

        # harmonic mean:
        # ret = 1.0 / xp.nanmean(1.0 / var1_var2_ratio, axis=0)

        return ret

    ###CTM @staticmethod
    def set_seed(seed=None):
        """Seed all the relevent PRNGS."""
        # move to util?
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(seed)
            if xp.is_cupy:
                xp.random.seed(seed)

    def reset(self):
        """Reset the state of the model and generate new inital data from a new random seed."""
        # TODO we should refactor reset of the compartments to be real pop numbers then /Nij at the end

        # reroll model params
        sampled_params = self.cfg["model"].sample_distributions()
        vac_params = sampled_params["vaccine"]
        epi_params = sampled_params["epi"]
        mc_params = sampled_params["monte_carlo"]._to_arrays()

        if mc_params["Aij_gaussian_perturbation_scale"] > 0.0:
            self.g_data.Aij.perturb(mc_params["Aij_gaussian_perturbation_scale"])

        epi_params = add_derived_params(epi_params, self.cfg["model.structure"])

        # Reject some invalid param combinations
        if (
            (mc_params["Te_min"] > epi_params["Te"])
            or (mc_params["Ti_min"] > epi_params["Ti"])
            or (epi_params["Te"] > epi_params["Tg"])
        ):
            raise SimulationException

        # Reroll vaccine allocation
        if self.base_mc_instance.vacc_data.reroll:
            self.base_mc_instance.vacc_data.reroll_distribution()
            self.base_mc_instance.vacc_data.reroll_doses()
            epi_params["vacc_eff_1"] = vac_params["vacc_eff_1"]
            epi_params["vacc_eff_2"] = vac_params["vacc_eff_2"]

        # TODO move most of below into a function like:
        # test = calc_initial_state(self.g_data, self.params, self.base_mc_instance)

        # Estimate the current age distribution of S, S_age_dist
        nonvaccs = 1.0
        if self.base_mc_instance.vacc_data is not None:
            nonvaccs = xp.clip(
                1 - self.base_mc_instance.vacc_data.V_tot(vac_params, 0) * mc_params["R_fac"],
                a_min=0,
                a_max=1,
            )
        else:
            nonvaccs = 1.0
        tmp = nonvaccs * self.g_data.Nij / self.g_data.Nj
        S_age_dist = tmp / xp.sum(tmp, axis=0)

        # estimate IFR for our age bins
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7721859/
        mean_ages = xp.mean(xp.array(sampled_params["structure.age_bins"]), axis=1)
        ifr = xp.exp(-7.56 + 0.121 * mean_ages) / 100.0
        ifr = ifr  # * epi_params["HR_vs_wildtype"]

        # Estimate the case reporting rate
        # crr_days_needed = max( #TODO this depends on all the Td params, and D_REPORT_TIME...
        case_reporting = estimate_crr(
            self.g_data,
            case_to_death_lag=epi_params["CASE_TO_DEATH_TIME"],
            ifr=ifr[..., None],  # self.params.F,
            # case_lag=14,
            days_back=25,
            # min_deaths=self.consts.case_reporting_min_deaths,
            S_dist=nonvaccs,  # S_age_dist * 16.0,
        )

        self.case_reporting = approx_mPERT(  # TODO these facs should go in param file
            mu=xp.clip(case_reporting, a_min=0.05, a_max=0.95),
            a=xp.clip(0.7 * case_reporting, a_min=0.01, a_max=0.9),
            b=xp.clip(1.3 * case_reporting, a_min=0.1, a_max=1.0),
            gamma=50.0,
        )

        # TODO need case reporting in cfg, move all the CRR calc stuff to its own func
        case_reporting_N_historical_days = 14
        mean_case_reporting = xp.nanmean(self.case_reporting[-case_reporting_N_historical_days:], axis=0)

        # Fill in and correct the shapes of some parameters
        # TODO make a broadcast_to func in the cfg

        epi_params["CFR"] = ifr[..., None] * mean_case_reporting[None, ...]
        epi_params["CHR"] = xp.broadcast_to(epi_params["CHR"][:, None], self.Nij.shape)
        epi_params["CRR"] = mean_case_reporting
        epi_params["THETA"] = xp.broadcast_to(
            epi_params["THETA"][:, None],
            self.Nij.shape,
        )  # TODO move all the broadcast_to's to one place, they're all over reset()
        epi_params["GAMMA_H"] = xp.broadcast_to(epi_params["GAMMA_H"][:, None], self.Nij.shape)

        epi_params["overall_adm2_ifr"] = xp.sum(ifr[:, None] * self.g_data.Nij / self.g_data.Nj, axis=0)

        # Build init state vector (self.y)
        yy = buckyState(self.cfg["model.structure"], self.Nij)

        # Ti = self.params.Ti
        current_I = xp.sum(frac_last_n_vals(self.g_data.csse_data.incident_cases, epi_params["Ti"], axis=0), axis=0)

        current_I[xp.isnan(current_I)] = 0.0
        current_I[current_I < 0.0] = 0.0
        current_I *= 1.0 / (epi_params["CRR"])

        # Roll some random factors for the init compartment values
        # TODO move these inline
        R_fac = mc_params["R_fac"]
        E_fac = mc_params["E_fac"]
        H_fac = mc_params["H_fac"]
        Rt_fac = mc_params["Rt_fac"]
        F_fac = mc_params["F_fac"]
        # TODO add an mPERT F_fac instead of the truncnorm

        age_dist_fac = self.g_data.Nij / self.g_data.Nj[None, ...]
        I_init = E_fac * current_I[None, :] * S_age_dist / self.Nij
        D_init = self.g_data.csse_data.cumulative_deaths[-1][None, :] * age_dist_fac / self.Nij
        recovered_init = (self.g_data.csse_data.cumulative_cases[-1] / epi_params["SYM_FRAC"]) * R_fac
        R_init = (
            (recovered_init) * age_dist_fac / self.Nij - D_init - I_init / epi_params["SYM_FRAC"]
        )  # Rh is factored in later

        Rt = estimate_Rt(
            self.g_data,
            generation_interval=epi_params["Tg"],
            E_gamma_k=yy.E_gamma_k,
            days_back=7,
            case_reporting=self.case_reporting,
        )
        Rt = Rt * Rt_fac

        epi_params["Rt"] = Rt
        epi_params["BETA"] = Rt * epi_params["GAMMA"] / self.g_data.Aij.diag

        exp_frac = (
            E_fac
            * xp.ones(I_init.shape[-1])
            * epi_params["Rt"]
            * epi_params["GAMMA"]
            / epi_params["SIGMA"]
            / (1.0 - R_init)
            / epi_params["SYM_FRAC"]
        )

        epi_params["CHR"] = estimate_chr(
            self.g_data,
            base_CHR=epi_params["CHR"],
            I_to_H_time=epi_params["I_TO_H_TIME"],
            Rh_gamma_k=yy.Rh_gamma_k,
            S_age_dist=S_age_dist,
            days_back=14,
        )
        yy.I = (1.0 - epi_params["CHR"] * epi_params["CRR"]) * I_init / yy.I_gamma_k  # noqa: E741
        yy.Ic = epi_params["CHR"] * I_init / yy.I_gamma_k * epi_params["CRR"]
        yy.Rh = epi_params["CHR"] * I_init / yy.Rh_gamma_k * epi_params["CRR"]

        if self.cfg["model.flags.rescale_chr"]:
            adm1_hosp = self.g_data.sum_adm1(xp.sum(yy.Rh * self.Nij, axis=(0, 1)))
            adm2_hosp_frac = (self.g_data.hhs_data.current_hospitalizations[-1] / adm1_hosp)[self.g_data.adm1_id]
            adm0_hosp_frac = xp.nansum(self.g_data.hhs_data.current_hospitalizations[-1]) / xp.nansum(adm1_hosp)
            adm2_hosp_frac[~xp.isfinite(adm2_hosp_frac) | (adm2_hosp_frac == 0.0)] = adm0_hosp_frac

            # adm2_hosp_frac = xp.sqrt(adm2_hosp_frac * adm0_hosp_frac)

            scaling_H = adm2_hosp_frac * H_fac  # * self.consts.F_scaling
            F_RR_fac = xp.broadcast_to(F_fac, (adm1_hosp.size,))  # /scaling_H
            epi_params["CFR"] = estimate_cfr(
                self.g_data,
                base_CFR=epi_params["CFR"],
                case_to_death_time=epi_params["CASE_TO_DEATH_TIME"],
                Rh_gamma_k=yy.Rh_gamma_k,
                S_age_dist=S_age_dist,
                days_back=14,
            )
            epi_params["CFR"] = xp.clip(
                epi_params["CFR"] * mc_params["F_scaling"] * F_RR_fac[self.g_data.adm1_id],
                0.0,
                1.0,
            )
            epi_params["CHR"] = xp.clip(epi_params["CHR"] * scaling_H, epi_params["CFR"], 1.0)

            adm2_chr_delay = xp.sum(
                epi_params["I_TO_H_TIME"][:, None] * S_age_dist,
                axis=0,
            )
            adm2_chr_delay_int = adm2_chr_delay.astype(int)  # TODO temp, this should be a distribution of floats
            adm2_chr_delay_mod = adm2_chr_delay % 1
            inc_case_h_delay = (1.0 - adm2_chr_delay_mod) * xp.take_along_axis(
                self.g_data.csse_data.incident_cases,
                -adm2_chr_delay_int[None, :],
                axis=0,
            )[0] + adm2_chr_delay_mod * xp.take_along_axis(
                self.g_data.csse_data.incident_cases,
                -adm2_chr_delay_int[None, :] - 1,
                axis=0,
            )[
                0
            ]
            inc_case_h_delay[(inc_case_h_delay > 0.0) & (inc_case_h_delay < 1.0)] = 1.0
            inc_case_h_delay[inc_case_h_delay < 0.0] = 0.0
            adm2_chr = xp.sum(epi_params["CHR"] * S_age_dist, axis=0)

            tmp = (
                xp.sum(epi_params["CHR"] * I_init / yy.I_gamma_k * self.g_data.Nij, axis=0) / epi_params["CRR"]
            ) * epi_params[
                "GAMMA_H"
            ]  # * self.params.SIGMA #** 2
            tmp2 = inc_case_h_delay * adm2_chr

            ic_fac = tmp2 / tmp
            ic_fac[~xp.isfinite(ic_fac)] = xp.nanmean(ic_fac[xp.isfinite(ic_fac)])
            # ic_fac = xp.clip(ic_fac, a_min=0.2, a_max=5.0)  #####

            epi_params["HFR"] = xp.clip(mc_params["F_fac"] * epi_params["CFR"] / epi_params["CHR"], 0.0, 1.0)
            yy.I = (1.0 - epi_params["CHR"] * epi_params["CRR"]) * I_init / yy.I_gamma_k  # * 0.8  # noqa: E741
            yy.Ic *= ic_fac * 0.5  # * 0.9 * .9
            yy.Rh *= 1.0  # * adm2_hosp_frac * mc_params["F_fac"]

        R_init -= xp.sum(yy.Rh, axis=0)

        yy.Ia = epi_params["ASYM_FRAC"] / epi_params["SYM_FRAC"] * I_init / yy.I_gamma_k
        yy.E = exp_frac[None, :] * I_init / yy.E_gamma_k  # this should be calcable from Rt and the time before symp
        yy.R = xp.clip(R_init, a_min=0.0, a_max=None)
        yy.D = D_init

        # TMP
        yy.state = xp.clip(yy.state, a_min=0.0, a_max=None)
        mask = xp.sum(yy.N, axis=0) > 1.0
        yy.state[:, mask] /= xp.sum(yy.N, axis=0)[mask]
        mask = xp.sum(yy.N, axis=0) < 1.0
        yy.S[mask] /= 1.0 - xp.sum(yy.N, axis=0)[mask]

        yy.init_S()
        # init the bin we're using to track incident cases
        # (it's filled with cumulatives until we diff it later)
        # TODO should this come from the rolling hist?
        yy.incC = (
            xp.clip(self.g_data.csse_data.cumulative_cases[-1][None, :], a_min=0.0, a_max=None) * S_age_dist / self.Nij
        )

        # Sanity check state vector
        yy.validate_state()

        # TODO return y rather than keeping it in self
        self.y = yy

        return epi_params  # sampled_params

    def run_once(self, seed=None):
        """Perform one complete run of the simulation."""
        # rename to integrate or something? it also resets...

        # reset everything
        logger.debug("Resetting state")
        self.set_seed(seed)
        epi_params = self.reset()
        logger.debug("Done reset")

        self.base_mc_instance.epi_params = epi_params  # self.params
        self.base_mc_instance.state = self.y
        self.base_mc_instance.Aij = self.g_data.Aij.A
        self.base_mc_instance.rhs = RHS_func
        self.base_mc_instance.dy = self.y.zeros_like()

        # TODO this logic needs to go somewhere else (its rescaling beta to account for S/N term)
        # TODO R0 need to be changed before reset()...
        S_eff = self.base_mc_instance.S_eff(0, self.base_mc_instance.state)
        adm2_S_eff = xp.sum(S_eff * self.g_data.Nij / self.g_data.Nj, axis=0)
        adm2_beta_scale = xp.clip(1.0 / (adm2_S_eff + 1e-10), a_min=0.1, a_max=10.0)
        adm1_S_eff = xp.sum(self.g_data.sum_adm1((S_eff * self.g_data.Nij).T).T / self.g_data.adm1_Nj, axis=0)
        adm1_beta_scale = xp.clip(1.0 / (adm1_S_eff + 1e-10), a_min=0.1, a_max=10.0)
        adm2_beta_scale = adm1_beta_scale[self.g_data.adm1_id]

        # adm2_beta_scale = xp.sqrt(adm2_beta_scale)

        # self.base_mc_instance.epi_params["R0"] = self.base_mc_instance.epi_params["R0"] * adm2_beta_scale
        epi_params["Rt"] = epi_params["Rt"] * adm2_beta_scale
        self.base_mc_instance.epi_params["BETA"] = self.base_mc_instance.epi_params["BETA"] * adm2_beta_scale
        adm2_E_tot = xp.sum(self.y.E * self.g_data.Nij / self.g_data.Nj, axis=(0, 1))
        adm2_new_E_tot = adm2_beta_scale * adm2_E_tot
        S_dist = S_eff / (xp.sum(S_eff, axis=0) + 1e-10)

        new_E = xp.tile(
            (S_dist * adm2_new_E_tot / self.g_data.Nij * self.g_data.Nj / self.cfg["model.structure.E_gamma_k"])[
                None,
                ...,
            ],
            (xp.to_cpu(self.cfg["model.structure.E_gamma_k"]), 1, 1),
        )
        new_S = self.y.S - xp.sum(new_E - self.y.E, axis=0)

        self.base_mc_instance.state.E = new_E
        self.base_mc_instance.state.S = new_S

        self.base_mc_instance.epi_params["BETA"] = xp.broadcast_to(
            self.base_mc_instance.epi_params["BETA"],
            self.g_data.Nij.shape,
        )
        # do integration
        logger.debug("Starting integration")
        sol = xp_ivp.solve_ivp(**self.base_mc_instance.integrator_args)
        logger.debug("Done integration")

        return sol, epi_params

    def run_multiple(self, n_mc, base_seed=42, out_columns=None, invalid_ret=None):
        """Perform multiple monte carlos and return their postprocessed results."""
        seed_seq = np.random.SeedSequence(base_seed)
        success = 0
        fail = 0
        ret = []
        pbar = tqdm.tqdm(total=n_mc, desc="Performing Monte Carlos", dynamic_ncols=True)
        while success < n_mc:
            mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]  # inc spawn key then grab next seed
            pbar.set_postfix_str(
                "seed=" + str(mc_seed),
                refresh=True,
            )

            ###CTM_START
            # try:
            #     if fail > n_mc:
            #         return invalid_ret
            #
            #     with xp.optimize_kernels():
            #         sol, epi_params = self.run_once(seed=mc_seed)
            #         mc_data = self.postprocess_run(sol, mc_seed, out_columns)
            #     ret.append(mc_data)
            #     success += 1
            #     pbar.update(1)
            # except SimulationException:
            #     fail += 1
            #
            # except ValueError:
            #     fail += 1
            #     logger.warning("nan in rhs")
            ###CTM_END
            ###CTM_START
            if fail > n_mc:
                return invalid_ret

            ###CTM NOTE: b/c turning off @enable_cupy, this uses null context manager, so removing with is ok
            # with xp.optimize_kernels():
            #     sol, epi_params = self.run_once(seed=mc_seed)
            #     mc_data = self.postprocess_run(sol, mc_seed, out_columns)
            sol, epi_params = self.run_once(seed=mc_seed)
            mc_data = self.postprocess_run(sol, mc_seed, out_columns)
            ###CTM

            ret.append(mc_data)
            success += 1
            pbar.update(1)
            ###CTM_END

        pbar.close()
        return ret

    # TODO Also provide methods like to_dlpack, to_pytorch, etc
    def save_run(self, sol, epi_params, seed):
        """Postprocess and write to disk the output of run_once."""

        mc_data = self.postprocess_run(sol, epi_params, seed)

        self.writer.write_mc_data(mc_data)

        # TODO write params out (to yaml?) in another subfolder

        # TODO we should output the per monte carlo param rolls, this got lost when we switched from hdf5

    def postprocess_run(self, sol, epi_params, seed, columns=None):
        """Process the output of a run (sol, returned by the integrator) into the requested output vars."""
        if columns is None:
            columns = [
                "adm2_id",
                "date",
                "rid",
                "total_population",
                "current_hospitalizations",
                "active_asymptomatic_cases",
                "cumulative_deaths",
                "daily_hospitalizations",
                "daily_cases",
                "daily_reported_cases",
                "daily_deaths",
                "cumulative_cases",
                "cumulative_reported_cases",
                "current_icu_usage",
                "current_vent_usage",
                "case_reporting_rate",
                "Rt",
            ]

            columns = set(columns)

        mc_data = {}

        out = buckyState(self.cfg["model.structure"], self.Nij)

        y = sol.y.reshape(self.y.state_shape + (sol.y.shape[-1],))

        # rescale by population
        out.state = self.Nij[None, ..., None] * y

        # collapse age groups
        out.state = xp.sum(out.state, axis=1)

        # population_conserved = (xp.diff(xp.around(xp.sum(out.N, axis=(0, 1)), 1)) == 0.0).all()
        # if not population_conserved:
        #    pass  # TODO we're getting small fp errors here
        #    # print(xp.sum(xp.diff(xp.around(xp.sum(out[:incH], axis=(0, 1)), 1))))
        #    # logging.error("Population not conserved!")
        #    # print(xp.sum(xp.sum(y[:incH],axis=0)-1.))
        #    # raise SimulationException

        if "adm2_id" in columns:
            adm2_ids = np.broadcast_to(self.g_data.adm2_id[:, None], out.state.shape[1:])
            mc_data["adm2_id"] = adm2_ids

        if "date" in columns:
            mc_data["date"] = np.broadcast_to(np.arange(len(self.projected_dates)), out.state.shape[1:])

        if "rid" in columns:
            mc_data["rid"] = np.broadcast_to(seed, out.state.shape[1:])

        if "current_icu_usage" in columns or "current_vent_usage" in columns:
            icu = self.Nij[..., None] * epi_params["ICU_FRAC"][:, None, None] * xp.sum(y[out.indices["Rh"]], axis=0)
            if "current_icu_usage" in columns:
                mc_data["current_icu_usage"] = xp.sum(icu, axis=0)

            if "current_vent_usage" in columns:
                vent = epi_params["ICU_VENT_FRAC"][:, None, None] * icu
                mc_data["current_vent_usage"] = xp.sum(vent, axis=0)

        if "daily_deaths" in columns:
            daily_deaths = xp.gradient(out.D, axis=-1, edge_order=2)
            daily_deaths[:, 0] = xp.maximum(0.0, daily_deaths[:, 0])
            mc_data["daily_deaths"] = daily_deaths

            if self.reject_runs:
                init_inc_death_mean = xp.mean(xp.sum(daily_deaths[:, 1:4], axis=0))
                hist_inc_death_mean = xp.mean(xp.sum(self.g_data.csse_data.incident_deaths[-7:], axis=-1))

                inc_death_rejection_fac = 2.0  # TODO These should come from the cli arg -r
                if (init_inc_death_mean > inc_death_rejection_fac * hist_inc_death_mean) or (
                    inc_death_rejection_fac * init_inc_death_mean < hist_inc_death_mean
                ):
                    logger.info("Inconsistent inc deaths, rejecting run")
                    raise SimulationException

        if "daily_cases" in columns or "daily_reported_cases" in columns:
            daily_reported_cases = xp.gradient(out.incC, axis=-1, edge_order=2)
            daily_reported_cases[:, 0] = xp.maximum(0.0, daily_reported_cases[:, 0])

            if self.reject_runs:
                init_inc_case_mean = xp.mean(xp.sum(daily_reported_cases[:, 1:4], axis=0))
                hist_inc_case_mean = xp.mean(xp.sum(self.g_data.csse_data.incident_cases[-7:], axis=-1))

                inc_case_rejection_fac = 1.5  # TODO These should come from the cli arg -r
                if (init_inc_case_mean > inc_case_rejection_fac * hist_inc_case_mean) or (
                    inc_case_rejection_fac * init_inc_case_mean < hist_inc_case_mean
                ):
                    logger.info("Inconsistent inc cases, rejecting run")
                    raise SimulationException

            if "daily_reported_cases" in columns:
                mc_data["daily_reported_cases"] = daily_reported_cases

            if "daily_cases" in columns:
                daily_cases_total = daily_reported_cases / epi_params["CRR"][:, None]
                mc_data["daily_cases"] = daily_cases_total

        if "cumulative_reported_cases" in columns:
            cum_cases_reported = out.incC
            mc_data["cumulative_reported_cases"] = cum_cases_reported

        if "cumulative_cases" in columns:
            cum_cases_total = out.incC / epi_params["CRR"][:, None]
            mc_data["cumulative_cases"] = cum_cases_total

        if "daily_hospitalizations" in columns:
            out.incH[:, 0] = out.incH[:, 1] - out.incH[:, 2]
            daily_hosp = xp.gradient(out.incH, axis=-1, edge_order=2)
            daily_hosp[:, 0] = xp.maximum(0.0, daily_hosp[:, 0])
            mc_data["daily_hospitalizations"] = daily_hosp

        if "total_population" in columns:
            N = xp.broadcast_to(self.g_data.Nj[..., None], out.state.shape[1:])
            mc_data["total_population"] = N

        if "current_hospitalizations" in columns:
            hosps = xp.sum(out.Rh, axis=0)  # why not just using .H?
            mc_data["current_hospitalizations"] = hosps

        if "cumulative_deaths" in columns:
            cum_deaths = out.D
            mc_data["cumulative_deaths"] = cum_deaths

        if "active_asymptomatic_cases" in columns:
            asym_I = xp.sum(out.Ia, axis=0)
            mc_data["active_asymptomatic_cases"] = asym_I

        if "case_reporting_rate" in columns:
            crr = xp.broadcast_to(epi_params["CRR"][:, None], adm2_ids.shape)
            mc_data["case_reporting_rate"] = crr

        if "Rt" in columns:
            #    r_eff = self.npi_params["r0_reduct"].T * np.broadcast_to(
            #        (self.params.R0 * self.g_data.Aij.diag)[:, None], adm2_ids.shape
            #    )
            mc_data["R_eff"] = xp.broadcast_to(epi_params["Rt"][:, None], adm2_ids.shape)

        if self.cfg["model.flags.vacc_reroll"]:
            dose1 = xp.sum(self.base_mc_instance.vacc_data.dose1 * self.Nij[None, ...], axis=1).T
            dose2 = xp.sum(self.base_mc_instance.vacc_data.dose2 * self.Nij[None, ...], axis=1).T
            mc_data["vacc_dose1"] = dose1
            mc_data["vacc_dose2"] = dose2
            dose1_65 = xp.sum((self.base_mc_instance.vacc_data.dose1 * self.Nij[None, ...])[:, -3:], axis=1).T
            dose2_65 = xp.sum((self.base_mc_instance.vacc_data.dose2 * self.Nij[None, ...])[:, -3:], axis=1).T
            mc_data["vacc_dose1_65"] = dose1_65
            mc_data["vacc_dose2_65"] = dose2_65

            pop = xp.sum((self.Nij[None, ...]), axis=1).T
            pop_65 = xp.sum((self.Nij[None, ...])[:, -3:], axis=1).T

            mc_data["frac_vacc_dose1"] = dose1 / pop
            mc_data["frac_vacc_dose2"] = dose2 / pop
            mc_data["frac_vacc_dose1_65"] = dose1_65 / pop_65
            mc_data["frac_vacc_dose2_65"] = dose2_65 / pop_65

            # TODO just use the already inited state object if we need immunity estimates
            # tmp = buckyState(self.consts, self.Nij)
            # v_eff = xp.zeros_like(self.base_mc_instance.vacc_data.dose1)
            # for i in range(y.shape[-1]):
            #    tmp.state = y[..., i]
            #    v_eff[i] = self.base_mc_instance.vacc_data.V_eff(tmp, epi_params, i) + tmp.R

            # imm = xp.sum(v_eff * self.Nij[None, ...], axis=1).T
            # imm_65 = xp.sum((v_eff * self.Nij[None, ...])[:, -3:], axis=1).T
            # mc_data["immune"] = imm
            # mc_data["immune_65"] = imm_65
            # mc_data["frac_immune"] = imm / pop
            # mc_data["frac_immune_65"] = imm_65 / pop_65

            # phase
            mc_data["state_phase"] = self.base_mc_instance.vacc_data.phase_hist.T

        # Check for any negative values in the ouput data
        negative_values = False
        for k, val in mc_data.items():
            if k != "date" and xp.any(xp.around(val, 5) < 0.0):
                logger.info("Negative values present in " + k)
                negative_values = True

        if negative_values and self.reject_runs:
            logger.info("Rejecting run b/c of negative values in output")
            raise SimulationException

        # TODO output epi_params, might need to happen before we do all the broadcasting?
        # self.writer.write_params(seed, self.params)

        return mc_data


def main(cfg=None):
    """Main method for a complete simulation called with a set of CLI args."""

    if cfg["runtime.use_cupy"]:
        logger.info("Using CuPy backend")
        ###CTM enable_cupy(optimize=True, cache_dir=cfg["system.cache_dir"])  # TODO need optk in cfg (args.optimize_kernels)
    else:
        logger.info("Using numpy backend")

    ###CTM reimport_numerical_libs("model.main.main")

    # Make sure output data folder exists  TODO this should happen somewhere else...
    if not cfg["system.raw_output_dir"].exists():
        cfg["system.raw_output_dir"].mkdir(parents=True)

    # Display banner
    _banner()

    # need to adapt loguru to tqdm (temporarily)
    # see https://loguru.readthedocs.io/en/stable/resources/recipes.html#interoperability-with-tqdm-iterations

    ###CTM Turning off logging_redirect_tqdm() context in with clause
    ###CTM START with
    ###CTM with logging_redirect_tqdm():

    # Init main model class
    # TODO this should happen in the ctrl+c catching below but it can leave the write thread zombied
    env = buckyModelCovid(
        cfg=cfg,
        npi_file=None,  # args.npi_file,
        disable_npi=False,  # args.disable_npi,
        reject_runs=False,  # args.reject_runs,
    )

    ###CTM_START try
    # try:
    #     # Monte Carlo loop
    #     seed_seq = np.random.SeedSequence(cfg["runtime.seed"])
    #     pbar = tqdm.tqdm(total=cfg["runtime.n_mc"], desc="Performing Monte Carlos", dynamic_ncols=True)
    #     total_start = datetime.datetime.now()
    #     success = 0
    #     n_runs = 0
    #
    #     while success < cfg["runtime.n_mc"]:
    #         # inc spawn key then grab next seed
    #         mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]
    #         pbar.set_postfix_str(
    #             "seed=" + str(mc_seed),
    #             # + ", rej%="  # TODO disable rej% if not -r
    #             # + str(np.around(float(n_runs - success) / (n_runs + 0.00001) * 100, 1)),
    #             # refresh=True,
    #         )
    #         try:
    #             n_runs += 1
    #             with xp.optimize_kernels():
    #                 sol, epi_params = env.run_once(seed=mc_seed)
    #                 env.save_run(sol, epi_params, mc_seed)
    #
    #             success += 1
    #             pbar.update(1)
    #         except SimulationException as e:
    #             logger.debug(e)
    #
    # except (KeyboardInterrupt, SystemExit):
    #     logger.warning("Caught SIGINT, cleaning up")
    #     env.writer.close()  # TODO need env.close() which checks if writer is inited
    # finally:
    #     env.writer.close()
    #     if "pbar" in locals():
    #         pbar.close()
    #         logger.info(f"Total runtime: {datetime.datetime.now() - total_start}")
    ###CTM_END try
    ###CTM_START try
    # Monte Carlo loop
    seed_seq = np.random.SeedSequence(cfg["runtime.seed"])
    pbar = tqdm.tqdm(total=cfg["runtime.n_mc"], desc="Performing Monte Carlos", dynamic_ncols=True)
    total_start = datetime.datetime.now()
    success = 0
    n_runs = 0

    while success < cfg["runtime.n_mc"]:
        # inc spawn key then grab next seed
        mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]
        pbar.set_postfix_str(
            "seed=" + str(mc_seed),
            # + ", rej%="  # TODO disable rej% if not -r
            # + str(np.around(float(n_runs - success) / (n_runs + 0.00001) * 100, 1)),
            # refresh=True,
        )

        ###CTM Start of nested try block
        n_runs += 1
        ###CTM NOTE: b/c turning off @enable_cupy, this uses null context manager, so removing with is ok
        # with xp.optimize_kernels():
        #     sol, epi_params = env.run_once(seed=mc_seed)
        #     env.save_run(sol, epi_params, mc_seed)
        sol, epi_params = env.run_once(seed=mc_seed)
        env.save_run(sol, epi_params, mc_seed)
        ###CTM

        success += 1
        pbar.update(1)
    ###CTM_END try
    ###CTM END with

    return env.writer.output_dir


###CTM_START
if __name__ == "__main__":
    file = "../base_config"
    cfg = BuckyConfig().load_cfg(file)
    main(cfg=cfg)
###CTM_END
