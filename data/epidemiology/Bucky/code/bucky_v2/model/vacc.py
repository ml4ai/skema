"""Vaccine related functions."""
import datetime
import pickle  # noqa: S403

import pandas as pd
import us

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END

from ..util.distributions import truncnorm


class buckyVaccAlloc:
    """Class managing all the vaccine rollout related estimates."""

    ###CTM @sync_numerical_libs
    def __init__(self, g_data, cfg, first_date, scen_params=None):
        """Initialize."""
        end_t = cfg["runtime.t_max"]
        self.flags = cfg["model.flags"]

        if not self.flags["vaccines"]:
            self.active = False
            self.reroll = False
            return

        self.Nij = g_data.Nij
        self.adm1_pop_frac = g_data.adm1_Nj / g_data.N

        state_abbr_map = us.states.mapping("abbr", "fips")

        self.end_t = end_t  # + 1
        self.g_data = g_data
        # alloc the arrays for doses indexed by time, age_grp, adm2
        self.dose1 = xp.zeros((end_t + 1,) + self.Nij.shape)
        self.dose2 = xp.zeros((end_t + 1,) + self.Nij.shape)

        self.scen_params = scen_params

        vac_hist = pd.read_csv("data/raw/vaccine_timeseries/vacc-timeseries.csv")
        vac_hist["adm1"] = vac_hist.Location.map(state_abbr_map)
        vac_hist = vac_hist.loc[~vac_hist.adm1.isna()]
        vac_hist["adm1"] = vac_hist["adm1"].astype(int)
        vac_hist["Date"] = pd.to_datetime(vac_hist.Date)
        vac_hist["Doses_Distributed_rolling_daily"] = vac_hist["Doses_Distributed"].diff().rolling(7).mean()

        self.dose1_t = cfg["model.vaccine.dose1_t"]
        self.dose2_t = cfg["model.vaccine.dose2_t"]
        hist_t = cfg["model.vaccine.dose2_t"] + 1  # consts.vacc_dose2_t.item()  # + 1
        self.hist_t = hist_t

        # init daily allocs
        adm0_hist_dist = xp.array(
            vac_hist.groupby("Date").sum()["Doses_Distributed"].diff().rolling(7).mean().to_numpy()[-50:],
        )
        self.mean_vac_daily = xp.mean(adm0_hist_dist[-14:])
        self.std_vac_daily = xp.std(adm0_hist_dist[-28:])
        if scen_params is not None:
            self.mean_vac_daily = 2.0 * scen_params["dose1_per_mo"] / 30.0
        # TODO this needs to be added to the MC down to line 67? (some of it can be cached for sure)

        daily_vaccs_dist_adm1 = self.adm1_pop_frac * self.mean_vac_daily

        self.adm1_vac_timeseries = xp.tile(daily_vaccs_dist_adm1, (end_t + 1 + hist_t, 1))
        self.adm1_vac_timeseries = xp.zeros_like(self.adm1_vac_timeseries) - 1.0

        # fill in first day of timeseries w/ cumulative up until that point
        init_vacs = vac_hist.loc[vac_hist.Date == pd.Timestamp(first_date - datetime.timedelta(days=hist_t))]
        init_vacs = init_vacs.set_index("adm1").sort_index().Doses_Distributed.reset_index().to_numpy().astype(int).T

        adm1_init_vac = xp.array(init_vacs[1])

        self.adm1_vac_timeseries[0] = adm1_init_vac

        # overwrite daily allocs w/ historical data if present
        # vac_hist['Doses_Distributed_rolling_daily'] = vac_hist['Doses_Distributed'].diff().rolling(7).mean()

        for t in range(1, hist_t + 1):
            daily_vacs = vac_hist.loc[vac_hist.Date == pd.Timestamp(first_date - datetime.timedelta(days=hist_t - t))]

            daily_vacs = (
                daily_vacs.set_index("adm1")
                .sort_index()
                .Doses_Distributed_rolling_daily.reset_index()
                .to_numpy()
                .astype(int)
                .T
            )

            adm1_daily_vac = xp.array(daily_vacs[1])

            ###CTM_START
            # try:
            #     self.adm1_vac_timeseries[t] = adm1_daily_vac
            # except ValueError:
            #     # TODO warn about missing data
            #     self.adm1_vac_timeseries[t] = self.adm1_vac_timeseries[t - 1]
            ###CTM_END
            ###CTM_START
            self.adm1_vac_timeseries[t] = adm1_daily_vac
            ###CTM_END

        self.dist_future_mask = xp.all(self.adm1_vac_timeseries < 0, axis=1)
        self.n_future_days = xp.to_cpu(xp.sum(self.dist_future_mask))

        self.adm1_vac_timeseries[self.dist_future_mask] = xp.tile(daily_vaccs_dist_adm1, (self.n_future_days, 1))

        self.child_vac = False

        # get cumulative vacc dist
        vaccs_dist_adm1 = xp.cumsum(self.adm1_vac_timeseries, axis=0)
        vaccs_dist_adm1 = xp.clip(vaccs_dist_adm1, a_min=0.0, a_max=2.0 * self.g_data.adm1_Nj)
        self.vaccs_dist_adm1 = vaccs_dist_adm1

        ###CTM_START
        # with open(cfg["system.data_dir"] / "raw/included_data/adm2_phased_age_dists.p", "rb") as f:
        #     phase_demos = pickle.load(f)  # noqa: S301
        ###CTM_END
        ###CTM_START
        f = open(cfg["system.data_dir"] / "raw/included_data/adm2_phased_age_dists.p", "rb")
        phase_demos = pickle.load(f)  # noqa: S301
        f.close()
        ###CTM_END

        # fill in missing state plans w/ mean of demos from acip states
        tmp = -xp.ones(g_data.Nij.shape).T
        tmp = xp.tile(tmp[None, ...], (3, 1, 1))
        states_following_acip = [1, 26, 27, 32, 42, 47, 55]
        for adm1 in states_following_acip:
            # TODO this depends on the fact that both Nij and phase_demos are sorted by fips
            adm1_mask = g_data.adm_mapping.adm1.ids[g_data.adm_mapping.adm1.idx] == adm1
            tmp[:, adm1_mask] = phase_demos[adm1]
        tmp = xp.clip(tmp.T / (g_data.Nij[..., None] + 1.0), a_min=-1.0, a_max=1.0)
        mean_acip_demos = xp.mean(tmp[:, (tmp >= 0).all((0, 2)), :], axis=1)

        adm1_without_phase_data = xp.unique(
            g_data.adm1_id[~xp.isin(g_data.adm1_id, xp.array(list(phase_demos.keys())))],
        )

        for adm1 in adm1_without_phase_data:
            n_adm2 = xp.sum(g_data.adm1_id == adm1)
            adm1_phase_demo = xp.tile(mean_acip_demos[:, None, ...], (1, n_adm2.item(), 1))
            adm1_phase_pop = (adm1_phase_demo * g_data.Nij[:, g_data.adm1_id == adm1, None]).T
            phase_demos[adm1.item()] = adm1_phase_pop

        # rescale all phase demos to be pop frac and add in a 'general population' phase at the end
        for adm1 in phase_demos:
            phases_frac = xp.array(phase_demos[adm1]) / g_data.Nij[:, g_data.adm1_id == adm1].T[None, ...]

            phases_frac = xp.clip(phases_frac, a_min=0.0, a_max=1.0)
            gen_pop = 1.0 - xp.clip(xp.sum(phases_frac, axis=0), a_min=0.0, a_max=1.0)
            phase_demos[adm1] = xp.vstack([phases_frac, gen_pop[None, ...]])

        # put all the phase demographics into one big xp array
        max_phases = max([x.shape[0] for x in phase_demos.values()])
        self.max_phases = max_phases
        vacc_demos = xp.zeros((max_phases,) + g_data.Nij.shape)
        for adm1 in phase_demos:
            n_phases = phase_demos[adm1].shape[0]
            adm1_mask = g_data.adm1_id == adm1
            adm1_demos = xp.swapaxes(phase_demos[adm1], 1, 2)
            padded_demos = xp.pad(adm1_demos, ((0, max_phases - n_phases), (0, 0), (0, 0)))

            vacc_demos[:, :, adm1_mask] = padded_demos

        # apply hesitancy
        self.baseline_vacc_demos = vacc_demos

        # Read in hes data
        df = pd.read_csv(cfg["system.data_dir"] / "raw/included_data/vaccine_hesitancy/vaccine_hesitancy_all_cols.csv")
        last_wk = xp.sort(df.wk.unique())[-1]
        df = df.loc[df.wk == last_wk]
        df = df.drop(columns=["wk"])
        df["adm1"] = df.state.map(state_abbr_map).astype(int)
        # df = df.set_index(['age_grp', 'adm1']).Total.unstack(0)
        df_se = pd.read_csv(
            cfg["system.data_dir"] / "raw/included_data/vaccine_hesitancy/vaccine_hesitancy_all_cols_se.csv",
        )
        df_se = df_se.loc[df_se.wk == last_wk]
        df_se = df_se.drop(columns=["wk"])
        df_se["adm1"] = df_se.state.map(state_abbr_map).astype(int)

        df = df.set_index(["age_grp", "adm1"]).unstack(0)
        df_se = df_se.set_index(["age_grp", "adm1"]).unstack(0)
        age_map = {"18-24": (3, 5), "25-39": (5, 8), "40-54": (8, 11), "55-64": (11, 13), "65+": (13, 16)}
        # resp = pd.read_csv('data/vac/hes/household_survey_respondents_all.csv')
        # resp['adm1'] = resp.state.map(state_name_map).astype(int)
        # resp = resp.set_index('adm1')['WEEK '+str(last_wk)]
        # resp_factor = xp.zeros_like(self.g_data.adm1_Nj)
        # resp_factor[resp.index.to_numpy()] = resp.to_numpy()
        # resp_factor = xp.sqrt(resp_factor)
        self.hes_frac_ij_adm1 = xp.zeros_like(self.g_data.adm1_Nij)
        self.hes_se_ij_adm1 = xp.zeros_like(self.g_data.adm1_Nij)
        adm1_ind_map = {v: i for i, v in enumerate(g_data.adm_mapping.adm1.ids)}
        df.index = df.index.map(adm1_ind_map)
        df_se.index = df_se.index.map(adm1_ind_map)
        if scen_params is not None:
            df["hes"] = df[scen_params["hes_col"]]
        for col in age_map:  # pylint: disable=consider-using-dict-items

            adm1_age_grp_pop = xp.sum(self.g_data.adm1_Nij[slice(*(age_map[col]))], axis=0)
            hes_pop = xp.zeros_like(adm1_age_grp_pop)
            hes_pop[df["hes"][col].index.to_numpy()] = df["hes"][col].to_numpy()
            hes_pop_se = xp.zeros_like(adm1_age_grp_pop)
            hes_pop_se[df["hes"][col].index.to_numpy()] = df_se["hes"][col].to_numpy()
            total_resp = xp.zeros_like(adm1_age_grp_pop)
            total_resp[df["total_resp"][col].index.to_numpy()] = df["total_resp"][col].to_numpy()
            # hes_pop_se *= resp_factor

            # hes_frac = hes_pop/adm1_age_grp_pop
            self.hes_frac_ij_adm1[slice(*(age_map[col]))] = (hes_pop / (total_resp + 1e-10))[None, ...]
            self.hes_se_ij_adm1[slice(*(age_map[col]))] = (hes_pop_se / (total_resp + 1e-10))[None, ...]
            # self.hes_frac_ij_adm1[slice(*(age_map[col]))] =  hes_pop/(adm1_age_grp_pop+ 1e-10)
            # self.hes_se_ij_adm1[slice(*(age_map[col]))] =  hes_pop_se/(adm1_age_grp_pop+ 1e-10)

        # cover the non surveyed groups
        self.hes_frac_ij_adm1[:3] = xp.sum(self.hes_frac_ij_adm1[3:] * self.g_data.adm1_Nij[3:], axis=0) / (
            xp.sum(self.g_data.adm1_Nij[3:], axis=0) + 1e-10
        )
        self.hes_se_ij_adm1[:3] = xp.sum(self.hes_se_ij_adm1[3:] * self.g_data.adm1_Nij[3:], axis=0) / (
            xp.sum(self.g_data.adm1_Nij[3:], axis=0) + 1e-10
        )
        self.hes_frac_ij_adm1[:3] = 1.0
        self.hes_se_ij_adm1[:3] = 1e-10
        self.hes_frac_ij_adm1[2] = self.hes_frac_ij_adm1[3] + 0.4  # 12-14 yo
        self.hes_se_ij_adm1[2] = self.hes_se_ij_adm1[3]

        pop_but_no_hes_data_mask = (xp.sum(self.g_data.adm1_Nij, axis=0) > 0.0) & (
            xp.sum(self.hes_frac_ij_adm1[5:], axis=0) == 0.0
        )
        pop_hes_data_mask = (xp.sum(self.g_data.adm1_Nij, axis=0) > 0.0) & (
            xp.sum(self.hes_frac_ij_adm1[5:], axis=0) > 0.0
        )
        mean_hes = xp.mean(self.hes_frac_ij_adm1[:, pop_hes_data_mask], axis=1)
        mean_se = xp.mean(self.hes_se_ij_adm1[:, pop_hes_data_mask], axis=1)
        self.hes_frac_ij_adm1[:, pop_but_no_hes_data_mask] = mean_hes[..., None]
        self.hes_se_ij_adm1[:, pop_but_no_hes_data_mask] = mean_se[..., None]

        # vacc_demos = consts.vacc_hesitancy * self.baseline_vacc_demos

        hes_adm1 = truncnorm(
            (1.0 - self.hes_frac_ij_adm1),
            self.hes_se_ij_adm1,
            self.hes_frac_ij_adm1.shape,
        )  # , a_min=0., a_max=1.)
        hes_adm1 = xp.clip(hes_adm1, a_min=0.0, a_max=1.0)

        # if scen_params is not None:
        # hes_adm1 = xp.minimum(hes_adm1, scen_params['max_uptake'])
        # national_uptake = (xp.sum((self.dose2 * self.g_data.Nij), axis=[1,2])/xp.sum(self.g_data.adm0_Ni[3:]))[-1]

        vacc_demos = hes_adm1[:, self.g_data.adm_mapping.adm1.idx] * self.baseline_vacc_demos

        self.adm1_phase = xp.zeros((g_data.max_adm1 + 1,))
        self.pop_per_phase_adm1 = xp.zeros((max_phases, g_data.max_adm1 + 1))
        for p in range(max_phases):
            self.pop_per_phase_adm1[p] = g_data.sum_adm1(xp.sum(vacc_demos[p] * g_data.Nij, axis=0))

        phase_hist_adm1 = xp.zeros((self.end_t + 1,) + (self.g_data.max_adm1 + 1,))

        # TODO we can do this faster without the loops
        for t in range(0, end_t + 1 + hist_t):
            dose1_t = t + self.dose1_t - hist_t
            dose2_t = t + self.dose2_t - hist_t
            for p in range(max_phases):
                previous_phases_pop = xp.zeros_like(self.pop_per_phase_adm1[0])
                if p > 0:
                    previous_phases_pop = xp.sum(self.pop_per_phase_adm1[:p], axis=0)

                frac_dist = (vaccs_dist_adm1[t] / 2.0 - previous_phases_pop) / self.pop_per_phase_adm1[p]
                frac_dist = xp.clip(frac_dist, a_min=0.0, a_max=1.0)

                # if p == 2:
                #    print(t, p)
                #    print(frac_dist)
                tmp = xp.zeros_like(phase_hist_adm1[0])
                frac_dist_adm2 = frac_dist[g_data.adm_mapping.adm1.idx][None, ...]

                if dose1_t <= end_t and dose1_t >= 0:
                    self.dose1[dose1_t] += frac_dist_adm2 * vacc_demos[p]
                    tmp[(frac_dist < 1.0) & (frac_dist > 0.0)] = p
                    phase_hist_adm1[dose1_t] += tmp
                    # phase_hist_adm1[dose1_t,(frac_dist < 1.) & (frac_dist > 0.)] = p
                if dose2_t <= end_t and dose2_t >= 0:
                    self.dose2[dose2_t] += frac_dist_adm2 * vacc_demos[p]

        self.phase_hist = phase_hist_adm1[:, self.g_data.adm_mapping.adm1.idx]
        self.active = self.flags["vaccines"]
        self.reroll = self.flags["vaccine_monte_carlo"]

    def reroll_distribution(self):
        """Reroll the vaccine distributions to states after updating the params."""
        daily_vaccs_dist_adm1 = self.adm1_pop_frac * truncnorm(self.mean_vac_daily, self.std_vac_daily, a_min=0.0)

        self.adm1_vac_timeseries[self.dist_future_mask] = xp.tile(daily_vaccs_dist_adm1, (self.n_future_days, 1))
        # get cumulative vacc dist
        vaccs_dist_adm1 = xp.cumsum(self.adm1_vac_timeseries, axis=0)

        vaccs_dist_adm1 = xp.clip(vaccs_dist_adm1, a_min=0.0, a_max=2.0 * self.g_data.adm1_Nj)
        # self.vaccs_dist_adm1 = vaccs_dist_adm1

        daily_dists = xp.gradient(self.vaccs_dist_adm1[self.dist_future_mask], axis=0)
        daily_dists = daily_dists * truncnorm(1.0, self.std_vac_daily / self.mean_vac_daily, a_min=0.0)
        daily_dists[0] = self.vaccs_dist_adm1[self.dist_future_mask][0, :]
        self.vaccs_dist_adm1[self.dist_future_mask] = xp.cumsum(daily_dists, axis=0)

    def reroll_doses(self):
        """Reroll the number of vaccinated people with updated params (from the MC)."""
        self.dose1 = xp.zeros((self.end_t + 1,) + self.Nij.shape)
        self.dose2 = xp.zeros((self.end_t + 1,) + self.Nij.shape)

        # vacc_demos = params.vacc_hesitancy * self.baseline_vacc_demos
        # vacc_demos = (1. - self.hes_frac_ij_adm1)[:,self.g_data.adm1_id] * self.baseline_vacc_demos
        hes_adm1 = truncnorm((1.0 - self.hes_frac_ij_adm1), self.hes_se_ij_adm1, self.hes_frac_ij_adm1.shape)
        hes_adm1 = xp.clip(hes_adm1, a_min=0.0, a_max=1.0)
        # if self.scen_params is not None:
        #    hes_adm1 = xp.minimum(hes_adm1, self.scen_params['max_uptake'])

        vacc_demos = hes_adm1[:, self.g_data.adm1_id] * self.baseline_vacc_demos

        # print(params.vacc_hesitancy)
        self.adm1_phase = xp.zeros((self.g_data.max_adm1 + 1,))
        self.pop_per_phase_adm1 = xp.zeros((self.max_phases, self.g_data.max_adm1 + 1))
        for p in range(self.max_phases):
            self.pop_per_phase_adm1[p] = self.g_data.sum_adm1(xp.sum(vacc_demos[p] * self.g_data.Nij, axis=0))

        phase_hist_adm1 = xp.zeros((self.end_t + 1,) + (self.g_data.max_adm1 + 1,))

        # TODO we can do this faster without the loops
        hist_t = self.hist_t
        for t in range(0, self.end_t + 1 + hist_t):
            dose1_t = t + xp.to_cpu(self.dose1_t) - hist_t
            dose2_t = t + xp.to_cpu(self.dose2_t) - hist_t
            for p in xp.arange(self.max_phases, dtype=int):
                previous_phases_pop = xp.zeros_like(self.pop_per_phase_adm1[0])
                if p > 0:
                    previous_phases_pop = xp.sum(self.pop_per_phase_adm1[:p], axis=0)

                frac_dist = (self.vaccs_dist_adm1[t] / 2.0 - previous_phases_pop) / self.pop_per_phase_adm1[p]
                frac_dist = xp.clip(frac_dist, a_min=0.0, a_max=1.0)

                frac_dist_adm2 = frac_dist[self.g_data.adm1_id][None, ...]
                tmp = xp.zeros_like(phase_hist_adm1[0])
                if dose1_t <= self.end_t and dose1_t >= 0:
                    self.dose1[dose1_t] += frac_dist_adm2 * vacc_demos[p]
                    tmp[(frac_dist < 1.0) & (frac_dist > 0.0)] = p
                    phase_hist_adm1[dose1_t] += tmp
                if dose2_t <= self.end_t and dose2_t >= 0:
                    self.dose2[dose2_t] += frac_dist_adm2 * vacc_demos[p]

        self.phase_hist = phase_hist_adm1[:, self.g_data.adm1_id]

        if self.child_vac:
            self.dose1[:, 1] = self.dose1[:, 1] + self.cpct_dose1[:, self.g_data.adm1_id]
            self.dose1[:, 2] = self.dose1[:, 2] + 0.4 * self.cpct_dose1[:, self.g_data.adm1_id]
            self.dose2[:, 1] = self.dose2[:, 1] + self.cpct_dose2[:, self.g_data.adm1_id]
            self.dose2[:, 2] = self.dose2[:, 2] + 0.4 * self.cpct_dose2[:, self.g_data.adm1_id]

    def V_tot(self, params, t):
        """Total number of effectively vaccinated people as a fraction of the pop."""
        return self.dose2[t] * params["vacc_eff_2"] + params["vacc_eff_1"] * (self.dose1[t] - self.dose2[t])

    def V_eff(self, y, params, t):
        """Total number of people with immunity as a fraction of the pop."""
        R_asym = (1.0 - params["SYM_FRAC"] * params["CFR"]) * y.R
        eligable = y.S + R_asym
        V_tot = self.V_tot(params, t)
        S_frac = y.S / (eligable + 1.0e-10)
        V_eff = S_frac * V_tot
        return V_eff

    def S_eff(self, y, params, t):
        """Fraction of the population that is susceptible after removing those with immunity."""
        return xp.clip(y.S - self.V_eff(y, params, t), a_min=0.0, a_max=1.0)
