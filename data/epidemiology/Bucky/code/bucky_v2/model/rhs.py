"""RHS function that is passed to scipy.solve_ivp."""
###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END
###CTM from ..util.ode_constraints import constrain_y_range

#
# RHS for odes - d(sstate)/dt = F(t, state, *mats, *pars)
# NB: requires the state vector be 1d
#


###CTM @sync_numerical_libs
###CTM @constrain_y_range([0, 1])
def RHS_func(t, y_flat, mc_inst):
    """RHS function for the ODEs, get's called in ivp.solve_ivp."""

    # TODO we're passing in y.state just to overwrite it, we probably need another class
    # reshape to the usual state tensor (compartment, age, node)
    y = mc_inst.state
    y.state = y_flat.reshape(y.state_shape)

    # init d(state)/dt
    dy = mc_inst.dy

    if mc_inst.npi_active or mc_inst.vacc_active:
        t_index = min(int(t), mc_inst.t_max)  # prevent OOB error when the integrator overshoots
    else:
        t_index = None

    # TODO add a function to mc instance that fills all these in using nonlocal?
    npi = mc_inst.npi_params
    par = mc_inst.epi_params
    BETA_eff = mc_inst.BETA_eff(t_index)
    if hasattr(mc_inst, "scen_beta_scale"):
        BETA_eff = mc_inst.scen_beta_scale[t_index] * BETA_eff

    HFR = par["HFR"]
    CHR = par["CHR"]
    THETA = y.Rh_gamma_k * par["THETA"]
    GAMMA = y.I_gamma_k * par["GAMMA"]
    GAMMA_H = y.I_gamma_k * par["GAMMA_H"]
    SIGMA = y.E_gamma_k * par["SIGMA"]
    SYM_FRAC = par["SYM_FRAC"]
    CRR = par["CRR"]

    Cij = mc_inst.Cij(t_index)
    Aij = mc_inst.Aij  # TODO needs to take t and return Aij_eff

    if mc_inst.npi_active:
        Aij_eff = npi["mobility_reduct"][t_index] * Aij
    else:
        Aij_eff = Aij

    S_eff = mc_inst.S_eff(t_index, y)

    Nij = mc_inst.Nij

    # perturb Aij
    # new_R0_fracij = truncnorm(xp, 1.0, .1, size=Aij.shape, a_min=1e-6)
    # new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
    # A = Aij * new_R0_fracij
    # Aij_eff = A / xp.sum(A, axis=0)

    # Infectivity matrix (I made this name up, idk what its really called)
    I_tot = xp.sum(Nij * y.Itot, axis=0) - (1.0 - par["rel_inf_asym"]) * xp.sum(Nij * y.Ia, axis=0)

    I_tmp = I_tot @ Aij_eff  # using identity (A@B).T = B.T @ A.T

    # beta_mat = y.S * xp.squeeze((Cij @ I_tmp.T[..., None]), axis=-1).T
    beta_mat = S_eff * (Cij @ xp.atleast_3d(I_tmp.T)).T[0]
    beta_mat /= Nij

    # dS/dt
    dy.S = -BETA_eff * (beta_mat)
    # dE/dt
    dy.E[0] = BETA_eff * (beta_mat) - SIGMA * y.E[0]
    dy.E[1:] = SIGMA * (y.E[:-1] - y.E[1:])

    # dIa/dt
    dy.Ia[0] = (1.0 - SYM_FRAC) * SIGMA * y.E[-1] - GAMMA * y.Ia[0]
    dy.Ia[1:] = GAMMA * (y.Ia[:-1] - y.Ia[1:])

    # dI/dt
    dy.I[0] = SYM_FRAC * (1.0 - CHR * CRR) * SIGMA * y.E[-1] - GAMMA * y.I[0]
    dy.I[1:] = GAMMA * (y.I[:-1] - y.I[1:])

    # dIc/dt
    dy.Ic[0] = SYM_FRAC * CHR * CRR * SIGMA * y.E[-1] - GAMMA_H * y.Ic[0]
    dy.Ic[1:] = GAMMA_H * (y.Ic[:-1] - y.Ic[1:])

    # dRhi/dt
    dy.Rh[0] = GAMMA_H * y.Ic[-1] - THETA * y.Rh[0]
    dy.Rh[1:] = THETA * (y.Rh[:-1] - y.Rh[1:])

    # dR/dt
    dy.R = GAMMA * (y.I[-1] + y.Ia[-1]) + (1.0 - HFR) * THETA * y.Rh[-1]

    # dD/dt
    dy.D = HFR * THETA * y.Rh[-1]

    dy.incH = GAMMA_H * y.Ic[-1]  # SYM_FRAC * CHR * SIGMA * y.E[-1]
    dy.incC = SYM_FRAC * CRR * SIGMA * y.E[-1]

    # bring back to 1d for the ODE api
    dy_flat = dy.state.ravel()

    return dy_flat
