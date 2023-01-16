"""Calculate derived SEIR parameters from the input parameterization."""


###CTM from ..numerical_libs import sync_numerical_libs


def calc_Te(Tg, Ts, n, f):
    """Calculate the latent period."""
    num = 2.0 * n * f / (n + 1.0) * Tg - Ts
    den = 2.0 * n * f / (n + 1.0) - 1.0
    return num / den


def calc_Reff(m, n, Tg, Te, r):
    """Calculate the effective reproductive number."""
    num = 2.0 * n * r / (n + 1.0) * (Tg - Te) * (1.0 + r * Te / m) ** m
    den = 1.0 - (1.0 + 2.0 * r / (n + 1.0) * (Tg - Te)) ** (-n)
    return num / den


def calc_Ti(Te, Tg, n):
    """Calcuate the infectious period."""
    return (Tg - Te) * 2.0 * n / (n + 1.0)


def calc_beta(Te):
    """Derive beta from Te."""
    return 1.0 / Te


def calc_gamma(Ti):
    """Derive gamma from Ti."""
    return 1.0 / Ti


###CTM @sync_numerical_libs
def add_derived_params(epi_params, structure_cfg):
    """Add the derived params that are calculated from the rerolled ones."""
    epi_params["Te"] = calc_Te(
        epi_params["Tg"],
        epi_params["Ts"],
        structure_cfg["E_gamma_k"],
        epi_params["frac_trans_before_sym"],
    )
    epi_params["Ti"] = calc_Ti(epi_params["Te"], epi_params["Tg"], structure_cfg["E_gamma_k"])
    # r = xp.log(2.0) / epi_params["D"]
    # epi_params["R0"] = calc_Reff(
    #    structure_cfg["I_gamma_k"],
    #    structure_cfg["E_gamma_k"],
    #    epi_params["Tg"],
    #    epi_params["Te"],
    #    r,
    # )

    epi_params["SIGMA"] = 1.0 / epi_params["Te"]
    epi_params["GAMMA"] = 1.0 / epi_params["Ti"]
    # epi_params["BETA"] = epi_params["R0"] * epi_params["GAMMA"]
    epi_params["SYM_FRAC"] = 1.0 - epi_params["ASYM_FRAC"]
    epi_params["THETA"] = 1.0 / epi_params["H_TIME"]
    epi_params["GAMMA_H"] = 1.0 / epi_params["I_TO_H_TIME"]
    return epi_params
