"""Provides a class that contains all the data/info needed to perform one full integration."""
import datetime

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END

# from IPython import embed


# TODO add logging


def norm_Cij(Cij):
    """Sum a stack of contact matrices to get the overall Cij thats symmetrized then normalized."""
    # (n_adm2, n_contact_mats, n_age_grps, n_age_grps)
    _Cij = xp.sum(Cij, axis=1)
    _Cij = (_Cij + xp.swapaxes(_Cij, 1, 2)) / 2.0  # make symmetric as (C + C.T) / 2
    _Cij = _Cij / xp.sum(_Cij, axis=2, keepdims=True)
    return _Cij


class buckyMCInstance:
    """Class that holds basic information and provides helper variables needed to perform an MC."""

    ###CTM @sync_numerical_libs
    def __init__(self, init_date, n_days, Nij, Cij):
        """Initialize MC instance."""

        # Time integration related params
        self.dt = 1.0  # time step for model output (the internal step is adaptive...)
        self.t_max = n_days
        self._integrator_args = {
            "method": "RK23",
            "t_eval": xp.arange(0, self.t_max + self.dt, self.dt),
            "t_span": (0.0, self.t_max),
        }
        self.dates = [str(init_date + datetime.timedelta(days=t)) for t in range(n_days + 1)]
        self.rhs = None

        # Demographic data
        self.Nij = Nij
        self.n_adm2 = self.Nij.shape[-1]

        # Contact Matrices
        self._Cij = xp.broadcast_to(Cij, (self.n_adm2,) + Cij.shape)
        self.baseline_Cij = norm_Cij(self._Cij)

        # Mobility matrix
        self._Aij = None

        # Epi params
        self._epi_params = None

        # State vector
        self._state = None

        # NPI related variables/flag
        self.npi_active = False
        self.npi_params = None

        # Vaccine realted variables/flag
        self.vacc_active = False
        self.vacc_data = None

    def set_tmax(self, t_max):
        """Set a new value for the max integration time."""
        self.t_max = t_max
        self._integrator_args["t_eval"] = xp.arange(0, self.t_max + self.dt, self.dt)
        self._integrator_args["t_span"] = (0.0, self.t_max)

    def add_npi(self, npi_params):
        """Enable dynamic NPIs during the time integration."""
        self.npi_active = True
        self.npi_params = npi_params
        self.npi_params["contact_weights"] = self.npi_params["contact_weights"][..., None, None]

    def add_vacc(self, vacc_data):
        """Enable vaccine simulation during the simulation."""
        self.vacc_active = True
        self.vacc_data = vacc_data

    ###CTM @property
    def integrator_args(self):
        """Dict of standard arguments passed to solve_ivp."""
        return {"fun": self.rhs, "y0": self.state.state.ravel(), "args": (self,), **self._integrator_args}

    ###CTM @property
    def epi_params(self):
        """Epi parameters property."""
        return self._epi_params

    ###CTM @epi_params.setter
    def epi_params(self, v):
        """Epi parameters property setter."""
        self._epi_params = v

    ###CTM @property
    def state(self):
        """State variable property."""
        return self._state

    ###CTM @state.setter
    def state(self, v):
        """State variable setter."""
        self._state = v

    ###CTM @property
    def Aij(self):  # TODO this is time dep w/ npis
        """Adj matrix property."""
        return self._Aij

    ###CTM @Aij.setter
    def Aij(self, v):
        """Adj matrix setter."""
        self._Aij = v

    #
    # define some RHS variables that depend on which features are turned on
    #
    def Cij(self, t):
        """Return the contact matrices after applying time dependent changes."""
        if self.npi_active:
            return norm_Cij(self.npi_params["contact_weights"][t] * self._Cij)
        return self.baseline_Cij

    def BETA_eff(self, t):
        """Return effective value of BETA after time dependent changes."""
        if self.npi_active:
            return self.npi_params["r0_reduct"][t] * self.epi_params["BETA"]
        return self.epi_params["BETA"]

    def S_eff(self, t, y):
        """Return the effective of susceptable (S_ij) after applying time dependent modifications."""
        if self.vacc_active:
            return self.vacc_data.S_eff(y, self.epi_params, t)
        return y.S
