"""Method of smoothing data w/ splines. Based of a GAM from mgcv with a cr() basis."""
import logging
import warnings
from collections import defaultdict

import tqdm
###CTM from joblib import Memory

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END

# from .read_config import bucky_cfg

dtype = xp.float32

# memory = Memory(bucky_cfg["cache_dir"], verbose=0, mmap_mode="r")


# @memory.cache
def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives."""
    h = knots[:, 1:] - knots[:, :-1]
    diag = (h[:, :-1] + h[:, 1:]) / 3.0
    ul_diag = h[:, 1:-1] / 6.0

    d = xp.zeros((knots.shape[0], knots.shape[1] - 2, knots.shape[1]), dtype=dtype)
    for i in range(knots.shape[1] - 2):
        d[:, i, i] = 1.0 / h[:, i]
        d[:, i, i + 2] = 1.0 / h[:, i + 1]
        d[:, i, i + 1] = -d[:, i, i] - d[:, i, i + 2]

    A = xp.zeros((knots.shape[0], knots.shape[1] - 2, knots.shape[1] - 2), dtype=dtype)
    for i in range(knots.shape[0]):
        A[i] += xp.diag(ul_diag[i], -1) + xp.diag(diag[i]) + xp.diag(ul_diag[i], 1)

    fm = xp.linalg.solve(A, d)

    # using solve_banded would be ideal here but it's not in cupy atm...
    # fm = linalg.solve_banded((1, 1), banded_b, d)

    full_f = xp.hstack(
        [
            xp.zeros((knots.shape[0], 1, knots.shape[1])),
            fm,
            xp.zeros((knots.shape[0], 1, knots.shape[1])),
        ],
    )

    s = xp.einsum("ikj,ikl->ijl", d, fm)

    return full_f, s


def _find_knots_lower_bounds(x, knots):
    """Find the lower bound for the knots."""
    lb = xp.empty(x.shape, dtype=int)
    for i in range(knots.shape[0]):
        lb[i] = xp.searchsorted(knots[i], x[i]) - 1
    lb[lb == -1] = 0
    lb[lb == knots.shape[1] - 1] = knots.shape[1] - 2

    return lb


def _compute_base_functions(x, knots):
    """Return base functions for the spline basis."""
    j = _find_knots_lower_bounds(x, knots)

    h = knots[:, 1:] - knots[:, :-1]
    hj = xp.take_along_axis(h, j, axis=1)
    xj1_x = xp.take_along_axis(knots, j + 1, axis=1) - x
    x_xj = x - xp.take_along_axis(knots, j, axis=1)

    ajm = xj1_x / hj
    ajp = x_xj / hj

    # cjm_3 = xj1_x * xj1_x * xj1_x / (6.0 * hj)
    cjm_3 = xj1_x ** 3 / (6.0 * hj)
    cjm_3[x > xp.max(knots, axis=1, keepdims=True)] = 0.0
    cjm_1 = hj * xj1_x / 6.0
    cjm = cjm_3 - cjm_1

    # cjp_3 = x_xj * x_xj * x_xj / (6.0 * hj)
    cjp_3 = x_xj ** 3 / (6.0 * hj)
    cjm_3[x < xp.min(knots, axis=1, keepdims=True)] = 0.0
    cjp_1 = hj * x_xj / 6.0
    cjp = cjp_3 - cjp_1

    return ajm, ajp, cjm, cjp, j


def nunique(arr, axis=-1):
    """Return the number of uniq values along a given axis."""
    arr_sorted = xp.sort(arr, axis=axis)
    n_not_uniq = (xp.diff(arr_sorted, axis=axis) == 0).sum(1)
    return arr.shape[axis] - n_not_uniq


# @memory.cache
def _get_free_crs_dmatrix(x, knots):
    """Builds an unconstrained cubic regression spline design matrix."""
    knots_dict = {}
    # find the uniques sets of knots so we don't do alot of redundant work
    # batch_u_knots, u_knots_x_map = xp.unique(knots, return_inverse=True, axis=0)

    # enforce uniqueness of knots for set of knots still in x

    knots_dict = defaultdict(list)
    knots_dict_map = defaultdict(list)
    for i, k in enumerate(knots):
        u_knots = xp.unique(k)
        knots_dict[u_knots.size].append(u_knots)
        knots_dict_map[u_knots.size].append(i)

    for n_knots in knots_dict:
        knots_dict[n_knots] = xp.stack(knots_dict[n_knots])
        knots_dict_map[n_knots] = xp.array(knots_dict_map[n_knots])

    # handle each basis with n knots seperately
    dm_dict = {}
    for n in knots_dict:
        if n < 3:
            continue
        ajm, ajp, cjm, cjp, j = _compute_base_functions(x[knots_dict_map[n]], knots_dict[n])
        j1 = j + 1
        f, s = _get_natural_f(knots_dict[n])
        # dmt = ajm * i[j, :].T + ajp * i[j1, :].T + cjm * f[j, :].T + cjp * f[j1, :].T

        eye = xp.identity(n, dtype=dtype)
        if True:  # pylint: disable=using-constant-test
            # if we're using cupy we cant batch it b/c it will build an intermediate array in mem that is HUGE
            # this if statement should eventually be a version check once it's fixed
            dm = xp.empty(x[knots_dict_map[n]].shape + (n,), dtype=dtype)
            for i in range(dm.shape[0]):
                dm[i] = xp.einsum("j,jk->jk", ajm[i], eye[j[i], :])
                dm[i] += xp.einsum("j,jk->jk", ajp[i], eye[j1[i], :])
                dm[i] += xp.einsum("j,jk->jk", cjm[i], f[i][j[i], :])
                dm[i] += xp.einsum("j,jk->jk", cjp[i], f[i][j1[i], :])
        else:
            dm = xp.einsum("ij,ijk->ijk", ajm, eye[j, :])
            dm += xp.einsum("ij,ijk->ijk", ajp, eye[j1, :])
            dm += xp.einsum("ij,iijk->ijk", cjm, f[:, j, :])
            dm += xp.einsum("ij,iijk->ijk", cjp, f[:, j1, :])
        dm_dict[n] = dm

    return dm_dict, knots_dict_map, s


# @memory.cache
def _absorb_constraints(design_matrix, constraints, pen=None):
    """Apply constraints to the design matrix."""
    m = constraints.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        q, _ = xp.linalg.qr(xp.swapaxes(constraints, 1, 2), mode="complete")
    ret = xp.einsum("ijk,ikl->ijl", design_matrix, q[..., m:])
    tmp = xp.einsum("ijk,ikl->ijl", pen, q[..., m:])
    pen_ret = xp.swapaxes(q[..., m:], 1, 2) @ tmp
    return ret, pen_ret


# @memory.cache
def _cr(x, df, center=True):
    """Python version of the R lib mgcv function cr()."""

    # TODO make df settable to a vector
    n_constraints = 0
    if center:
        n_constraints = 1

    n_inner_knots = df - 2 + n_constraints

    # _get_all_sorted_knots from patsy
    # TODO add lower_bound param, well need to mask those values out the x array too
    lower_bound = xp.min(x, axis=-1)
    upper_bound = xp.max(x, axis=-1)
    inner_knots_q = xp.linspace(0, 100, n_inner_knots + 2, dtype=dtype)[1:-1]
    inner_knots = xp.asarray(xp.percentile(x, inner_knots_q, axis=-1))
    all_knots = xp.vstack([lower_bound, inner_knots, upper_bound]).T
    # all_knots = xp.unique(all_knots)

    dm_dict, dict_x_knot_map, pen = _get_free_crs_dmatrix(x, all_knots)
    if center:
        for n, dm in dm_dict.items():
            constraint = dm.mean(axis=1).reshape((dm.shape[0], 1, dm.shape[2]))
            dm_dict[n], pen_out = _absorb_constraints(  # pylint: disable=unnecessary-dict-index-lookup
                dm,
                constraint,
                pen,
            )

    return dm_dict, dict_x_knot_map, pen_out


class log_link:
    """Class for log link functions."""

    ###CTM @staticmethod
    def g(mu):
        """Log link - $g$."""
        return xp.log(mu + 1.0e-12)

    ###CTM @staticmethod
    def mu(eta):
        r"""Log link - $\mu$."""
        return xp.exp(eta) + 1.0e-12

    ###CTM @staticmethod
    def g_prime(mu):
        """Log link - $g'$."""
        return 1.0 / (mu + 1.0e-12)


class identity_link:
    """Class for idenity link functions."""

    ###CTM @staticmethod
    def g(mu):
        """Id link - $g$."""
        return mu

    ###CTM @staticmethod
    def mu(eta):
        r"""Id link - $\mu$."""
        return eta

    ###CTM @staticmethod
    def g_prime(mu):
        """Id link - $g'$."""
        return xp.ones_like(mu)


def make_DP(x):
    """Add I to a batch of matrices (...,M,M) until all are positive-definite (and cholesky-able)."""
    # test sym: (make optional?)
    # xp.allclose(x,xp.swapaxes(x.conj(),1,2))
    eye = xp.eye(x[0].shape[-1])[None, ...]
    eye_facs = xp.zeros((x.shape[0],))
    complete = xp.full((x.shape[0],), False, dtype=bool)
    current_exp = -8.0
    while True:
        eye_fac = 10.0**current_exp
        eye_facs[~complete] = 10.0 ** (current_exp + 1)
        complete[~complete] = xp.all(xp.linalg.eigvalsh(x[~complete] + eye_fac * eye) > 1.0e-12, axis=1)
        current_exp += 1.0
        if current_exp > 20.0:
            raise ValueError
        if xp.all(complete):
            return x + eye_facs[:, None, None] * eye


def PIRLS(
    x,
    y,
    alp,
    pen,
    tol=1.0e-7,
    dist="g",
    max_it=10000,
    w=None,
    gamma=1.0,
    tqdm_label="PIRLS",
    fixed_lam=False,
    ret_beta=False,
    bootstrap=False,
):
    """Penalized iterativly reweighted least squares."""
    if dist == "g":
        link = identity_link()
        V_func = xp.ones_like
    elif dist == "p":
        logging.warning("Poisson link functions are WIP")
        link = log_link()
        V_func = xp.copy  # lambda x: x
        y = xp.clip(y, a_min=1e-6, a_max=None)
    y_all = y
    x_all = x
    pen_all = pen
    step_size_all = 1.0 * xp.ones(x_all.shape[0])
    beta_k_all = ridge(x_all, link.g(y_all), alp=alp)
    var_beta_k_all = xp.empty((x_all.shape[0], x_all.shape[2], x_all.shape[2]))
    mu_k_all = y_all.copy()
    alp_k_all = xp.full((x_all.shape[0],), alp)
    lp_k_all = xp.einsum("bij,bj->bi", x_all, beta_k_all)
    complete = xp.full((y_all.shape[0],), False, dtype=bool)
    vg_all = xp.full((y_all.shape[0],), xp.inf)
    it_since_step_all = xp.zeros((y_all.shape[0],))
    it = 0
    bar_format = "{desc}: {percentage:3.0f}% converged |{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
    pbar = tqdm.tqdm(desc=tqdm_label, total=y_all.shape[0], bar_format=bar_format, dynamic_ncols=True)
    while True:  # pylint: disable=too-many-nested-blocks
        x = x_all[~complete]
        y = y_all[~complete]
        pen = pen_all[~complete]
        step_size = step_size_all[~complete]
        beta_k = beta_k_all[~complete]
        # var_beta_k = var_beta_k_all[~complete]
        mu_k = mu_k_all[~complete]
        alp_k = alp_k_all[~complete]
        lp_k = lp_k_all[~complete]
        it_since_step = it_since_step_all[~complete]
        # vg_k = vg_all[~complete]
        if ~xp.all(xp.isfinite(mu_k) & xp.isfinite(link.g_prime(mu_k))):
            div_mask = ~xp.all(xp.isfinite(mu_k) & xp.isfinite(link.g_prime(mu_k)), axis=1)
            mu_k[div_mask] = xp.clip(mu_k[div_mask], a_min=1.0e-12, a_max=1.0e12)
            lp_k = link.g(mu_k)

        V = V_func(mu_k)
        g_prime = link.g_prime(mu_k)
        if w is None:
            data_w = 1.0
        else:
            data_w = w[~complete]
        w_k_diag = data_w / (V * g_prime * g_prime)
        # mask = (xp.abs(w_k_diag) >= xp.sqrt(1e-15)) * xp.isfinite(w_k_diag)
        z = link.g_prime(mu_k) * (y - mu_k) + lp_k
        y_tilde = xp.sqrt(w_k_diag) * z
        x_tilde = xp.sqrt(w_k_diag)[..., None] * x
        _, beta_new, var_beta_new, alp_new, vg_new = opt_lam(
            x_tilde,
            y_tilde,
            alp=alp_k,
            pen=pen,
            step_size=step_size,
            tol=tol,
            max_it=1,
            gamma=gamma,
            fixed_lam=fixed_lam,
        )
        diff = xp.sqrt(xp.sum((beta_k - beta_new) ** 2, axis=1)) / xp.sqrt(xp.sum(beta_new**2, axis=1))
        alp_diff = xp.abs(alp_k - alp_new) / alp_new

        batch_beta = step_size[..., None] * beta_new + (1.0 - step_size[..., None]) * beta_k
        beta_k_all[~complete] = batch_beta

        var_beta_k_all[~complete] = var_beta_new
        alp_k_all[~complete] = step_size * alp_new + (1.0 - step_size) * alp_k
        lp_new = xp.einsum("bij,bj->bi", x, batch_beta)
        mu_k_all[~complete] = link.mu(lp_new)
        lp_k_all[~complete] = lp_new
        step_size_all[~complete] = step_size

        step_mask = (vg_new > vg_all[~complete]) & (it_since_step > 10)
        vg_all[~complete] = vg_new
        if xp.any(step_mask) and (it > 50):
            step_size[step_mask] = 0.5 * step_size[step_mask]
            step_size_all[~complete] = step_size
            step_stop = step_size < 0.5**10
            it_since_step[step_mask] = 0
        else:
            step_stop = False

        it_since_step_all[~complete] = it_since_step + 1

        if it > 0:
            batch_complete = ((diff < tol) & ((alp_diff < tol) | (alp_new < tol))) | xp.isnan(diff) | step_stop
            # batch_complete = ((diff < tol) & (alp_diff < tol)) | xp.isnan(diff) | step_stop

            # ignore stepsize if last values met tol (so var_beta is consistant)
            if xp.any(batch_complete):
                batch_complete_all = xp.full_like(complete, False)
                batch_complete_all[~complete] = batch_complete
                beta_k_all[batch_complete_all] = beta_new[batch_complete]
                alp_k_all[batch_complete_all] = alp_new[batch_complete]

            complete[~complete] = batch_complete
        # print("pirls", it, xp.sum(complete), xp.sum(~complete))
        pbar.set_postfix_str(
            "iter="
            + str(it)
            + ", med_err/tol="
            + str(xp.round(xp.median(diff) / tol, 2))
            + ", med_alp_err/tol="
            + str(xp.round(xp.median(alp_diff) / tol, 2)),
        )
        pbar.update(xp.to_cpu(xp.sum(complete)) - pbar.n)
        it = it + 1

        if xp.all(complete) | (it > max_it):

            pbar.close()
            if ret_beta:
                return mu_k_all, alp_k_all, beta_k_all, var_beta_k_all
            else:
                if bootstrap:
                    # pylint: disable=import-outside-toplevel
                    logging.warning("Testing GAM bootstrap, this is currently broken and unstable")
                    # var_beta_DP = make_DP(var_beta_k_all)
                    var_beta_DP = var_beta_k_all
                    for k in range(100):
                        if link == "g":
                            continue
                        i = 50 + k
                        var_beta = var_beta_DP[i]
                        beta = beta_k_all[i]
                        lam = alp_k_all[i]
                        bs_beta_k = [
                            beta,
                        ]
                        bs_var_beta_k = [
                            var_beta,
                        ]
                        bs_lam_k = [
                            lam,
                        ]
                        for _ in range(20):
                            with warnings.catch_warnings():
                                warnings.simplefilter(action="ignore", category=FutureWarning)
                                beta_rand = xp.random.multivariate_normal(  # pylint: disable=unexpected-keyword-arg
                                    beta,
                                    var_beta,
                                    method="svd",
                                )
                            lp_rand = xp.einsum("ij,j->i", x[0], beta_rand)
                            _, bs_lam, bs_beta, bs_var_beta = PIRLS(
                                x=x_all[i][None, :],
                                y=lp_rand[None, :],
                                alp=alp,
                                pen=pen_all[i][None, ...],
                                tol=tol,
                                dist=dist,
                                max_it=max_it,
                                w=w,
                                gamma=gamma,
                                tqdm_label="BS",
                                ret_beta=True,
                            )
                            _, _, bs_beta, bs_var_beta = PIRLS(
                                x=x_all[i][None, :],
                                y=y_all[i][None, ...],
                                alp=bs_lam,
                                pen=pen_all[i][None, ...],
                                tol=tol,
                                dist=dist,
                                max_it=max_it,
                                w=w,
                                gamma=gamma,
                                tqdm_label="BS",
                                ret_beta=True,
                                fixed_lam=True,
                            )
                            # bs_var_beta = make_DP(bs_var_beta)
                            bs_beta_k.append(bs_beta[0])
                            bs_var_beta_k.append(bs_var_beta[0])
                            bs_lam_k.append(bs_lam[0])
                            beta = bs_beta[0]
                            var_beta = bs_var_beta[0]
                            lam = bs_lam[0]
                        lp_rands = []
                        for _ in range(1000):
                            j = xp.random.randint(20)
                            with warnings.catch_warnings():
                                warnings.simplefilter(action="ignore", category=FutureWarning)
                                beta_rand = xp.random.multivariate_normal(  # pylint: disable=unexpected-keyword-arg
                                    bs_beta_k[int(j)],
                                    bs_var_beta_k[int(j)],
                                    method="svd",
                                )
                            lp_rand = xp.einsum("ij,j->i", x[0], beta_rand)
                            lp_rands.append(lp_rand)
                        lp_rand = xp.stack(lp_rands)
                        a = xp.quantile(lp_rand, (0.25, 0.75), axis=0)
                        b = xp.quantile(lp_rand, (0.025, 0.975), axis=0)
                        import matplotlib.pyplot as plt

                        plt.plot(xp.to_cpu(lp_rand.T), alpha=0.1)
                        plt.plot(xp.to_cpu(a.T), color="red")
                        plt.plot(xp.to_cpu(b.T), color="red")
                        plt.plot(xp.to_cpu(lp_k_all[i]), linewidth=3, color="black")
                        plt.show()

                return mu_k_all


def ridge(x, y, alp=0.0):
    """Calculate the exact soln to the ridge regression of the weights for basis x that fit batched data y."""
    xtx = xp.einsum("ijk,ijl->ikl", x, x)
    t1 = xp.linalg.inv(alp * xp.tile(xp.identity(xtx.shape[1]), (xtx.shape[0], 1, 1)) + xtx)

    t2 = xp.einsum("ijk,ij->ik", x, y)

    w = xp.einsum("ijk,ij->ik", t1, t2)
    return w


###CTM @sync_numerical_libs
def lin_reg(y, x=None, alp=0.0, quad=False, return_fit=True):
    """Calculate exact soln for batched linear regression and return either weights or fitted values."""
    if x is None:
        x = xp.arange(y.shape[1], dtype=float)
        x = xp.tile(x, (y.shape[0], 1))
    basis_list = [xp.ones_like(x), x]
    if quad:
        basis_list.append(x**2)
    basis = xp.stack(basis_list, axis=1).swapaxes(1, 2)

    w = ridge(basis, y, alp)

    if return_fit:
        ret = xp.sum((w[:, None, :] * basis), axis=-1)
        return ret
    else:
        return w


###CTM @sync_numerical_libs
def logistic_fit(y, x_out, x=None, alp=0.6, t0_max=200, L=None):
    """WIP Fit a logistic function to batched y."""
    # TODO this is WIP
    if x is None:
        x = xp.arange(y.shape[1], dtype=float)
        x = xp.tile(x, (y.shape[0], 1))
    slopes = xp.gradient(y, axis=1)
    ratio = slopes / y
    w = lin_reg(ratio, xp.array(y), alp=alp)
    k = w[:, 0]
    if L is None:
        L = -k / w[:, 1]
    else:
        L = xp.full_like(k, L)
    test = L[:, None, None] / (
        1.0 + xp.exp(-k[:, None, None] * (xp.arange(y.shape[1])[None, :] - xp.arange(t0_max)[:, None])[None, :, :])
    )
    err = xp.nansum((test - y[:, None, :]) ** 2.0, axis=-1)
    t0 = xp.argmin(err, axis=-1)

    y_out = L[:, None] / (1.0 + xp.exp(-k[:, None] * (x_out - t0[:, None])))
    return y_out


# @memory.cache
def opt_lam(x, y, alp=0.6, pen=None, min_lam=0.1, step_size=None, tol=1e-3, max_it=100, gamma=1.0, fixed_lam=False):
    """Calculate the exact soln to the ridge regression of the weights for basis x that fit data y."""

    xtx_all = xp.einsum("ijk,ijl->ikl", x, x)

    if "ndarray" in str(type(alp)):
        lam_all = alp.copy()
    else:
        lam_all = xp.full((x.shape[0],), alp)

    if pen is None:
        raise NotImplementedError
        # d = xp.ones(x.shape[-1])
        # d[0] = 0.0
        # d[1] = 0.0
        # pen_mat_all = xp.tile(xp.diag(d), (xtx.shape[0], 1, 1))
    else:
        pen_mat_all = xp.pad(pen, ((0, 0), (2, 0), (2, 0)))

    if step_size is None:
        step_size = xp.ones_like(lam_all)

    q_all = xp.empty_like(x)
    r_all = xp.empty((x.shape[0], x.shape[-1], x.shape[-1]))
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        q_all, r_all = xp.linalg.qr(x.astype(xp.float64))

    complete = xp.full((y.shape[0],), False, dtype=bool)
    Vg_out = xp.empty((y.shape[0],))
    y_out = xp.empty(y.shape)
    beta_out = xp.empty((x.shape[0], x.shape[2]))
    var_beta_out = xp.empty((x.shape[0], x.shape[2], x.shape[2]))
    x_in = x.copy()
    y_in = y.copy()

    it = 0
    while True:
        lam = lam_all[~complete]
        pen_mat = pen_mat_all[~complete]
        q = q_all[~complete]
        r = r_all[~complete]
        xtx = xtx_all[~complete]
        x = x_in[~complete]
        y = y_in[~complete]

        s = (min_lam + lam[..., None, None]) * pen_mat
        t1 = xp.linalg.inv(xtx + s)

        # t2 = xp.einsum("ijk,ij->ik", x, y)
        # w = xp.einsum("ijk,ij->ik", t1, t2)

        a = xp.einsum("ijk,ikl,iml->ijm", x, t1, x)

        s_DP = make_DP(s.astype(xp.float64))
        b = xp.linalg.cholesky(s_DP)

        if xp.any(xp.isnan(b)):
            logging.error("batch cholsky hit nan")
            raise ValueError
            # from IPython import embed
            # embed()

        aug = xp.hstack((r, b))

        # NB: cupy's batched svd is giving incorrect results for float64?
        u = xp.empty((aug.shape[0], aug.shape[1], aug.shape[1]))
        d_diag = xp.empty((aug.shape[0], aug.shape[2]))
        vt = xp.empty((aug.shape[0], aug.shape[2], aug.shape[2]))
        u, d_diag, vt = xp.linalg.svd(aug.astype(xp.float64))

        # eps = xp.finfo(x.dtype).eps
        # check D isn't rank deficient here
        # if xp.any(d_diag < (d_diag[:, 0] * xp.sqrt(eps))[..., None]):
        #    # TODO if they are we can remove them but for now just throw an err
        #    raise ValueError

        u1 = u[:, : r.shape[1], : r.shape[2]]
        trA = xp.einsum("bij,bij->b", u1, u1)

        y1 = xp.einsum("bji,bkj,bk->bi", u1, q, y)

        invd_diag = 1.0 / d_diag

        m = invd_diag[:, None, :] * (vt @ s @ xp.swapaxes(vt, 1, 2)) * invd_diag[:, :, None]

        k = xp.einsum("bij,bkj,bkl->bil", m, u1, u1)

        y1t = y1[:, None, :]

        dalpdrho = 2.0 * lam[..., None, None] * (y1t @ m @ y1[..., None] - y1t @ k @ y1[..., None])
        d2alpdrho = (
            2.0
            * lam[..., None, None]
            * lam[..., None, None]
            * y1t
            @ (2.0 * m @ k - 2.0 * m @ m + k @ m)
            @ y1[..., None]
            + dalpdrho
        )

        n = x.shape[1]

        dtrAdrho = xp.einsum("b,bii->b", -lam, k)
        d2trAd2rho = 2.0 * xp.einsum("b,b,bii->b", lam, lam, m @ k) + dtrAdrho
        ddeltadrho = -gamma * dtrAdrho
        d2deltad2rho = -gamma * d2trAd2rho  # todo double check

        delta = n - gamma * trA
        fitted_y = xp.einsum("bij,bj->bi", a, y)
        alpha = xp.sum((y - fitted_y) ** 2.0, axis=-1)

        Vg = n * alpha / delta / delta
        dVgdrho = n / delta / delta * dalpdrho[:, 0, 0] - 2.0 * n * alpha / delta / delta / delta * ddeltadrho
        d2Vgd2rho = (
            -2.0 * n / delta / delta / delta * ddeltadrho * dalpdrho[:, 0, 0]
            + n / delta / delta * d2alpdrho[:, 0, 0]
            - 2.0 * n / delta / delta / delta * dalpdrho[:, 0, 0] * ddeltadrho
            + 6.0 * n * alpha / (delta**4) * ddeltadrho * ddeltadrho
            - 2.0 * n * alpha / (delta**3) * d2deltad2rho
        )

        rho = xp.log(lam)
        drho = dVgdrho / d2Vgd2rho
        nanmask = xp.isnan(drho)
        drho[nanmask] = 0.0
        drho = xp.clip(drho, a_min=-2.0, a_max=2.0)
        new_rho = rho - drho
        y_out[~complete] = fitted_y
        lam_all[~complete] = xp.exp(new_rho)
        beta_out[~complete] = ((xp.swapaxes(vt, 1, 2) * invd_diag[:, None]) @ y1[..., None])[:, :, 0]
        Vg_out[~complete] = Vg
        var_beta_out[~complete] = xp.einsum(
            "bij,bj,bjk->bik",
            xp.swapaxes(vt, 1, 2),
            invd_diag**2,
            vt,
        )  # TODO double check this
        if (it > 0) or fixed_lam:
            batch_complete = xp.abs(drho / rho) < tol
            complete[~complete] = batch_complete
            if fixed_lam:
                complete = xp.full_like(complete, True)

        if xp.sum(~complete) < 1:
            break
        it += 1
        if it >= max_it:
            break

    return y_out, beta_out, var_beta_out, lam_all, Vg_out


# @memory.cache(ignore=["label"])
###CTM @sync_numerical_libs
def fit(
    y,
    x=None,
    df=10,
    alp=2.0,
    dist="g",
    standardize=True,
    w=None,
    gamma=1.4,
    tol=1.0e-7,
    clip=(None, None),
    label="fit",
    bootstrap=False,
):
    """Perform fit of natural cubic splines to the vector y, return the smoothed y."""
    # TODO handle df and alp as vectors

    # standardize inputs
    if x is None:
        x = xp.arange(0, y.shape[1])
        x = xp.tile(x, (y.shape[0], 1))

    if standardize:
        if dist == "g":
            y_mean = xp.mean(y, axis=1, keepdims=True)
            y_var = xp.var(y, axis=1, keepdims=True)
            # y_range = xp.max(y, axis=1, keepdims=True) - xp.min(y, axis=1, keepdims=True)
            y_in = (y - y_mean) / (y_var + 1e-10)
            # y_in = (y - y_mean) / (y_range + 1.e-6)
        elif dist == "p":
            # y_var = xp.var(y, axis=1, keepdims=True)
            # y_in = y / (y_var + 1e-10)
            # y_in = 1. - xp.mean(y_in, axis=1, keepdims=True)
            y_in = xp.sqrt(y + 1e-2)
    else:
        y_in = y

    x_in = x
    y_in = y

    bs_dict, x_map, pen = _cr(x_in, df=df, center=True)
    y_fit = xp.empty(y_in.shape)
    for n, bs in bs_dict.items():
        full_bs = xp.dstack([xp.ones((bs.shape[0], bs.shape[1], 1)), x_in[..., None], bs])

        with xp.optimize_kernels():
            y_fit = PIRLS(
                full_bs,
                y_in[x_map[n]],
                alp=alp,
                pen=pen,
                dist=dist,
                w=w,
                gamma=gamma,
                tol=tol,
                tqdm_label=label,
                bootstrap=bootstrap,
            )
        y_fit[x_map[n]] = y_fit

    # rescale the standaridized output
    if standardize:
        y_out = y_fit * y_var + y_mean
    else:
        y_out = y_fit

    # Clip output if requested
    if (clip[0] is not None) or (clip[1] is not None):
        y_out = xp.clip(y_out, a_min=clip[0], a_max=clip[1])

    return y_out
