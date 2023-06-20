def sir(s, v, i, i_v, r, vaccination_rate, beta, gamma_unvaccinated, gamma_vaccinated, vaccine_efficacy, n):
    """
    The SIR model, one time step
    :param s: Current amount of individuals that are susceptible
    :param v: Current amount of individuals that are vaccinated
    :param i: Current amount of individuals that are infectious
    :param i_v: Current amount of vaccinated individuals that are infectious
    :param r: Current amount of individuals that are recovered
    :param beta: The rate of exposure of individuals to persons infected with COVID-19
    :param gamma_unvaccinated: Rate of recovery for infected unvaccinated individuals
    :param gamma_vaccinated: Rate of recovery for infected vaccinated individuals
    :param vaccination_rate: The rate of vaccination of susceptible individuals
    :param vaccine_efficacy: The efficacy of the vaccine
    :param n: Total population size
    :return:
    """
    s_n = (
                      -beta * s * i - beta * s * i_v - vaccination_rate * s) + s  # Update to the amount of individuals that are susceptible ## sir_s_n_exp
    v_n = (vaccination_rate * s - beta * (1 - vaccine_efficacy) * v * i - beta * (
                1 - vaccine_efficacy) * v * i_v) + v  # Update to the amount of individuals that are susceptible ## sir_v_n_exp
    i_n = (
                      beta * s * i + beta * s * i_v - gamma_unvaccinated * i) + i  # Update to the amount of individuals that are infectious ## sir_i_n_exp
    i_v_n = (beta * (1 - vaccine_efficacy) * v * i + beta * (
                1 - vaccine_efficacy) * v * i_v - gamma_vaccinated * i_v) + i_v  # Update to the amount of individuals that are infectious ## sir_i_v_n_exp
    r_n = gamma_vaccinated * i_v + gamma_unvaccinated * i + r  # Update to the amount of individuals that are recovered ## sir_r_n_exp

    scale = n / (
                s_n + v_n + i_n + i_v_n + r_n)  # A scaling factor to compute updated disease variables ## sir_scale_exp

    s = s_n * scale  ## sir_s_exp
    v = v_n * scale  ## sir_v_exp
    i = i_n * scale  ## sir_i_exp
    i_v = i_v_n * scale  ## sir_i_v_exp
    r = r_n * scale  ## sir_r_exp
    return s, v, i, i_v, r