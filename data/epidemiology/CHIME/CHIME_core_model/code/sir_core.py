def sir(s, i, r, beta, gamma, n): # sir: ID 2 s: ID 14, i: ID 15, r: ID 16, beta: ID 17, gamma: ID 18, n: ID 19
    """
    The SIR model, one time step
    :param s: Current amount of individuals that are susceptible
    :param i: Current amount of individuals that are infectious
    :param r: Current amount of individuals that are recovered
    :param beta: The rate of exposure of individuals to persons infected with COVID-19
    :param gamma: Rate of recovery for infected individuals
    :param n: Total population size
    :return:
    """
    s_n = (-beta * s * i) + s  # s_n: ID 20 = ID 17 * ID 14 * ID 15 + ID 14
    i_n = (beta * s * i - gamma * i) + i # i_n: ID 21 = ID 17 * ID 14 * ID 15 - ID 18 * ID 15 + ID 15
    r_n = gamma * i + r  # r_n: ID 22 = ID 18 * ID 15 + ID 16

    scale = n / (s_n + i_n + r_n) # scale: ID 23 = ID 19 / (ID 20 + ID 21 + ID 22)

    s = s_n * scale # ID 14 = ID 20 * ID 23 
    i = i_n * scale # ID 15 = ID 21 * ID 23
    r = r_n * scale # ID 16 = ID 22 * ID 23
    return s, i, r  # return ID 14, ID 15, ID 16 