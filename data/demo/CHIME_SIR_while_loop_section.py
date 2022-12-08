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

def get_beta(intrinsic_growth_rate, gamma,           # get_beta: ID 0 
             susceptible, relative_contact_rate):    # instrinsic_groth rate: ID 5, gamma: ID 6, susceptible: ID 7, rel_cont_rate: ID 8
    """
    Calculates a rate of exposure given an intrinsic growth rate for COVID-19
    :param intrinsic_growth_rate: Rate of spread of COVID-19 cases
    :param gamma: The expected recovery rate from COVID-19 for infected individuals
    :param susceptible: Current amount of individuals that are susceptible
    :param relative_contact_rate: The relative contact rate amongst individuals in the population
    :return: beta: The rate of exposure of individuals to persons infected with COVID-19
    """
    inv_contact_rate = 1.0 - relative_contact_rate  # inv_contact_rate: ID 9 = 1.0 - ID 8
    updated_growth_rate = intrinsic_growth_rate + gamma  # updated_growth_rate: ID 10 = ID 5 + ID 6
    beta = updated_growth_rate / susceptible * inv_contact_rate # beta: ID 11 = ID 10 / ID 7 * ID 9
 
    return beta


infections_days = 14.0  # ID 55
gamma = 1.0 / infections_days
relative_contact_rate = 0.05  # ID 56
growth_rate = 5
# initial population
s_n = 1000 # ID 65
i_n = 1  # ID 66
r_n = 1  # ID 67
n = s_n + i_n + r_n

beta = get_beta(growth_rate, gamma, s_n,     # Call ID 1
            relative_contact_rate)

s, i, r = sir(s_n, i_n, r_n, beta, gamma, n)
