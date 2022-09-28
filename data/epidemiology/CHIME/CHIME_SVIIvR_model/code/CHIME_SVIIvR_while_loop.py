# import sys
# from csv import DictWriter, QUOTE_NONNUMERIC

def get_beta(intrinsic_growth_rate, gamma, susceptible, relative_contact_rate):
    inv_contact_rate = 1.0 - relative_contact_rate  
    updated_growth_rate = intrinsic_growth_rate + gamma  
    beta = updated_growth_rate / susceptible * inv_contact_rate  

    return beta

def get_growth_rate(doubling_time):
    if doubling_time == 0:  
        growth_rate = 0  
    else:
        growth_rate = 2.0 ** (1.0 / doubling_time) - 1.0  

    return growth_rate

def sir(s, v, i, i_v, r, vaccination_rate, beta, gamma_unvaccinated, gamma_vaccinated, vaccine_efficacy, n):
    s_n = (-beta * s * i - beta * s * i_v - vaccination_rate * s) + s  
    v_n = (vaccination_rate * s - beta * (1 - vaccine_efficacy) * v * i - beta * (1 - vaccine_efficacy) * v * i_v) + v  
    i_n = (beta * s * i + beta * s * i_v - gamma_unvaccinated * i) + i  
    i_v_n = (beta * (1 - vaccine_efficacy) * v * i + beta * (1 - vaccine_efficacy) * v * i_v - gamma_vaccinated * i_v) + i_v  
    r_n = gamma_vaccinated * i_v + gamma_unvaccinated * i + r  

    scale = n / (s_n + v_n + i_n + i_v_n + r_n)  

    s = s_n * scale  
    v = v_n * scale  
    i = i_n * scale  
    i_v = i_v_n * scale  
    r = r_n * scale  
    return s, v, i, i_v, r

def sim_sir(s, v, i, i_v, r, vaccination_rate, gamma_unvaccinated, gamma_vaccinated, vaccine_efficacy, i_day,
            
            N_p, betas, days,  
            d_a, s_a, v_a, i_a, i_v_a, r_a, e_a,
            
            ):
    n = s + v + i + i_v + r  
    d = i_day  


    index = 0  
    p_idx = 0 # NOTE: added
    while p_idx < N_p: # NOTE: changed 
        beta = betas[p_idx]  
        n_days = days[p_idx]  
        d_idx = 0  # NOTE: added 
        while d_idx < n_days: # NOTE: changed  
            d_a[index] = d  
            s_a[index] = s  
            v_a[index] = v  
            i_a[index] = i  
            i_v_a[index] = i_v  
            r_a[index] = r  
            e_a[index] = i + i_v + r  

            index = index + 1 # NOTE: simplified 

            s, v, i, i_v, r = sir(s, v, i, i_v, r, vaccination_rate, beta, gamma_unvaccinated, gamma_vaccinated,
                                  vaccine_efficacy, n)  

            d = d + 1  # NOTE: simplified
            d_idx = d_idx + 1 # NOTE: added 
        p_idx = p_idx + 1 # NOTE: added

    
    d_a[index] = d  
    s_a[index] = s  
    v_a[index] = v  
    i_a[index] = i  
    i_v_a[index] = i_v  
    r_a[index] = r  

    return s, v, i, i_v, r, d_a, s_a, v_a, i_a, i_v_a, r_a, e_a  

def main():
    i_day = 17.0  
    n_days = [14, 90]  
    N_p = 2  
    N_t = sum(n_days) + 1  
    infectious_days_unvaccinated = 14  
    infectious_days_vaccinated = 10  
    relative_contact_rate = [0.0, 0.45]  
    gamma_unvaccinated = 1.0 / infectious_days_unvaccinated  
    gamma_vaccinated = 1.0 / infectious_days_vaccinated  

    
    vaccination_rate = 0.02  
    vaccine_efficacy = 0.85  

    
    policys_betas = [0.0] * N_p  
    policy_days = [0] * N_p  
    d_a = [0.0] * N_t  
    s_a = [0.0] * N_t  
    v_a = [0.0] * N_t  
    i_a = [0.0] * N_t  
    i_v_a = [0.0] * N_t  
    r_a = [0.0] * N_t  
    e_a = [0.0] * N_t  

    
    s_n = 1000  
    v_n = 0  
    i_n = 1  
    i_v_n = 0  
    r_n = 0  

    
    p_idx = 0 # NOTE: added
    while p_idx < N_p:  # NOTE: changed
        doubling_time = 2

        growth_rate = get_growth_rate(doubling_time)  
        beta = get_beta(growth_rate, gamma_unvaccinated, s_n,  
                        relative_contact_rate[p_idx])
        policys_betas[p_idx] = beta  
        policy_days[p_idx] = n_days[p_idx]  
        p_idx = p_idx + 1 # NOTE: changed

    
    s_n, v_n, i_n, i_v_n, r_n, d_a, s_a, v_a, i_a, i_v_a, r_a, e_a \
        = sim_sir(s_n, v_n, i_n, i_v_n, r_n, vaccination_rate, gamma_unvaccinated, gamma_vaccinated, vaccine_efficacy,
                  i_day,  
                  N_p, policys_betas, policy_days,
                  d_a, s_a, v_a, i_a, i_v_a, r_a, e_a)

    return d_a, s_a, v_a, i_a, i_v_a, r_a, e_a  
