def SEIR_model(y, t, N, alpha, beta, gamma, epsilon, mu):
    S, E, I, R = y
    dSdt = -mu*N -beta * S * I - mu*S
    dEdt = beta * S * I - (mu + epsilon) * E
    dIdt = epsilon * E - (gamma + mu) * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

SEIR_model(y, t, N, alpha, beta, gamma, epsilon, mu)