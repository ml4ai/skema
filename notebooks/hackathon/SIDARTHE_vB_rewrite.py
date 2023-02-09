output = SIDARTHE_model(y, t, alpha, beta, gamma, delta, epsilon, mu, zeta, lamb, eta, rho, theta, kappa, nu, xi, sigma, tau)

def SIDARTHE_model(y, t, alpha, beta, gamma, delta, epsilon, mu, zeta, lamb, eta, rho, theta, kappa, nu, xi, sigma, tau):
    S, I, D, A, R, T, H, E = y
    dSdt = -S*(alpha*I + beta*D + gamma*A + delta*R)
    dIdt = S*(alpha*I + beta*D + gamma*A + delta*R) - (epsilon + zeta + lamb)*I
    dDdt = epsilon*I - (eta + rho)*D
    dAdt = zeta*I - (theta + mu + kappa)*A
    dRdt = eta*D + theta*A - (nu + xi)*R
    dTdt = mu*A + nu*R - (sigma + tau)*T
    dHdt = lamb*I + rho*D + kappa*A + xi*R + sigma*T
    dEdt = tau*T
    
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt