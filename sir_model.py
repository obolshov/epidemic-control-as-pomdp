import numpy as np
import matplotlib.pyplot as plt

def run_sir_model(N, I0, R0, beta, gamma, days):
    """
    Runs a simple SIR model simulation.
    
    :param N: Total population
    :param I0: Initial number of infected individuals
    :param R0: Initial number of recovered individuals
    :param beta: Transmission rate
    :param gamma: Recovery rate
    :param days: Number of days to simulate
    :return: A tuple of (t, S, I, R) arrays
    """
    S0 = N - I0 - R0
    
    S, I, R = [S0], [I0], [R0]
    
    t = np.linspace(0, days, days + 1)
    
    for _ in range(days):
        S_current, I_current, R_current = S[-1], I[-1], R[-1]
        
        new_infections = (beta * S_current * I_current) / N
        new_recoveries = gamma * I_current
        
        S_next = S_current - new_infections
        I_next = I_current + new_infections - new_recoveries
        R_next = R_current + new_recoveries
        
        S.append(S_next)
        I.append(I_next)
        R.append(R_next)
        
    return t, np.array(S), np.array(I), np.array(R)

if __name__ == '__main__':
    N = 1000 # Total population
    I0, R0 = 1, 0 # Initial number of infected and recovered individuals
    beta, gamma = 0.4, 0.1 # Transmission and recovery rates
    days = 160 # Number of days to simulate

    t, S, I, R = run_sir_model(N, I0, R0, beta, gamma, days)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', label='Susceptible (S)')
    plt.plot(t, I, 'r', label='Infected (I)')
    plt.plot(t, R, 'g', label='Recovered (R)')
    plt.title('SIR Model Simulation')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of people')
    plt.legend()
    plt.grid(True)
    plt.show()
