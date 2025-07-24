import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.0       # Total time
N = 1000      # Number of steps
dt = T / N
t = np.linspace(0, T, N)
sigma = 0.5   # Noise strength
y = np.zeros(N)
y[0] = 5      # Initial condition

# Simulate Brownian increments
dW = np.random.normal(0, np.sqrt(dt), size=N-1)

# Euler-Maruyama iteration
for i in range(1, N):
    y[i] = y[i-1] + (-2 * y[i-1]) * dt + sigma * dW[i-1]

# Plot
plt.plot(t, y)
plt.title('SDE Solution: Ornstein-Uhlenbeck Process')
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.show()
