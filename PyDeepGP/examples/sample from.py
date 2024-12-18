import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Define mean function
def m(x):
    coefs = [5, -2.5, -2.4, -0.2, 0.3, 0.04]
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef * (x ** exp)
    return total

# Define kernel function
def k(xs, ys, sigma=0.7, l=0.04):
    # xs and ys are arrays of shape (N,) or (N,1)
    xs = np.array(xs, dtype=float).flatten()
    ys = np.array(ys, dtype=float).flatten()
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    tmp = (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)
    return tmp

# Generate a set of input points for demonstration
X = np.linspace(0, 2.5, 40)  # 100 points from -1 to 1

# Construct the mean vector and covariance matrix for the first layer GP
M = m(X)
KXX = k(X, X)

# Add a small jitter for numerical stability
jitter = 1e-9
KXX += np.eye(len(X)) * jitter

# Sample from the first layer GP
num_samples = 5  # e.g., generate 5 sample functions
L = np.linalg.cholesky(KXX)
Z = np.random.randn(len(X), num_samples)
F1_samples = M[:, None] + L @ Z  # shape: (100, 5)
np.savetxt("samples_GP.csv", F1_samples.T, delimiter=",")
# Plot the first layer GP samples
plt.figure(figsize=(8,4))
for i in range(num_samples):
    plt.plot(X, F1_samples[:, i], alpha=0.7)
plt.title("Samples from the first-layer GP")
plt.xlabel("X")
plt.ylabel("f(X)")
plt.grid(True)
plt.show()

##################################
# Construct a simple Deep GP by using F1_samples as inputs to a second GP.
# This is a simplistic demonstration. In a real DGP, you'd have a learned mapping
# from X to some latent space. Here we just treat F1_samples as if they were inputs.

# For simplicity, pick one sample from the first layer to feed as inputs to the second.
F1_input = F1_samples.mean(axis=1)

# We must define a mean and kernel for the second layer as well.
# For demonstration, we'll reuse the same mean and kernel function, treating F1_input as "X".
# This means we think of the second layer as:
#     f2 ~ GP(mean=m(f1), cov=k(f1, f1))
# We must choose a new set of inputs for the second layer. The "inputs" now are the values from F1_input.
# Since F1_input is a sampled function (100 points), we treat these 100 points as inputs for the second GP.
X2 = F1_input

M2 = X2
KX2X2 = k(X2, X2)
KX2X2 += np.eye(len(X2)) * jitter
L2 = np.linalg.cholesky(KX2X2)
Z2 = np.random.randn(len(X2), num_samples)
F2_samples = M2[:, None] + L2 @ Z2
np.savetxt("samples_DGP.csv", F2_samples.T, delimiter=",")
# Plot second-layer GP samples
plt.figure(figsize=(8,4))
for i in range(num_samples):
    plt.plot(X, F2_samples[:, i], alpha=0.7)
plt.title("Samples from the second-layer GP (Deep GP)")
plt.xlabel("X (original space)")
plt.ylabel("f2(X) from second layer")
plt.grid(True)
plt.show()
