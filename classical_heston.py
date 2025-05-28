import numpy as np
import matplotlib.pyplot as plt


S0 = 100      # initial stock price
v0 = 0.04     # initial variance
mu = 0.05     # drift
kappa = 2.0   # rate of mean reversion
theta = 0.04  # long-term variance
xi = 0.3      # volatility of volatility
rho = -0.7    # correlation
T = 1.0       # maturity
n_steps = 250
n_paths = 10000
dt = T / n_steps

S_paths = np.zeros((n_paths, n_steps))
v_paths = np.zeros((n_paths, n_steps))
S_paths[:, 0] = S0
v_paths[:, 0] = v0

for t in range(1, n_steps):
    z1 = np.random.normal(size=n_paths)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=n_paths)
    
    v_paths[:, t] = np.maximum(
        v_paths[:, t-1] + kappa * (theta - v_paths[:, t-1]) * dt + xi * np.sqrt(v_paths[:, t-1]) * np.sqrt(dt) * z2,
        0
    )
    S_paths[:, t] = S_paths[:, t-1] * np.exp(
        (mu - 0.5 * v_paths[:, t-1]) * dt + np.sqrt(v_paths[:, t-1]) * np.sqrt(dt) * z1
    )

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(S_paths[:10].T, alpha=0.6)
plt.title("Heston Model - Stock Price Paths")
plt.xlabel("Time Step")
plt.ylabel("S(t)")

plt.subplot(1, 2, 2)
plt.plot(v_paths[:10].T, alpha=0.6)
plt.title("Heston Model - Variance Paths")
plt.xlabel("Time Step")
plt.ylabel("v(t)")
plt.tight_layout()
plt.show()

K = 100
r = 0.03

S_T = S_paths[:, -1]
v_T = v_paths[:, -1]

payoffs = np.maximum(S_T - K, 0)
discounted_payoffs = np.exp(-r * T) * payoffs

S_T_norm = (S_T - np.mean(S_T)) / np.std(S_T)
v_T_norm = (v_T - np.mean(v_T)) / np.std(v_T)

X = np.stack([S_T_norm, v_T_norm], axis=1)
y = discounted_payoffs

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
ansatz = RealAmplitudes(num_qubits=2, reps=1)

qnn_circuit = QuantumCircuit(2)
qnn_circuit.compose(feature_map, inplace=True)
qnn_circuit.compose(ansatz, inplace=True)

input_params = feature_map.parameters
weight_params = ansatz.parameters

estimator = Estimator()
qnn = EstimatorQNN(
    circuit=qnn_circuit,
    input_params=input_params,
    weight_params=weight_params,
    estimator=estimator,
)

model = TorchConnector(qnn)

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

epochs = 50
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_tensor).squeeze()
    loss = loss_func(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("QNN Training Loss Curve")
plt.grid()
plt.show()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred_qnn = model(X_test_tensor).squeeze().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_qnn, alpha=0.5)
plt.xlabel("True Discounted Payoff")
plt.ylabel("QNN Predicted Payoff")
plt.title("QNN vs True Payoff (Heston Model)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid()
plt.show()

mse = mean_squared_error(y_test, y_pred_qnn)
print("Test MSE:", mse)

from mpl_toolkits.mplot3d import Axes3D

S_range = np.linspace(min(S_T), max(S_T), 50)
v_range = np.linspace(min(v_T), max(v_T), 50)
S_mesh, v_mesh = np.meshgrid(S_range, v_range)

S_norm = (S_mesh - np.mean(S_T)) / np.std(S_T)
v_norm = (v_mesh - np.mean(v_T)) / np.std(v_T)

X_grid = np.stack([S_norm.ravel(), v_norm.ravel()], axis=1)
X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)

with torch.no_grad():
    y_grid_pred = model(X_grid_tensor).squeeze().numpy().reshape(S_mesh.shape)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_mesh, v_mesh, y_grid_pred, cmap='viridis')
ax.set_xlabel("S(T)")
ax.set_ylabel("v(T)")
ax.set_zlabel("Predicted Price")
ax.set_title("QNN Option Pricing Surface under Heston Model")
plt.tight_layout()
plt.show()