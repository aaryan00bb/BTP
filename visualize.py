import numpy as np
import matplotlib.pyplot as plt

# Parameters
for ny in [11,21,31,41,51,101]:
  nx= 400  # lattice dimensions
  nt = 5000  # number of time steps
  viscosity = 0.02  # fluid viscosity
  tau = 3 * viscosity + 0.5  # relaxation time
  omega = 1.0 / tau  # relaxation parameter
  dpdx = 0.001  # reversed pressure gradient for positive flow direction

  # Lattice weights and velocities for D2Q9 model
  w = [4/9] + [1/9]*4 + [1/36]*4  # weights
  cx = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1])
  cy = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1])

  # Initialize distribution function
  f = np.ones((ny, nx, 9)) * w

  # Main loop
  for it in range(nt):
      # Streaming step
      for i in range(9):
          f[:, :, i] = np.roll(f[:, :, i], cx[i], axis=1)
          f[:, :, i] = np.roll(f[:, :, i], cy[i], axis=0)

      # Bounce-back boundary conditions at walls
      for i in range(9):
          f[0, :, i] = f[1, :, i]  # bottom wall
          f[-1, :, i] = f[-2, :, i]  # top wall

      # Density and velocity calculations
      rho = np.sum(f, axis=2)
      ux = np.sum(f * cx, axis=2) / rho
      uy = np.sum(f * cy, axis=2) / rho

      # Apply pressure gradient
      ux[1:-1, :] += dpdx

      # Collision step
      feq = np.zeros((ny, nx, 9))
      for i in range(9):
          cu = 3 * (cx[i]*ux + cy[i]*uy)
          feq[:, :, i] = rho * w[i] * (1 + cu + 0.5 * cu*2 - 1.5 * (ux2 + uy*2))
      f += omega * (feq - f)

      

  # Final Velocity profile visualization
  velocity_profile = np.mean(ux, axis=1)
  plt.figure(figsize=(6, 4))
  plt.plot(np.linspace(0, ny, ny), velocity_profile, label='Velocity Profile')
  plt.ylabel('Velocity')
  plt.xlabel('Channel Width')
  plt.title('Final Velocity Profile in Pressure-Driven Flow')
  plt.legend()
  plt.show()
