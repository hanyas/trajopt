from casadi import *
import matplotlib.pyplot as plt


# -----
# Continuous-time dynamics and cost
# -----

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')  # TODO: Make x2 an MX symbolic variable with label 'x2'
x = vertcat(x1, x2)
u = MX.sym('u')

# Select a model
mdl = 1  # TODO: change me, if you want to

# Mass spring damper system
if mdl == 1:
    mass = 1.0
    zeta = 0.3
    omega_n = 2
    xdot = vertcat(x2, 1./mass * u - omega_n**2 * x1 - 2.*zeta*omega_n*x2)
# Van der Pol oscillator
elif mdl == 2:
    mu = 1.0
    xdot = vertcat(x2, mu * (1 - x1**2)*x2 - x1 + u)
else:
    raise ValueError

# Running cost
c = x1 ** 2 + x2 ** 2 + u ** 2  # TODO: Implement sum-of-squares running cost

# Function synopsis: name sting, list of inputs, list of symbolic outputs
f = Function('f', [x, u], [xdot, c])

# -----
# Discrete-time dynamics and cost
# -----

# Time variables
T = 10.0  # time horizon
N = 50  # number of control intervals
M = 4  # number of integration steps per interval

# Symbolic variables for integration
X0 = MX.sym('X0', 2)
U = MX.sym('U')
X = X0
C = 0

# Fixed-step Runge-Kutta 4 integrator for discrete dynamics
DT = T/N/M
for j in range(M):
    k1_x, k1_q = f(X, U)
    k2_x, k2_q = f(X + DT/2. * k1_x, U)
    k3_x, k3_q = f(X + DT/2. * k2_x, U)
    k4_x, k4_q = f(X + DT    * k3_x, U)
    X += DT/6. * (k1_x + 2.*k2_x + 2.*k3_x + k4_x)
    C += DT / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)
# Function synopsis: see above + list of labels inputs, list of labels output
F = Function('F', [X0, U], [X, C], ['x0', 'u'], ['xf', 'qf'])  # TODO: Create an integrator function with inputs [X0, U]
          # TODO: and outputs [X, C]. Label inputs and outputs
          # TODO: by ['x0', 'u'] and ['xf', 'qf'] respectively

# -----
# Formulate optimization problem
# -----
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# Set the initial state
x_init = [1, 1]  # TODO: Play with the initial condition
Xk = MX(x_init)
for k in range(N):
    # Create a new variable for the control
    Uk = MX.sym('U_' + str(k))
    # Add the optimization variables and boundaries in form of a list
    w += [Uk]    # TODO: Add action Uk to the list of optimization variables w
    lbw += [-1]  # TODO: Add the lower bound of -1 corresponding to Uk
    ubw += [1]
    w0 += [0]

    # Integrate one step
    Fk = F(x0=Xk, u=Uk)  # TODO: Perform one step of discrete dynamics
    Xk = Fk['xf']
    J += Fk['qf']  # TODO: Extract the running cost from Fk

    # Add inequality constraints
    g += [Xk[0]]
    lbg += [-0.25]  # TODO: Add lower bound of -0.25 on Xk[0]
    ubg += [inf]

# -----
# Call the optimizer
# -----
# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)  # TODO: Call solver with x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg
w_opt = sol['x']

# -----
# Simulate the system with optimal controls
# -----
u_opt = w_opt
x_opt = [x_init]
for k in range(N):
    Fk = F(x0=x_opt[-1], u=u_opt[k])  # TODO: Simulate one step with state x_opt[-1] and control u_opt[k]
    x_opt += [Fk['xf'].full()]
x1_opt = [r[0] for r in x_opt]
x2_opt = [r[1] for r in x_opt]

# Plot
t_grid = [T/N*k for k in range(N + 1)]
plt.figure(1)
plt.plot(t_grid, x1_opt)
plt.plot(t_grid, x2_opt)
plt.step(t_grid, vertcat(DM.nan(1), u_opt), c='r')
plt.xlabel('time')
plt.legend(['x1', 'x2', 'u'])
plt.grid()
plt.show()
