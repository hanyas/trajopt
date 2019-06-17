from casadi import *
import matplotlib.pyplot as plt


# Declare time variables
T = 10.0  # time horizon
N = 50  # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1, x2)
u = MX.sym('u')

# -----
# Model equations
# -----

# Select a model
mdl = 1

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

# Initial state
x_init = [1, 1]

# Objective
c = x1 ** 2 + x2 ** 2 + u ** 2

# -----
# Fixed step Runge-Kutta 4 integrator for discrete dynamics
# -----
M = 4  # number of steps per interval
DT = T/N/M
# Function synopsis: name sting, list of inputs, list of symbolic expressions for outputs
f = Function('f', [x, u], [xdot, c])
# Declare variables for integration
X0 = MX.sym('X0', 2)
U = MX.sym('U')
X = X0
C = 0

for j in range(M):
    k1_x, k1_q = f(X, U)
    k2_x, k2_q = f(X + DT/2. * k1_x, U)
    k3_x, k3_q = f(X + DT/2. * k2_x, U)
    k4_x, k4_q = f(X + DT    * k3_x, U)
    X += DT/6. * (k1_x + 2.*k2_x + 2.*k3_x + k4_x)
    C += DT / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)
# Function synopsis: see above + list of labels inputs, list of labels output
F = Function('F', [X0, U], [X, C], ['x0', 'u'], ['xf', 'qf'])

# -----
# Initialize empty NLP
# -----
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += x_init
ubw += x_init
w0 += [0, 0]

# -----
# Formulate the NLP
# -----
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    # Add the optimization variables and boundaries in form of a list
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]

    # Integrate one step
    Fk = F(x0=Xk, u=Uk)
    Xk_end = Fk['xf']
    J += Fk['qf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), 2)
    w += [Xk]
    lbw += [-0.25, -inf]
    ubw += [inf, inf]
    w0 += [0, 0]

    # Add equality constraint
    g += [Xk_end - Xk]
    lbg += [0, 0]
    ubg += [0, 0]

# -----
# Time for CasADi magic
# -----
# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)  # w0 = initial guess
w_opt = sol['x'].full().flatten()

# -----
# Plot the solution
# -----
# w_opt is an array of [x1_opt, x2_opt, u_opt] per time step, so take every 3rd element
x1_opt = w_opt[0::3]
x2_opt = w_opt[1::3]
u_opt = w_opt[2::3]

t_grid = [T/N*k for k in range(N+1)]
plt.figure(1)
plt.plot(t_grid, x1_opt)
plt.plot(t_grid, x2_opt)
plt.step(t_grid, vertcat(DM.nan(1), u_opt), c='r')
plt.title('Action trajectory')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()
