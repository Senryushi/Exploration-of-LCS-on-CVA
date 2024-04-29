import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

### WARNING:
# This version of the code is wrong as we don't account for all of the
# boundary conditions.


### SUMMARY

# Select a velocity field below and this program will give the animated FTLE plot
# for the integration times that you give. You can change the number of particles,
# the boundary dimensions and the length of time added between each calculation of
# of the FTLE.


### Velocity Field Definitions

eps = 0.25
A = 0.1
w = 2 * m.pi / 10


def b(t, eps=eps, w=w):
    return 1 - 2 * eps * np.sin(w * t)


def a(t, eps=eps, w=w):
    return eps * np.sin(w * t)


def f(x, t):
    return a(t) * x**2 + b(t) * x


def gyreVel(x, y, t, A=A):
    return np.array(
        (
            -m.pi * A * np.sin(m.pi * f(x, t)) * np.cos(m.pi * y),
            m.pi
            * A
            * np.cos(m.pi * f(x, t))
            * np.sin(y * m.pi)
            * (2 * a(t) * x + b(t)),
        )
    )


def randomthing(x, y, t):
    return np.array((1 + (np.tanh(x)) ** 2, -(2 * np.tanh(x)) * y / (np.cosh(x) ** 2)))


def saddleflow(x, y, t):
    return np.array((-x, y))

def sin_saddle(x, y, t):
    return np.array((np.sin(x), -np.sin(y)))

def sink(x, y, t):
    return np.array((-x, -y))

def oppositeFlow(x, y, t):
    xv = np.where(y > (gridDim[1][1] - gridDim[1][0]) / 2, -0.1, 0.1)
    return np.array((xv, np.zeros_like(xv)))

def TGVortex(x, y, t):
    return np.array((np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y)))


### Integration Properties

initial_t = 0
final_T = 10
dt = 0.05
frameCnt = int(np.abs(final_T - initial_t) / dt)

totalTime = abs(final_T - initial_t)

frames = np.linspace(initial_t, final_T, frameCnt)


### What velocity funciton are we studying?

velfunc = gyreVel

### Plot dimensions

gridDim = [[0, 2], [0, 1]]      # Dimensions of the plot
gridSpacing = [200, 100]        # Number of grid particles in each dimension

# Number of particles in x and y dimensions respectively
Nx = gridSpacing[0]             
Ny = gridSpacing[1]

# Total number of particles
num_particles = gridSpacing[0] * gridSpacing[1]


### Setting up the grid of particles

def set_grid(gridDim, gridSpacing):
    xGrid = np.linspace(gridDim[0][0], gridDim[0][1], gridSpacing[0])
    yGrid = np.linspace(gridDim[1][0], gridDim[1][1], gridSpacing[1])
    X, Y = np.meshgrid(xGrid, yGrid)
    return X.T, Y.T


### Calculate particle trajectories

def advection(x, y, frame, velfunc):
    x_dot, y_dot = velfunc(x, y, frame)
    x_new = x - dt * x_dot
    y_new = y - dt * y_dot

    return x_new, y_new


def flowLine(particle_data, velfunc, frames):
    xx, yy = particle_data

    fin_particle_data_x = np.zeros((Nx, Ny, len(frames)))
    fin_particle_data_y = np.zeros((Nx, Ny, len(frames)))

    for k in range(len(frames)):
        x_new, y_new = advection(xx, yy, frames[k], velfunc)
        xx = x_new
        yy = y_new
        fin_particle_data_x[:, :, k] = x_new
        fin_particle_data_y[:, :, k] = y_new

    fin_particle_data = np.array((fin_particle_data_x, fin_particle_data_y))

    return fin_particle_data


### Set the positions of the particles in the grid

particle_pp = set_grid(gridDim, gridSpacing)


### Calculate the FlowMap (Total trajectory of each grid particle)

particle_trajectory = flowLine(particle_pp, velfunc, frames)


### Calculate FTLE field

def calculate_FTLE(I, F):
    ### Calculating the gradient of the flowmap

    """
    Gradient matrix looks like this:
    [[aa bb],
    [cc dd]]


    """

    I_x = I[0]
    I_y = I[1]
    F_x = F[0]
    F_y = F[1]

    FTLE_field = np.zeros((gridSpacing[0], gridSpacing[1]))

    for i in range(gridSpacing[0]):
        for j in range(gridSpacing[1]):
            # Boundary Conditions
            if j == gridSpacing[1] - 1 and i == gridSpacing[0] - 1:
                aa = (F_x[i, j] - F_x[i - 1, j]) / (I_x[i, j] - I_x[i - 1, j])
                bb = (F_x[i, j] - F_x[i, j - 1]) / (I_y[i, j] - I_y[i, j - 1])
                cc = (F_y[i, j] - F_y[i - 1, j]) / (I_x[i, j] - I_x[i - 1, j])
                dd = (F_y[i, j] - F_y[i, j - 1]) / (I_y[i, j] - I_y[i, j - 1])

            elif j == gridSpacing[1] - 1:
                aa = (F_x[i + 1, j] - F_x[i - 1, j]) / (I_x[i + 1, j] - I_x[i - 1, j])
                bb = (F_x[i, j] - F_x[i, j - 1]) / (I_y[i, j] - I_y[i, j - 1])
                cc = (F_y[i + 1, j] - F_y[i - 1, j]) / (I_x[i + 1, j] - I_x[i - 1, j])
                dd = (F_y[i, j] - F_y[i, j - 1]) / (I_y[i, j] - I_y[i, j - 1])

            elif i == gridSpacing[0] - 1:
                aa = (F_x[i, j] - F_x[i - 1, j]) / (I_x[i, j] - I_x[i - 1, j])
                bb = (F_x[i, j + 1] - F_x[i, j - 1]) / (I_y[i, j + 1] - I_y[i, j - 1])
                cc = (F_y[i, j] - F_y[i - 1, j]) / (I_x[i, j] - I_x[i - 1, j])
                dd = (F_y[i, j + 1] - F_y[i, j - 1]) / (I_y[i, j + 1] - I_y[i, j - 1])

            # Central Differences
            else:
                aa = (F_x[i + 1, j] - F_x[i - 1, j]) / (I_x[i + 1, j] - I_x[i - 1, j])
                bb = (F_x[i, j + 1] - F_x[i, j - 1]) / (I_y[i, j + 1] - I_y[i, j - 1])
                cc = (F_y[i + 1, j] - F_y[i - 1, j]) / (I_x[i + 1, j] - I_x[i - 1, j])
                dd = (F_y[i, j + 1] - F_y[i, j - 1]) / (I_y[i, j + 1] - I_y[i, j - 1])

            grad = np.matrix([[aa, bb], [cc, dd]])
            C = np.matmul(grad.T, grad)
            Lamda = np.linalg.eigvals(C)
            Lamda = np.max(Lamda)
            FTLE = (1 / totalTime) * m.log(Lamda**0.5)
            FTLE_field[i, j] = FTLE

    return FTLE_field.T

### Plotting the results as an animation

# FTLE time step
step = 20

n_steps = np.arange(0, frameCnt, step)

fig, ax = plt.subplots()
ax.set(xlabel="X", ylabel="Y")

ims = []
for i in n_steps:
    im = ax.imshow(
        calculate_FTLE(
            particle_trajectory[:, :, :, 0], particle_trajectory[:, :, :, i]
        ),
        animated=True,
        origin="lower",
    )
    if i == 0:
        ax.imshow(
            calculate_FTLE(
                particle_trajectory[:, :, :, 0], particle_trajectory[:, :, :, i]
            ),
            origin="lower",
        )
        fig.colorbar(im, shrink = 0.5)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=200)
plt.show()

