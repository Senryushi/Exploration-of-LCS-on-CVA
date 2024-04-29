import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


### SUMMARY

# This program outputs an animation of the grid particles moving in a given
# velocity field. You can change the dimensions of the plots, you can
# choose to plot a grid of particles of any size but also choose to plot
# a uniformly distributed circle (or "blob") of particles.

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


def saddleVel(x, y, t):
    return np.array((-x, y))


def SHO(x, y, t):
    return np.array((y, -x))


def sin_saddle(x, y, t):
    return np.array((np.sin(x), -np.sin(y)))


### Integration Properties

initial_t = 0
final_T = 10
dt = 0.05
frameCnt = int(np.abs(final_T - initial_t) / dt)

frames = np.linspace(initial_t, final_T, frameCnt)


### What velocity funciton are we studying?

velfunc = gyreVel

### Plot dimensions

gridDim = [[0, 2], [0, 1]]
gridSpacing = [50, 25]

### Where do you want the blob's initial position and size to be?

centre = [0, 0]
rad = 0.3

# Number of particles in x and y dimensions respectively
Nx = gridSpacing[0]
Ny = gridSpacing[1]

# Total number of particles
num_particles = gridSpacing[0] * gridSpacing[1]

# Buffer to plot particles outside of the frame window
buffer = (gridDim[0][1] - gridDim[0][0] + gridDim[1][1] - gridDim[1][0]) * 0.5

### Type of particle plot (Grid or Blob?)

plot = "Grid"

### Initialising the particle positions in a grid


def set_grid(gridDim, gridSpacing):
    xGrid = np.linspace(gridDim[0][0], gridDim[0][1], gridSpacing[0])
    yGrid = np.linspace(gridDim[1][0], gridDim[1][1], gridSpacing[1])
    X, Y = np.meshgrid(xGrid, yGrid)
    return X, Y


### Calculate particle trajectories


def advection(x, y, frame, velfunc):
    x_dot, y_dot = velfunc(x, y, frame)
    x_new = x + dt * x_dot
    y_new = y + dt * y_dot

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


###particle generation

# Creating an even distribution of particles in the shape of a circle

phi = (1 + m.sqrt(5)) / 2  # golden ratio


def radius(k, n, b):
    if k > n - b:
        return 1.0
    else:
        return m.sqrt(k - 0.5) / m.sqrt(n - (b + 1) / 2)


def sunflower(n, centre, set_r, alpha=0, geodesic=False):
    points = []
    angle_stride = 360 * phi if geodesic else 2 * m.pi / phi**2
    b = round(alpha * m.sqrt(n))  # number of boundary points
    for k in range(1, n + 1):
        r = set_r * radius(k, n, b)
        theta = k * angle_stride
        points.append((r * m.cos(theta) + centre[0], r * m.sin(theta) + centre[1]))
    return points


# Sets the initial positions of the particles in a Grid or Blob
particle_data = []

# Initialise particles as a Grid
if plot == "Grid":

    x = np.linspace(gridDim[0][0] - buffer, gridDim[0][1] + buffer, Nx)
    y = np.linspace(gridDim[1][0] - buffer, gridDim[1][1] + buffer, Ny)
    xx, yy = np.meshgrid(x, y)
    xx = xx.T
    yy = yy.T

    particle_data = [xx, yy]

# Initialise particles as a Blob
if plot == "Blob":

    initial_x_values = np.vstack(sunflower(num_particles, centre, rad, 0)).T[0]
    initial_y_values = np.vstack(sunflower(num_particles, centre, rad, 0)).T[1]

    for i in range(num_particles):
        x0 = initial_x_values[i]  # Initial x-coordinate for particle i
        y0 = initial_y_values[i]  # Initial y-coordinate for particle i
        particle_data.append({"x": [x0], "y": [y0]})


particle_trajectories = flowLine(particle_data, velfunc, frames)


### Call for a global figure (only once)

fig, axes = plt.subplots()
axes.set(xlim=gridDim[0], ylim=gridDim[1], xlabel="X", ylabel="Y")
axes.legend
scat = axes.scatter(particle_data[0], particle_data[1])


### Animations for all particle trajectories


def update(t):
    # for each frame, update the data stored on each artist.
    data = []
    x = np.ndarray.flatten(particle_trajectories[0, :, :, t])
    y = np.ndarray.flatten(particle_trajectories[1, :, :, t])
    data = np.stack([x, y]).T

    # update the scatter plot:

    scat.set_offsets(data)

    return scat


ani = FuncAnimation(fig=fig, func=update, frames=frameCnt, interval=100)
plt.show()
