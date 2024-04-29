import numpy as np
import cv2 as cv
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


### SUMMARY

# Given a video, this program will output the Forwards or Backwards FTLE field for
# the whole video or for blocks in regular intervals. You can change the threshold
# values, the integration times and the number of frames to average over in each
# mean frame.

# Find video path and name
cv.samples.addSamplesDataSearchPath("Folder_path")
cap = cv.VideoCapture(cv.samples.findFile("File_path"))
# Retrieve the first frame of the video
ret, frame1 = cap.read()

# If the video frame is too big, scale the frame size down

scale = 0.5
tooBig = False
if frame1.shape[0] > 1000:
    frame1 = cv.resize(frame1, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    tooBig = True

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

### Video Properties

length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
frame_size = np.delete(frame1.shape, 2, 0)


### Final Output Properties

# FTLE_mode = ("[1]", "[2]")

# [1]:
# "Full" -  Calculate FTLE field for whole video
# "Blocks" - Blocks of equal time intervals

# [2]:
# "Forward" - Run Foward time FTLE
# "Backward" - Run Backward time FTLE

FTLE_mode = ("Blocks", "Forward")

# Set n_V frames to average over and total number of mean frames M
n_Frame = 10
num_mean_frames = int(np.ceil(length / n_Frame))

# Set number of mean frames in a block
block_length = 10
Tend = block_length

# Set parameter values for threshold
alpha = 0
beta = 1

### Particle Properties

grid_spacing = 3  # every p pixels there's a particle placed

Nx = int(np.floor(frame_size[1] / grid_spacing))  # Number of particles in x direction
Ny = int(np.floor(frame_size[0] / grid_spacing))  # Number of particles in y direction
num_particles = Nx * Ny
dt = 1  # integration time step (in frames)

gridNum = [Nx, Ny]


### Initialising the grid of Particles

x = np.linspace(0, frame_size[1] - 1, Nx)
y = np.linspace(0, frame_size[0] - 1, Ny)
xx, yy = np.meshgrid(x, y)


### Particle Trajectory Calculation


# Advect Particles
def advection(x, y, frame, velfunc):

    # Calculate new particle position with Foward Euler method:
    x_dot, y_dot = velfunc(x, y, frame)
    x_new = x + dt * x_dot
    y_new = y + dt * y_dot

    # Boundary Conditions
    x_new[x_new < 0] = 0
    x_new[x_new > frame_size[1] - 1] = frame_size[1] - 1
    y_new[y_new < 0] = 0
    y_new[y_new > frame_size[0] - 1] = frame_size[0] - 1

    return x_new, y_new


# Calculate particle trajectories
def flowLine(particle_data, velfunc, frames):

    # Initial Particle Positions
    xx, yy = particle_data

    fin_particle_data_x = np.zeros((gridNum[1], gridNum[0], len(frames)))
    fin_particle_data_y = np.zeros((gridNum[1], gridNum[0], len(frames)))

    # Advect Particles
    for k in range(len(frames)):
        x_new, y_new = advection(xx, yy, frames[k], velfunc)
        xx = x_new
        yy = y_new
        fin_particle_data_x[:, :, k] = x_new
        fin_particle_data_y[:, :, k] = y_new

    fin_particle_data = [fin_particle_data_x, fin_particle_data_y]

    return fin_particle_data


### Velocity field Calculation for whole video

u_mean_data = np.zeros((frame_size[1], frame_size[0], num_mean_frames))
v_mean_data = np.zeros((frame_size[1], frame_size[0], num_mean_frames))
k = 0

### Reading the video frames

vid_frames = np.zeros((frame_size[0], frame_size[1], length))
vid_frames[:, :, 0] = prvs

# Retrieve all video frames

for i in range(length - 1):
    ret, frame2 = cap.read()
    if tooBig == True:
        frame2 = cv.resize(
            frame2, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA
        )
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    vid_frames[:, :, i + 1] = next

# If Backwards, reverse the video frames

opFlowFrames = np.zeros_like(vid_frames)
if FTLE_mode[1] == "Forward":
    opFlowFrames = vid_frames
elif FTLE_mode[1] == "Backward":
    for i in range(length):
        opFlowFrames[:, :, i] = vid_frames[:, :, length - (i + 1)]

# Starting velocity field calculation

prvs = opFlowFrames[:, :, 0]

# calculate mean frames

while k < num_mean_frames:
    u_data = np.zeros((frame_size[1], frame_size[0], n_Frame))
    v_data = np.zeros((frame_size[1], frame_size[0], n_Frame))
    vid_end = False
    cnt_n_frame = 0

    # Get the frames for the mean optical flow field

    for i in range(n_Frame):

        if k * n_Frame + i == np.shape(opFlowFrames)[2] - 1:
            vid_end = True
            print("no more frames!")
            break

        next = opFlowFrames[:, :, k * n_Frame + i + 1]
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 5, 10, 3, 5, 1, 0)

        u_data[:, :, i] = flow[:, :, 0].T
        v_data[:, :, i] = flow[:, :, 1].T

        prvs = next
        cnt_n_frame = i + 1

    # Calculate mean field and add it to the array of velocity field data

    if not vid_end:
        u_mean_data[:, :, k] = np.mean(u_data, axis=2)
        v_mean_data[:, :, k] = np.mean(v_data, axis=2)
    else:
        if cnt_n_frame == 0:
            break
        else:
            u_mean_data[:, :, k] = np.mean(
                np.delete(u_data, np.s_[cnt_n_frame:n_Frame], 2), axis=2
            )
            v_mean_data[:, :, k] = np.mean(
                np.delete(v_data, np.s_[cnt_n_frame:n_Frame], 2), axis=2
            )

    k += 1

### APPLY A GAUSSIAN FILTER

for i in range(num_mean_frames):
    u_mean_data[:, :, i] = gaussian_filter(u_mean_data[:, :, i], 5)
    v_mean_data[:, :, i] = gaussian_filter(v_mean_data[:, :, i], 5)

# Interpolation scheme for the velocity field

grid_x = np.arange(0, frame_size[1])
grid_y = np.arange(0, frame_size[0])
grid_t = np.arange(0, num_mean_frames)
u_interp = RegularGridInterpolator((grid_x, grid_y, grid_t), u_mean_data)
v_interp = RegularGridInterpolator((grid_x, grid_y, grid_t), v_mean_data)

# Define the velocity field using interpolation


def flowfield(x, y, t):
    return np.array((u_interp((x, y, t)), v_interp((x, y, t))))


### Calculate Particle Trajectories

# Set the initial positions of the particles

particle_pp = [xx, yy]

# Number of frames for which FTLE is calculated
n_Steps = int(np.floor(num_mean_frames / block_length))

frames = np.arange(0, num_mean_frames)

# Prepare particle trajectory by initialising grid of particles
# every block_length mean frames
if FTLE_mode[0] == "Blocks":
    full_particle_trajectory = np.zeros(
        (2, gridNum[1], gridNum[0], block_length * n_Steps)
    )
    for i in range(n_Steps):
        full_particle_trajectory[:, :, :, i * block_length : (i + 1) * block_length] = (
            flowLine(
                particle_pp,
                flowfield,
                frames[i * block_length : (i + 1) * block_length],
            )
        )

# Calculate trajectory for full video without re-initialising grid

elif FTLE_mode[0] == "Full":
    full_particle_trajectory = np.zeros((2, gridNum[1], gridNum[0], len(frames)))
    full_particle_trajectory[:, :, :, :] = flowLine(
        particle_pp,
        flowfield,
        frames,
    )


def calculate_FTLE(F):
    ### Calculating the gradient of the flowmap

    """
    Gradient matrix looks like this:
    [[aa bb],
    [cc dd]]


    """
    c = 1  # number of pixels wide to do central differences

    F_x = np.matrix(F[0])
    F_y = np.matrix(F[1])

    dx = c * grid_spacing
    dy = c * grid_spacing

    FTLE_field = np.zeros((Ny, Nx))
    aa = np.zeros_like(F_x)
    bb = np.zeros_like(F_x)
    cc = np.zeros_like(F_x)
    dd = np.zeros_like(F_x)

    # Central Differences

    aa[c:-c, c:-c] = (F_x[2 * c :, c:-c] - F_x[: -2 * c, c:-c]) / (2 * dx)
    bb[c:-c, c:-c] = (F_x[c:-c, 2 * c :] - F_x[c:-c, : -2 * c]) / (2 * dy)
    cc[c:-c, c:-c] = (F_y[2 * c :, c:-c] - F_y[: -2 * c, c:-c]) / (2 * dx)
    dd[c:-c, c:-c] = (F_y[c:-c, 2 * c :] - F_y[c:-c, : -2 * c]) / (2 * dy)

    # Side Boundary Conditions

    aa[:c, c:-c] = (F_x[c : 2 * c, c:-c] - F_x[:c, c:-c]) / (dx)
    bb[:c, c:-c] = (F_x[:c, 2 * c :] - F_x[:c, : -2 * c]) / (2 * dy)
    cc[:c, c:-c] = (F_y[c : 2 * c, c:-c] - F_y[:c, c:-c]) / (dx)
    dd[:c, c:-c] = (F_y[:c, 2 * c :] - F_y[:c, : -2 * c]) / (2 * dy)

    aa[-c:, c:-c] = (F_x[-c:, c:-c] - F_x[-2 * c : -c, c:-c]) / (dx)
    bb[-c:, c:-c] = (F_x[-c:, 2 * c :] - F_x[-c:, : -2 * c]) / (2 * dy)
    cc[-c:, c:-c] = (F_y[-c:, c:-c] - F_y[-2 * c : -c, c:-c]) / (dx)
    dd[-c:, c:-c] = (F_y[-c:, 2 * c :] - F_y[-c:, : -2 * c]) / (2 * dy)

    aa[c:-c, :c] = (F_x[2 * c :, :c] - F_x[: -2 * c, :c]) / (2 * dx)
    bb[c:-c, :c] = (F_x[c:-c, c : 2 * c] - F_x[c:-c, :c]) / (dy)
    cc[c:-c, :c] = (F_y[2 * c :, :c] - F_y[: -2 * c, :c]) / (2 * dx)
    dd[c:-c, :c] = (F_y[c:-c, c : 2 * c] - F_y[c:-c, :c]) / (dy)

    aa[c:-c, -c:] = (F_x[2 * c :, -c:] - F_x[: -2 * c, -c:]) / (2 * dx)
    bb[c:-c, -c:] = (F_x[c:-c, -c:] - F_x[c:-c, -2 * c : -c]) / (dy)
    cc[c:-c, -c:] = (F_y[2 * c :, -c:] - F_y[: -2 * c, -c:]) / (2 * dx)
    dd[c:-c, -c:] = (F_y[c:-c, -c:] - F_y[c:-c, -2 * c : -c]) / (dy)

    # Corner Boundaries

    aa[:c, :c] = (F_x[c : 2 * c, :c] - F_x[:c, :c]) / (dx)
    bb[:c, :c] = (F_x[:c, c : 2 * c] - F_x[:c, :c]) / (dy)
    cc[:c, :c] = (F_y[c : 2 * c, :c] - F_y[:c, :c]) / (dx)
    dd[:c, :c] = (F_y[:c, c : 2 * c] - F_y[:c, :c]) / (dy)

    aa[-c:, :c] = (F_x[-c:, :c] - F_x[-2 * c : -c, :c]) / (dx)
    bb[-c:, :c] = (F_x[-c:, c : 2 * c] - F_x[-c:, :c]) / (dy)
    cc[-c:, :c] = (F_y[-c:, :c] - F_y[-2 * c : -c, :c]) / (dx)
    dd[-c:, :c] = (F_y[-c:, c : 2 * c] - F_y[-c:, :c]) / (dy)

    aa[:c, -c:] = (F_x[c : 2 * c, -c:] - F_x[:c, -c:]) / (dx)
    bb[:c, -c:] = (F_x[:c, -c:] - F_x[:c, -2 * c : -c]) / (dy)
    cc[:c, -c:] = (F_y[c : 2 * c, -c:] - F_y[:c, -c:]) / (dx)
    dd[:c, -c:] = (F_y[:c, -c:] - F_y[:c, -2 * c : -c]) / (dy)

    aa[-c:, -c:] = (F_x[-c:, -c:] - F_x[-2 * c : -c, -c:]) / (dx)
    bb[-c:, -c:] = (F_x[-c:, -c:] - F_x[-c:, -2 * c : -c]) / (dy)
    cc[-c:, -c:] = (F_y[-c:, -c:] - F_y[-2 * c : -c, -c:]) / (dx)
    dd[-c:, -c:] = (F_y[-c:, -c:] - F_y[-c:, -2 * c : -c]) / (dy)

    # Calculate FTLE field

    for i in range(Ny):
        for j in range(Nx):
            grad = np.matrix([[aa[i, j], bb[i, j]], [cc[i, j], dd[i, j]]])
            C = np.matmul(grad.T, grad)
            Lamda = np.linalg.eigvals(C)
            Lamda = np.max(Lamda)
            FTLE = (1 / Tend) * m.log(Lamda**0.5)

            # Threshold and Magnification Scheme

            if FTLE >= alpha:
                FTLE = beta * FTLE
            else:
                FTLE = (1 - beta) * FTLE
            FTLE_field[i, j] = FTLE

    return FTLE_field


### Plotting the Results as an animation

fig, ax = plt.subplots()
ax.set(xlabel="X", ylabel="Y")

ims = []
if FTLE_mode[0] == "Blocks":
    for i in range(n_Steps):
        im = ax.imshow(
            calculate_FTLE(
                full_particle_trajectory[:, :, :, (i + 1) * block_length - 1]
            ),
            animated=True,
        )

        if i == 0:
            ax.imshow(
                calculate_FTLE(
                    full_particle_trajectory[:, :, :, (i + 1) * block_length - 1]
                )
            )
            fig.colorbar(im, shrink=0.8)

        text = fig.text(0.5, 0.8, "t = " + str(i), animated=True)
        ims.append([im, text])


elif FTLE_mode[0] == "Full":
    for i in range(n_Steps):
        im = ax.imshow(
            calculate_FTLE(
                full_particle_trajectory[:, :, :, (i + 1) * block_length - 1]
            ),
            animated=True,
        )

        if i == 0:
            ax.imshow(
                calculate_FTLE(
                    full_particle_trajectory[:, :, :, (i + 1) * block_length - 1]
                )
            )
        if i == n_Steps - 1:
            fig.colorbar(im, shrink=0.8)

        text = fig.text(0.5, 0.8, "t = " + str(i), animated=True)
        ims.append([im, text])


ani = ArtistAnimation(fig, ims, interval=400)
plt.show()
