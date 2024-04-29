import numpy as np
import cv2 as cv
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter


### SUMMARY

# This program calculates the normal repulsion rate, tangent repulsion rate and
# repulaion ratio for a given video. With a bit of tweaking it is also possible
# to extract the prominent ridge lines for a given video.


# Get the path to the folder and path to the video from the given folder
cv.samples.addSamplesDataSearchPath("Folder_Path")
cap = cv.VideoCapture(cv.samples.findFile("File_Path"))
ret, frame1 = cap.read()

# If frame window is too big, scale the window size down by half
scale = 0.5
tooBig = False
if frame1.shape[0] > 1000:
    frame1 = cv.resize(frame1, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    tooBig = True
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)


### Video Properties

length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Total frame count of video
frame_size = np.delete(frame1.shape, 2, 0)  # Size of frame window


### Final Output Properties

# FTLE_mode = ("[1]", "[2]")

# [1]:
# "Full" -  Calculate FTLE field for whole video
# "Blocks" - Blocks of equal time intervals

# [2]:
# "Forward" - Run Foward time FTLE
# "Backward" - Run Backward time FTLE

FTLE_mode = ("Full", "Forward")

# Set n_V frames to average over and total number of mean frames M
n_Frame = 5
num_mean_frames = int(np.ceil(length / n_Frame))

# Set number of mean frames in a block
block_length = 5
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
u_interp = interpolate.RegularGridInterpolator((grid_x, grid_y, grid_t), u_mean_data)
v_interp = interpolate.RegularGridInterpolator((grid_x, grid_y, grid_t), v_mean_data)

# Define the velocity field using interpolation


def flowfield(x, y, t):
    return np.array((u_interp((x, y, t)), v_interp((x, y, t))))


### Initialise initial particle positions

particle_pp = [xx, yy]

## Calculate Particle Trajectories

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


# FTLE field only with strong ridges
strongFTLE = calculate_FTLE(full_particle_trajectory[:, :, :, frames[-1]])


# Finding the locally maximal points of the FTLE ridges
def local_max(FTLE, L, thresh):

    ridge_points = np.where(FTLE > thresh)
    ridge_points = np.stack(ridge_points, axis=-1)
    new_ridge_points = ridge_points
    for x in ridge_points:
        if x[0] < L or x[0] > Ny - L or x[1] < L or x[1] > Nx - L:
            ind_row = np.where(new_ridge_points[:, 0] == x[0])
            ind_col = np.where(new_ridge_points[:, 1] == x[1])
            new_ridge_points = np.delete(
                new_ridge_points, np.intersect1d(ind_row, ind_col), axis=0
            )

    max_ridge_points = [[0, 0]]
    for i in range(len(new_ridge_points)):
        C_row = new_ridge_points[i, 0]
        C_col = new_ridge_points[i, 1]
        max_ridge_points = np.vstack(
            (
                max_ridge_points,
                np.concatenate(
                    np.where(
                        FTLE
                        == np.max(FTLE[C_row - L : C_row + L, C_col - L : C_col + L])
                    )
                ),
            )
        )

    max_ridge_points = np.delete(max_ridge_points, 0, axis=0)
    FTLE_ridge = np.zeros_like(FTLE)

    for i in range(len(max_ridge_points)):
        n_row = max_ridge_points[i, 0]
        n_col = max_ridge_points[i, 1]
        FTLE_ridge[n_row, n_col] = FTLE[n_row, n_col]

    max_ridge_points = np.unique(max_ridge_points, axis=0)

    return FTLE_ridge, max_ridge_points


line_dict = {}
edge_list = []


def find_line(edge_point):
    pointKey = 0
    for key in line_dict:
        for i in range(len(line_dict[key])):

            if np.all(np.equal(line_dict[key][i, :], edge_point[:])):
                pointKey = key
                break

    return pointKey


def update_edges():
    num_lines = len(line_dict)
    edge_list = np.zeros((num_lines, 2, 2))
    for i in range(num_lines):
        if len(line_dict[i]) == 1:
            edge_list[i, :] = [line_dict[i][0], [10000, 10000]]
        else:
            edge_list[i, :] = [line_dict[i][0], line_dict[i][-1]]

    return edge_list


def add_pnt_to_line(key, p, edge_p):
    ind_p = 0

    if np.all(line_dict[key][0] == edge_p):
        ind_p = 0
    else:
        ind_p = -1

    if ind_p == 0:
        line_dict[key] = np.vstack((p, line_dict[key]))
    else:
        line_dict[key] = np.vstack((line_dict[key], p))
    return


tan_len = 1
avg_len = 1

### Create a dictionary of ridge lines


def create_lines(ridge_points, radius):
    cnt = 1

    search_list = ridge_points

    line_dict[0] = np.array([search_list[0]])
    search_list = np.delete(search_list, 0, axis=0)
    edge_list = update_edges()

    for p in search_list:

        if np.all(p == [1, 153]):
            print(edge_list)

        line_edge_vec = edge_list - p
        line_edge_dist = (
            line_edge_vec[:, :, 0] ** 2 + line_edge_vec[:, :, 1] ** 2
        ) ** 0.5

        if np.min(line_edge_dist) < radius:

            edge_point = edge_list[line_edge_dist == np.min(line_edge_dist)]

            if len(edge_point) > 1:
                edge_point = edge_point[0]
            line_key = find_line(edge_point)
            add_pnt_to_line(line_key, p, edge_point)

        else:
            line_dict[cnt] = np.array([p])
            cnt += 1

        edge_list = update_edges()

    for i in range(len(line_dict)):
        if len(line_dict[i]) <= 3:
            del line_dict[i]

    smooth_line = {}
    big_u = np.linspace(0, 1, 50)
    for key in line_dict:
        tck, u = interpolate.splprep([line_dict[key][:, 0], line_dict[key][:, 1]])
        smooth_line[key] = np.array(interpolate.splev(big_u, tck)).T

    return smooth_line


### Calculate the moving frame for each point along the ridge line


def create_frame(ridge_lines):
    ridge_tangent = {}
    ridge_normal = {}
    dxdt = {}
    dydt = {}
    for key in ridge_lines:

        x = ridge_lines[key][:, 0]
        y = ridge_lines[key][:, 1]

        x = frame_size[1] / Nx * x
        y = frame_size[0] / Ny * y

        tck, u = interpolate.splprep([x, y])

        dydt[key], dxdt[key] = interpolate.splev(u, tck, der=1)

        tangent = np.zeros((len(ridge_lines[key]), 2))

        tangent[:, 0] = dxdt[key]
        tangent[:, 1] = -dydt[key]
        ridge_tangent[key] = tangent

        ridge_normal[key] = np.roll(ridge_tangent[key], 1, axis=1)
        ridge_normal[key][:, 0] = -ridge_normal[key][:, 0]

        tan_norm = (
            ridge_tangent[key][:, 0] ** 2 + ridge_tangent[key][:, 1] ** 2
        ) ** 0.5
        tan_norm = np.stack((tan_norm, tan_norm), axis=-1)
        nor_norm = (ridge_normal[key][:, 0] ** 2 + ridge_normal[key][:, 1] ** 2) ** 0.5
        nor_norm = np.stack((nor_norm, nor_norm), axis=-1)
        ridge_tangent[key] = np.divide(ridge_tangent[key][:], tan_norm)
        ridge_normal[key] = np.divide(ridge_normal[key][:], nor_norm)

    return ridge_tangent, ridge_normal


### Initialise a grid of points along the ridge lines


def initialise_points(ridge_lines, frame):
    tangent_frame, normal_frame = frame
    ridge_particles = {}

    for key in ridge_lines:
        T = tangent_frame[key]
        N = normal_frame[key]
        C = ridge_lines[key]

        particle_grid = np.zeros((3, 3, len(T), 2))
        particle_grid[0, 0, :, :] = C - 3 * T + 3 * N
        particle_grid[0, 1, :, :] = C + 3 * N
        particle_grid[0, 2, :, :] = C + 3 * T + 3 * N
        particle_grid[1, 0, :, :] = C - 3 * T
        particle_grid[1, 1, :, :] = C
        particle_grid[1, 2, :, :] = C + 3 * T
        particle_grid[2, 0, :, :] = C - 3 * T - 3 * N
        particle_grid[2, 1, :, :] = C - 3 * N
        particle_grid[2, 2, :, :] = C + 3 * T - 3 * N
        ridge_particles[key] = particle_grid

    return ridge_particles


# Calculate the trajectories of the initialies grid particles


def ridge_flowline(ridge_grid, velfunc, frames):
    final_part_dict = {}

    for key in ridge_grid:
        y_data, x_data = ridge_grid[key][:, :, :, 0], ridge_grid[key][:, :, :, 1]

        Z = len(x_data[0, 0])

        x_data = frame_size[1] / Nx * x_data
        y_data = frame_size[0] / Ny * y_data

        x_data[x_data < 0] = 0
        x_data[x_data > frame_size[1] - 1] = frame_size[1] - 1
        y_data[y_data < 0] = 0
        y_data[y_data > frame_size[0] - 1] = frame_size[0] - 1

        fin_particle_data = np.zeros((len(frames), 3, 3, Z, 2))

        for k in range(len(frames)):
            x_dot, y_dot = velfunc(x_data, y_data, frames[k])
            x_new = x_data + dt * x_dot
            y_new = y_data + dt * y_dot

            x_new[x_new < 0] = 0
            x_new[x_new > frame_size[1] - 1] = frame_size[1] - 1
            y_new[y_new < 0] = 0
            y_new[y_new > frame_size[0] - 1] = frame_size[0] - 1

            x_data = x_new
            y_data = y_new

            fin_particle_data[k, :, :, :, 0] = x_new
            fin_particle_data[k, :, :, :, 1] = y_new

        final_part_dict[key] = fin_particle_data

    return final_part_dict


# Calculate the tangent repulsion rates, normal repulsion rates and
# repulsion ratio.


def perturbation_field(final_traj, initial_frame):

    tang_rep_rate = {}
    rep_ratio = {}
    norm_rep_rate = {}

    for key in final_traj:
        line_parts_traj = final_traj[key]
        ini_tang_frame, ini_norm_frame = initial_frame[0][key], initial_frame[1][key]

        Z = len(ini_tang_frame)

        initial_part_pos = line_parts_traj[0, :, :, :, :]
        final_part_pos = line_parts_traj[-1, :, :, :, :]

        # Calculate grid particles relative positions with respect to central particle

        dx_ = np.zeros((Z))
        dy_ = np.zeros((Z))
        for i in range(Z):
            dx_[i] = np.dot(
                initial_part_pos[1, 2, i, :] - initial_part_pos[1, 0, i, :],
                (initial_part_pos[1, 2, i, :] - initial_part_pos[1, 0, i, :]).reshape(
                    (2, 1)
                ),
            )
            dy_[i] = np.dot(
                initial_part_pos[0, 1, i, :] - initial_part_pos[2, 1, i, :],
                (initial_part_pos[0, 1, i, :] - initial_part_pos[2, 1, i, :]).reshape(
                    (2, 1)
                ),
            )
            dx_[i] = dx_[i] ** 0.5
            dy_[i] = dy_[i] ** 0.5

        dx = np.mean(dx_)
        dy = np.mean(dy_)

        aa = (final_part_pos[1, 2, :, 0] - final_part_pos[1, 0, :, 0]) / 2 * dx
        bb = (final_part_pos[0, 1, :, 0] - final_part_pos[2, 1, :, 0]) / 2 * dy
        cc = (final_part_pos[1, 2, :, 1] - final_part_pos[1, 0, :, 1]) / 2 * dx
        dd = (final_part_pos[0, 1, :, 1] - final_part_pos[2, 1, :, 1]) / 2 * dy

        grad = np.zeros((Z, 2, 2))
        C = np.zeros((Z, 2, 2))
        C_inv = np.zeros((Z, 2, 2))
        for i in range(Z):
            grad[i, :, :] = [[aa[i], bb[i]], [cc[i], dd[i]]]
            C[i, :, :] = np.matrix(grad[i, :, :]).T @ np.matrix(grad[i, :, :])
            C_inv[i, :, :] = np.linalg.inv(C[i, :, :])

        rho = np.zeros((Z))
        nu = np.zeros((Z))
        tan = np.zeros((Z))

        for i in range(Z):
            rho[i] = np.dot(
                ini_norm_frame[i, :],
                (np.matrix(C_inv[i, :, :]) @ ini_norm_frame[i, :]).reshape((2, 1)),
            )
            rho[i] = 1 / (rho[i] ** 0.5)
            tan[i] = np.dot(
                (np.matrix(grad[i, :, :]) @ ini_tang_frame[i, :]),
                (np.matrix(grad[i, :, :]) @ ini_tang_frame[i, :]).reshape((2, 1)),
            )

            nu[i] = rho[i] / (tan[i] ** 0.5)

        tang_rep_rate[key] = tan
        rep_ratio[key] = nu
        norm_rep_rate[key] = rho

    return tang_rep_rate, rep_ratio, norm_rep_rate


# Calculate array of points that are locally maximal and the heat map of
# those points in the FTLE field
FTLE_ridge, max_ridge_pts = local_max(strongFTLE, 5, 0.3)

# Obtain the ridge lines as a dictionary of points
ridge_lines = create_lines(max_ridge_pts, 15)

# Calculate the moving frame with respect to the ridge lines
frame = create_frame(ridge_lines)
tangent_frame, normal_frame = frame

# Initialise a grid of particles at each point along the grid line
ridge_grid = initialise_points(ridge_lines, frame)

# Calculate the trajectories of all particles
ridge_particle_traj = ridge_flowline(ridge_grid, flowfield, frames)

# Calculate the Tangent, Normal repulsion rates and the repulsion ratio
tang, ratio, norm = perturbation_field(ridge_particle_traj, frame)

# Create a heat map of the Tangent, Normal repulsion rates and the repulsion ratio
tang_rep_field = np.zeros((Ny, Nx))
rep_ratio_field = np.zeros((Ny, Nx))
norm_rep_field = np.zeros((Ny, Nx))
for key in ridge_lines:
    for i in range(len(ridge_lines[key])):
        row, col = np.rint(ridge_lines[key][i])
        tang_rep_field[int(row), int(col)] = tang[key][i]
        rep_ratio_field[int(row), int(col)] = ratio[key][i]
        norm_rep_field[int(row), int(col)] = norm[key][i]


for key in ridge_lines:
    ridge_lines[key] = np.roll(ridge_lines[key], 1, axis=1)


### Set the figure parameters and plot the figures

tang_max = np.max(tang_rep_field)
ratio_max = np.max(rep_ratio_field)
max = np.max((ratio_max, tang_max))
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.set_title("Tangent Repulsion Rate")
ax2.set_title("Repulsion Ratio")

fig.tight_layout()
# , vmin=0, vmax=max
tangent = ax1.imshow(tang_rep_field)
fig.colorbar(tangent, ax=ax1, shrink=0.5, location="left", pad=0.15)

Repulsion_ratio = ax2.imshow(norm_rep_field)
fig.colorbar(Repulsion_ratio, ax=ax2, shrink=0.5, pad=0.15)


### Uncomment if you want to plot the Normal Repulsion Rate

"""fig, axes = plt.subplots()
normal = axes.imshow(norm_rep_field)
fig.colorbar(normal, ax=axes, shrink=0.5)
axes.set_title("Normal Repulsion Rate")"""

plt.show()
