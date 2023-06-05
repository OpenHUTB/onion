#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[ ]:


# Download code to import locally.
# The '/content' path is meant for cloud-based Google Colab.
# If running on a local Colab, you may want to change that to a different
# local path.
import os
import requests
import shutil

path = "/content"
github_path = "https://raw.githubusercontent.com/deepmind/spatial_memory_pipeline/master/src/"
local_folder = "spatial_memory_pipeline"
filenames = ["dataset.py", "plot_utils.py"]

get_ipython().system('cd {path}')
get_ipython().system('mkdir {path}/{local_folder}')
get_ipython().system('touch {path}/{local_folder}/__init__.py')

for name in filenames:
  source = os.path.join(github_path, name)
  dest = os.path.join(path, local_folder, name)
  with requests.get(source, stream=True) as r, open(dest, 'wb') as w:
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=128):
        w.write(chunk)
    w.close()


# In[ ]:


# Imports.

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox

import numpy as np
import scipy.stats

from spatial_memory_pipeline import dataset
from spatial_memory_pipeline import plot_utils


# In[ ]:


# Functions to unroll the model RNNs.

def sigmoid(x):
  return 1. / (1.+np.exp(-x))


def softmax(x, axis=-1):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.amax(x, axis=axis, keepdims=True))
  return e_x / e_x.sum(axis=axis, keepdims=True)


def step_of_lstm(state, inputs, w, b):
  forget_bias = 1
  prev_hidden, prev_cell = state
  inputs_and_hidden = np.concatenate([inputs, prev_hidden], axis=-1)
  gates = np.matmul(inputs_and_hidden, w) + b

  # i = input_gate, j = next_input, f = forget_gate, o = output_gate
  i, j, f, o = np.split(gates, 4, axis=-1)

  forget_mask = sigmoid(f + forget_bias)
  next_cell = forget_mask * prev_cell + sigmoid(i) * np.tanh(j)
  cell_output = next_cell
  next_hidden = sigmoid(cell_output) * sigmoid(o)

  return next_hidden, (next_hidden, next_cell)


def get_visual_correction_codes(parameters, visual_embeddings):
  """Calculates visual correction codes for a set of visual embeddings."""
  # hd_embeddings corresponds to m^{(x)} in methods
  hd_embeddings = parameters["integrator/memory_map_0/column_context_3"]
  # memory_of_visual_embeddings corresponds to m^{(y)} in methods
  memory_of_visual_embeddings = parameters["integrator/memory_map_0/column_context_2"]
  # gamma is the correction inverse temperature
  gamma = np.exp(parameters["integrator/multi_integrator_nets/mmap_0_context_2_query_beta"])
  w_s = softmax(gamma * np.matmul(visual_embeddings, memory_of_visual_embeddings.T),
                axis=-1)
  correction_codes = np.matmul(w_s, hd_embeddings)
  return correction_codes


def unroll_rnn_with_fixed_vel(initial_state, ang_vel, n_steps, rnn_step_func):
  """Unroll one rnn with a constant velocity input."""
  B = initial_state.shape[0]
  vel_inputs = np.tile(np.asarray([[np.cos(ang_vel), np.sin(ang_vel)]]),
                       (B, 1))
  # Velocities are scaled to increase range
  vel_inputs *= 10
  state_traj = list()
  state_traj.append(initial_state)
  rnn_state = (initial_state, initial_state)
  for t in range(n_steps):
    output, rnn_state = rnn_step_func(rnn_state, vel_inputs)
    state_traj.append(output)
  return np.asarray(state_traj)


# In[ ]:


# Auxiliary functions.

def take_if_farther_than(points, distance_f, threshold):
  """Filters points choosing a subset such that they are all at least
  `threshold` distance from each other.

  Returns a tuple with the indices of the filtered points, and their values.
  """
  chosen_points = list()
  chosen_indices = list()
  for i, p in enumerate(points):
    if i == 0:
      chosen_points.append(p)
      chosen_indices.append(i)
    else:
      min_d = np.inf
      for ch in chosen_points:
        if distance_f(p, ch) < min_d:
          min_d = distance_f(p, ch)
      if min_d > threshold:
        chosen_points.append(p)
        chosen_indices.append(i)
  return chosen_indices, np.asarray(chosen_points)


def euclidean_dist(a, b):
  return np.linalg.norm(a-b)


def autocovariance(state_traj, ignore_n_first=0, max_lag=50):
  T, B, D = state_traj.shape
  autocov = list()
  for b in range(B):
    traj = state_traj[:, b, :]
    mean = np.mean(traj, axis=0)
    ctraj = traj - mean[np.newaxis, ...]
    var = np.var(ctraj, axis=0)
    covar = np.matmul(ctraj, ctraj.T)
    traj_autocov = np.asarray([np.mean([covar[i, i+l] for i in range(ignore_n_first, T-l)]) for l in range(max_lag)])
    autocov.append(traj_autocov)
  return np.asarray(autocov)


# ### Download data

# In[ ]:


# Download parameters, inputs, and hd cell stats.

dataset.download_figure3_files()
parameters = dataset.load_figure_3_parameters()
visual_inputs = dataset.load_figure_3_visual_inputs()
hd_stats = dataset.load_figure_3_hd_stats()

p_w = parameters["integrator/multi_integrator_nets/prediction_rnn_0/w_gates"]
p_b = parameters["integrator/multi_integrator_nets/prediction_rnn_0/b_gates"]
c_w = parameters["integrator/multi_integrator_nets/correction_rnn_0/w_gates"]
c_b = parameters["integrator/multi_integrator_nets/correction_rnn_0/b_gates"]
step_of_prediction = lambda state, inputs: step_of_lstm(state, inputs, p_w, p_b)
step_of_correction = lambda state, inputs: step_of_lstm(state, inputs, c_w, c_b)

rnn_initial_states = parameters["integrator/memory_map_0/column_context_3_written"]

correction_code = get_visual_correction_codes(parameters, visual_inputs["embeddings"])


# In[ ]:


# Select hd cells, sorted by preferred angle.

hd_cutoff = 0.56
n_units = len(hd_stats["rnn0_rv_angles"])
hd_ordering = np.argsort(hd_stats["rnn0_rv_angles"])
hd_ordering = hd_ordering[hd_stats["rnn0_rv_lengths"][hd_ordering] > hd_cutoff]
n_hd_cells = len(hd_ordering)


# ### Dynamics without corrections

# In[ ]:


# Trajectory with piecewise-constant velocities.

T = 1000
ang_vel = np.pi / 20
n_inputs = 2

# Take a valid initial state
initial_state_index = 8  # Select a state where the angle starts at 0
initial_state = rnn_initial_states[initial_state_index]
state = (initial_state, initial_state)

# Construct velocity inputs
vel = [ang_vel] * (T // 4) + [0] * (T // 4) + [-ang_vel/2.0] * (T // 2)
vel = np.asarray([[np.cos(v), np.sin(v)] for v in vel])

# Ground-truth integrated velocities
true_angular_offset = np.cumsum(np.arctan2(vel[:-1, 1], vel[:-1, 0]))

# The network receives the inputs scaled by 10 (to increase the range)
vel = vel * 10
trajectory = []
for i in range(T):
  output, state = step_of_prediction(state, vel[i])
  trajectory.append(output)
trajectory = np.asarray(trajectory)
demo_trajectory = trajectory[:, hd_ordering]


# In[ ]:


# Calculate PCA space.

# Remove repeated points
pca_from = take_if_farther_than(trajectory, euclidean_dist, 0.1)[1]
pca_from = np.asarray(pca_from)
cov_mat = np.cov(pca_from.T)
cov_mat += np.eye(cov_mat.shape[0]) * 1e-4

# Covariance matrix of all units with 16 times the variance for random inits
diag_cov_mat = cov_mat * np.eye(cov_mat.shape[0]) * 16
# Distribution of random initialisations
initial_dist = scipy.stats.multivariate_normal(mean=np.mean(pca_from, axis=0),
                                               cov=diag_cov_mat)

# PCA
cov = np.cov(pca_from[:, hd_ordering].T)
mean = np.mean(pca_from[:, hd_ordering].T, axis=1)
eiglambda, eigvec = np.linalg.eig(cov)
def project_on_pca(x):
  return np.matmul(x - mean, eigvec[:, :2]).real


# In[ ]:


# Trajectories with zero velocity, from different random initialisations.

T = 2000
B = 100
ang_vel = 0.0
vel = np.asarray([[np.cos(ang_vel), np.sin(ang_vel)]] * B)
vel = vel * 10

trajectories = list()
for k in range(1000 // B):
  print(".", end="")
  state = initial_dist.rvs(B)
  out_traj = [state]
  state = (state, state)
  for _ in range(T):
    output, state = step_of_prediction(state, vel)
    out_traj.append(output)
  trajectories.append(np.rollaxis(np.asarray(out_traj), 1))

trajectories = np.concatenate(trajectories)
trajectories_ordered = trajectories[:, :, hd_ordering]


# In[ ]:


# Find attractor states

all_pca = project_on_pca(trajectories_ordered[:, 3:, :])
minpca, maxpca = np.amin(all_pca, axis=(0, 1)), np.amax(all_pca, axis=(0, 1))
pca_range = maxpca - minpca
pcalims = np.stack([np.abs(minpca - 0.1 * pca_range), np.abs(maxpca + 0.1 * pca_range)]).max(axis=0)
xlim_pca = [-pcalims[0], pcalims[0]]
ylim_pca = [-pcalims[1], pcalims[1]]
chosen = list()
pca = project_on_pca(trajectories_ordered[:, -1, :])
converged_angles_on_pca = np.arctan2(pca[:, 1], pca[:, 0])
for i, ang in zip(converged_angles_on_pca.argsort(), converged_angles_on_pca[converged_angles_on_pca.argsort()]):
  if not chosen:
    chosen.append(i)
    last_ang_added = ang
  elif (ang - last_ang_added) > 0.1*np.pi/360 * 2:
    chosen.append(i)
    last_ang_added = ang

attractor_states = trajectories[chosen, -1, :]


# ### Dynamics with visual corrections

# In[ ]:


# Run the correction + dynamics for T steps from the attractor states

T = 200
B = attractor_states.shape[0]
I = 512  # Different image corrections

ang_vel = 0.0
vel_inputs = np.asarray([np.cos(ang_vel), np.sin(ang_vel)]) * 10.0
vel_inputs = np.tile(vel_inputs[np.newaxis, :], (B, 1))

traj_pca_points = list()

for i in range(I):
  rnn_state = attractor_states
  correction_code_batch = np.tile(correction_code[[i], :], (B, 1))
  image = visual_inputs["images"][i]

  state_traj = list()
  state_traj.append(rnn_state)
  rnn_state = (rnn_state, rnn_state)
  for t in range(T):
    out, rnn_state = step_of_correction(rnn_state, correction_code_batch)
    out, rnn_state = step_of_prediction(rnn_state, vel_inputs)
    state_traj.append(out)
  state_traj = np.asarray(state_traj)

  pc = project_on_pca(state_traj[:, :, hd_ordering])
  traj_pca_points.append(pc)

traj_pca_points = np.asarray(traj_pca_points)


# In[ ]:


# Time to convergence

time_to_single_attractor = np.asarray([np.inf] * traj_pca_points.shape[0])
for i in range(traj_pca_points.shape[0]):
  for t in range(traj_pca_points.shape[1]):
    att_inds, att_points = take_if_farther_than(traj_pca_points[i, t],
                                                euclidean_dist,
                                                0.1)
    if len(att_inds) == 1:
      time_to_single_attractor[i] = t
      break

examples_in_bin_1 = np.where(np.logical_and(time_to_single_attractor > 0,
                                            time_to_single_attractor < 10))[0]
examples_in_bin_final = np.where(time_to_single_attractor > 200)[0]


# ### Plot figure 3

# In[ ]:


figsize_y = 5.05
fig = plot_utils.SuperFigure(6.2, figsize_y, dpi=180)

simulation_panel_coords = [0.45, 0.6, 4.1, 0.5]
groundtruth_simulation_panel_coords = [0.45, 0.2, 4.1, 0.3]
pca_panel_coords = [5.1, 0.2, 1.0, 1.0]
delta_panel_coords = [0.3, 1.6, 1.0, 1.0]
pca_random_panel_coords = [1.9, 1.6, 1.0, 1.0]
pca_attractor_panel_coords = [3.5, 1.6, 1.0, 1.0]
autocorr_panel_coords = [5.1, 1.6, 1.0, 1.0]
ttc_panel_coords = [0.3, 3.4, 2.0, 1.2]

# Panel A. Simulation of integration of velocities
ax = fig.add_subplots_to_figure(1, 1, *simulation_panel_coords,
                                text=None, text_offset=-0.15)[0, 0]
ax.matshow(demo_trajectory.T, cmap="coolwarm", aspect=3.5)
ax.grid("off")
ax.set_yticks(np.arange(0, n_hd_cells+1, 10))
ax.set_ylabel("Unit")
ax.set_xticks(np.arange(0, 1000+1, 100))
ax.xaxis.tick_bottom()
ax.set_xlabel("Integration steps")
plot_utils.setup_spines(ax)

# Panel A. Ground truth angular offset
unit0_angular_offset = -np.pi
display_angular_offset = np.arctan2(np.sin(true_angular_offset + unit0_angular_offset),
                                    np.cos(true_angular_offset + unit0_angular_offset))

ax = fig.add_subplots_to_figure(1, 1, *groundtruth_simulation_panel_coords,
                                hspace=0.3, text="A)", text_offset=-0.3)[0, 0]
ax.plot(display_angular_offset, ".")
ax.set_xticks(np.arange(0, 1000+1, 100))
ax.set_xlim(0, 1000)
ax.set_ylim(-np.pi, np.pi)
ax.set_yticks([-np.pi, 0., np.pi])
ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

ax.set_xticks(np.arange(0, 1000+1, 100))
ax.set_xticklabels([])
ax.xaxis.tick_bottom()
ax.set_ylabel("True angle")
plot_utils.setup_spines(ax)

# Panel B. PCA of simulation
ax = fig.add_subplots_to_figure(1, 1, *pca_panel_coords,
                                text="B)", text_offset=-0.15)[0, 0]
pca_values = project_on_pca(demo_trajectory)
ax.scatter(pca_values[:, 0], pca_values[:, 1], alpha=0.1, color="grey")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_xlim(xlim_pca)
ax.set_ylim(ylim_pca)
plot_utils.setup_spines(ax)


# Panel C. Norm change in state vs time
ax = fig.add_subplots_to_figure(1, 1, *delta_panel_coords,
                                text="C)", text_offset=-0.15)[0, 0]
delta = np.linalg.norm(trajectories_ordered[:, :-1, :] - trajectories_ordered[:, 1:, :], axis=-1)
plt.plot(np.quantile(delta, .95, axis=0)[:100], linewidth=1, color="b")
plt.plot(np.quantile(delta, .05, axis=0)[:100], linewidth=1, color="b")
plt.plot(delta.mean(axis=0)[:100], linewidth=2, color="b")
ax.set_xlim(0, 100)
ax.set_ylim(0, 2)
ax.set_xlabel("Steps of dynamics")
ax.set_ylabel("Change in activations")
plot_utils.setup_spines(ax)

# Panel D. PCA of simulations from random initialisations
ax = fig.add_subplots_to_figure(1, 1, *pca_random_panel_coords,
                                text="D)", text_offset=-0.15)[0, 0]
initializations = trajectories_ordered[:, 0, :]
pca_values = project_on_pca(initializations)
plt.scatter(pca_values[:, 0], pca_values[:, 1], alpha=0.1, color="grey")

not_yet_converged_states = trajectories_ordered[:, 10, :]
pca_values = project_on_pca(not_yet_converged_states)
plt.scatter(pca_values[:, 0], pca_values[:, 1], alpha=0.1, color="green")

converged_states = trajectories_ordered[:, -1, :]
pca_values = project_on_pca(converged_states)
plt.scatter(pca_values[:, 0], pca_values[:, 1], alpha=1.0, color="k", marker="x")

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_xlim(xlim_pca)
ax.set_ylim(ylim_pca)
plot_utils.setup_spines(ax)

# Panel E. PCA CW and CCW cyclic attractor
init = 0
ax = fig.add_subplots_to_figure(1, 1, *pca_attractor_panel_coords,
                                text="E)", text_offset=-0.15)[0, 0]
for (i, ang_vel, col, label) in zip([11, 41], [-0.16, 0.16], ["r", "b"],
                                    [r"$\omega=\frac{\pi}{20}$", r"$\omega=-\frac{\pi}{20}$"]):
  vel_state_traj = unroll_rnn_with_fixed_vel(trajectories[i, [0], :], ang_vel, 50, step_of_prediction)
  pca_values = project_on_pca(vel_state_traj[:, :, hd_ordering])
  vels = pca_values[1:, :, :] - pca_values[:-1, :, :]
  alpha = 0.5
  for b in range(vel_state_traj.shape[1]):
    plt.scatter(pca_values[0, b, 0], pca_values[0, b, 1], alpha=1.0, color=col)
    ax.quiver(pca_values[init:-1, b, 0], pca_values[init:-1, b, 1], vels[init:, b, 0], vels[init:, b, 1],
              scale_units="xy", scale=1.0, color=col, alpha=alpha, width=0.005, headwidth=8, label=label)

ax.legend(loc="lower right")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_xlim(xlim_pca)
ax.set_ylim(ylim_pca)
plot_utils.setup_spines(ax)

# Panel F. Autocorrelation plot
ax = fig.add_subplots_to_figure(1, 1, *autocorr_panel_coords,
                                text="F)", text_offset=-0.15)[0, 0]

for ang_vel, col, label in zip([-0.16, 0.16], ["r", "b"], [r"$\omega=\frac{\pi}{20}$", r"$\omega=-\frac{\pi}{20}$"]):
  state_traj = unroll_rnn_with_fixed_vel(trajectories[:, 0, :], ang_vel, 200, step_of_prediction)
  autocovs = autocovariance(state_traj)
  ax.plot(autocovs.mean(axis=0)[:100], linewidth=1, color=col, label=label)

ax.set_xlabel("Steps of dynamics")
ax.set_ylabel("Autocovariance")
ax.legend(loc="lower right")
plot_utils.setup_spines(ax)

# Panel G. Time to convergence to a single attractor point while correctiong visually
T = 200
ax = fig.add_subplots_to_figure(1, 1, *ttc_panel_coords)[0, 0]
fig.add_text(ttc_panel_coords[0]-0.15, ttc_panel_coords[1]-0.30, text="G)")

nbins = 20
ax.hist(time_to_single_attractor, bins=np.linspace(0, T, nbins))
ax.bar(T, np.sum(np.isinf(time_to_single_attractor)), width=T/nbins, color="r", align="edge")
ax.set_xlim(0, 210)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xticklabels(["0", "50", "100", "150", "  >200"])

# Plot the quick-convergence visual image examples
im_width = 40
for column, image_id in enumerate(examples_in_bin_1[:3]):
  im = OffsetImage(visual_inputs["images"][image_id], zoom=0.8)
  im.image.axes = ax
  xy = (5, 250)
  xybox = (50 + im_width * column, 250)
  if column == 0:
    arrowprops = dict(arrowstyle="->", color="k")
  else:
    arrowprops = None
  ab = AnnotationBbox(im, xy,
                      xybox=xybox,
                      xycoords="data",
                      boxcoords="data",
                      box_alignment=(0.5, 0.5),
                      pad=0.1,
                      bboxprops=dict(color="#4C72B0"),
                      arrowprops=arrowprops)
  ax.add_artist(ab)

# Plot the non-convergent visual image examples
for column, image_id in enumerate(examples_in_bin_final[:3]):
  im = OffsetImage(visual_inputs["images"][image_id], zoom=0.8)
  im.image.axes = ax
  xy = (205, 100)
  xybox = (150 - im_width * column, 100)
  if column == 0:
    arrowprops = dict(arrowstyle="->", color="k")
  else:
    arrowprops = None
  ab = AnnotationBbox(im, xy,
                      xybox=xybox,
                      xycoords="data",
                      boxcoords="data",
                      box_alignment=(0.5, 0.5),
                      pad=0.1,
                      bboxprops=dict(color="r"),
                      arrowprops=arrowprops)
  ax.add_artist(ab)

plot_utils.setup_spines(ax)
ax.set_xlabel("Time to convergence")
ax.set_ylabel("Proportion of images")
ax.set_ylim(0, 384)
ax.set_yticks([0, 128, 256, 384])
ax.set_yticklabels([0.0, 0.25, 0.5, 0.75])

# Panels H and I. State trajectories (in PC-space) correcting with 3 particular images
# Different colours show different initial rnn states
colors = get_cmap("tab20b").colors
best_examples = np.concatenate((examples_in_bin_1[:2], examples_in_bin_final[:1]))
for ex_i, (ex, label) in enumerate(zip(best_examples, ["H)", "", "I)", ""])):
  corr_att1_x = 2.7 + ex_i * 1.2
  ax = fig.add_subplots_to_figure(1, 1,
                                  corr_att1_x + 0.20, ttc_panel_coords[1]-0.4, 0.5, 0.5,
                                  text=label, text_offset=[-0.35, 0.1])[0, 0]
  ax.imshow(visual_inputs["images"][ex])
  ax.set_axis_off()
  plot_utils.setup_spines(ax)

  ax = fig.add_subplots_to_figure(1, 1,
                                  corr_att1_x, ttc_panel_coords[1] + 0.3, 0.9, 0.9,
                                  text="", text_offset=-0.15)[0, 0]
  B = attractor_states.shape[0]
  rnn_state = attractor_states
  correction_code_batch = np.tile(correction_code[[ex], :], (B, 1))

  state_traj = list()
  state_traj.append(rnn_state)
  rnn_state = (rnn_state, rnn_state)
  for t in range(T):
    if t % 1 == 0:
      out, rnn_state = step_of_correction(rnn_state, correction_code_batch)
    out, rnn_state = step_of_prediction(rnn_state, vel_inputs)
    state_traj.append(out)
  state_traj = np.asarray(state_traj)

  pc = project_on_pca(state_traj[:, :, hd_ordering])
  vels = pc[1:, :, :] - pc[:-1, :, :]

  for b in range(B):
    col = colors[b % len(colors)]
    plt.scatter(pc[0, b, 0], pc[0, b, 1], alpha=1.0, color=col)
    ax.quiver(pc[:-1, b, 0], pc[:-1, b, 1], vels[:, b, 0], vels[:, b, 1],
              scale_units="xy", scale=1.0, color=col, width=0.005, headwidth=8)
    plt.scatter(pc[-1, b, 0], pc[-1, b, 1], alpha=1.0, color="k", marker="x")

  ax.set_xlabel("PC 1")
  if ex_i == 0:
    ax.set_ylabel("PC 2")
  ax.set_xlim(xlim_pca)
  ax.set_ylim(ylim_pca)
  ax.set_aspect(1.0)
  plot_utils.setup_spines(ax)

