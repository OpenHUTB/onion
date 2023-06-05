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
import numpy as np

from src import classify_units
from src import dataset
from src import plot_utils
from src import scores

path = "/data3/dong/data/brain/memory"
github_path = "https://raw.githubusercontent.com/deepmind/spatial_memory_pipeline/master/src/"
local_folder = "spatial_memory_pipeline"
filenames = ["classify_units.py", "dataset.py", "plot_utils.py", "scores.py"]

# get_ipython().system('cd {path}')
local_dir = os.path.join(path, local_folder)
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
# get_ipython().system('mkdir {path}/{local_folder}')
# get_ipython().system('touch {path}/{local_folder}/__init__.py')


# 下载四个.py文件（已经在src文件下有了，所以不需要）
# for name in filenames:
#   source = os.path.join(github_path, name)
#   dest = os.path.join(path, local_folder, name)
#   with requests.get(source, stream=True) as r, open(dest, 'wb') as w:
#     r.raise_for_status()
#     for chunk in r.iter_content(chunk_size=128):
#         w.write(chunk)
#     w.close()


# Load 800 trajectories of 500 steps each.

# dataset.download_figure2_files()
traj_ds = dataset.load_figure_2_trajectories()
ds = dataset.flatten_trajectories(traj_ds)

print(dataset.description(traj_ds))


# In[ ]:


# Compute classification thresholds for different kinds of units
# (this may take ~15 minutes).
#
# For each type of cell (HD cells, egoBVC cells and BVC cells) we compute
# resultant vectors associated with the representation and compare them
# to a threshold obtained from the 99-percentile resultant vector length
# fromo shuffled versions of the cell responses. See Supplementary methods
# section in the paper for details.

all_activities = np.concatenate(ds["predicted_outputs"][:3], axis=-1)

(is_hd, hd_threshold, hd_score, hd_rv_len, hd_rv_angles,  # HD-cell parameters
 is_ego, ego_threshold, ego_score, ego_dist, ego_ang,  # egoBVC parameters
 is_allo, allo_threshold, allo_score, allo_dist, allo_ang) = (  # BVC parameters
     classify_units.classify_units_into_representations(
         all_activities, ds["hd"], ds["pos"], ds["world_distances"], percentile=99)
 )

print("HD Resultant vector threshold:", hd_threshold)
print("HD units:", np.where(is_hd)[0])
print("EgoBVC Resultant vector threshold:", ego_threshold)
print("EgoBVC units:", np.where(is_ego)[0])
print("BVC Resultant vector threshold:", allo_threshold)
print("BVC units:", np.where(is_allo)[0])


# Types of cell per RNN.
#
# RNN-1 contains 32 units, RNN-2 128 units, RNN-3 128 units.

units_in_each_rnn = [range(32), range(32, 160), range(160, 288)]

for rnn_idx in range(3):
  units = units_in_each_rnn[rnn_idx]

  print(f"------ RNN-{rnn_idx + 1} -------",)
  print("\tHD %d (%.1f%%)" % (np.sum(is_hd[units]), 100 * np.mean(is_hd[units])))
  print("\tegoBVC %d (%.1f%%)" % (np.sum(is_ego[units]), 100 * np.mean(is_ego[units])))
  print("\tBVC %d (%.1f%%)" % (np.sum(is_allo[units]), 100 * np.mean(is_allo[units])))

  more_allo_than_ego = np.logical_and(allo_score > ego_score, is_ego)[units]
  print("\t\tBVC (allocentric) score > egoBVC score %d" % (np.sum(more_allo_than_ego)))


# In[ ]:


# Compute positional stability of the representations (panel F of figure 2).

rnn0_stability = scores.positional_correlations(ds["pos"], ds["hd"], ds["predicted_outputs"][0])
rnn1_stability = scores.positional_correlations(ds["pos"], ds["hd"], ds["predicted_outputs"][1])
rnn2_stability = scores.positional_correlations(ds["pos"], ds["hd"], ds["predicted_outputs"][2])


# In[ ]:


# Auxiliary functions for plotting figure

def rnn_label(index):
  return "RNN-%d" % (index + 1)


def rnn_polars(fig, x, y, text, ds, rnn_index, chosen_units, x_multiplier=1,
               text_offset=0.0, color="b"):
  """Polar plot used to represent HD cells."""
  polar_axs = fig.add_subplots_to_figure(
      1,
      len(chosen_units),
      x + 0.05 + 0.25 * (x_multiplier - 1),
      y,
      0.4,
      0.4,
      0.15,
      0.05,
      text=text,
      text_offset=-0.05 + text_offset,
      projection="polar")
  rv_scorer = scores.ResultantVectorHeadDirectionScorer(36)
  for i, unit in enumerate(chosen_units):
    polar_rm = rv_scorer.calculate_hd_ratemap(
        ds["hd"][:, 0],
        ds["predicted_outputs"][rnn_index][:, unit])
    rv_scorer.plot_polarplot([polar_rm],
                             ax=polar_axs[0, i], positive_lobe_color=color)
    polar_axs[0, i].set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    polar_axs[0, i].set_xticklabels(["E", "N", "W", "S"])
    polar_axs[0, i].xaxis.set_tick_params(pad=-5)
    polar_axs[0, i].set_yticks([0.5])
    polar_axs[0, i].set_ylim(0, 1)
    polar_axs[0, i].spines["polar"].set_linewidth(1)
    polar_axs[0, i].spines["polar"].set_color("k")


def rnn_ratemaps(fig, x, y, text, ds, rnn_index, chosen_units, x_multiplier=1,
                 cmap="jet", text_offset=0.0):
  """Ratemap activity plot, including per-octant ratemaps."""
  axs = fig.add_subplots_to_figure(1, len(chosen_units),
                                   x, y, 0.5 * x_multiplier, 0.5,
                                   0.05, 0.05,
                                   text_offset=text_offset,
                                   text=text)
  for i, unit in enumerate(chosen_units):
    ratemap = scores.calculate_ratemap(
        ds["pos"][:, 0],
        ds["pos"][:, 1],
        ds["predicted_outputs"][rnn_index][:, unit],
        hd=ds["hd"])[0]
    rmpo = plot_utils.octant_ratemap_to_rgb(ratemap,
                                            per_octant=True,
                                            cmap=cmap)
    plot_utils.imshow(axs[0, i], rmpo)
    axs[0, i].set_title("%s:%d" % (rnn_label(rnn_index), unit))


def place_cell_ratemaps(fig, x, y, text, ds, mmap_index, chosen_units, x_multiplier=1,
                        cmap="jet", text_offset=0.0, size=0.5):
  """Plot ratemap for place cells."""
  octant_axs = fig.add_subplots_to_figure(
      1,
      len(chosen_units),
      x,
      y,
      size * x_multiplier,
      size,
      0.05,
      0.05,
      text_offset=text_offset,
      text=text)
  for i, unit in enumerate(chosen_units):
    activations = np.exp(ds["log_ps_given_y"][mmap_index][:, unit])
    ratemap = scores.calculate_ratemap(
        ds["pos"][:, 0], ds["pos"][:, 1], activations, hd=ds["hd"])[0]
    rmpo = plot_utils.octant_ratemap_to_rgb(
        ratemap, per_octant=False, cmap=cmap)
    plot_utils.imshow(octant_axs[0, i], rmpo)
    octant_axs[0, i].set_title("Mem:%d" % (unit,))


# In[ ]:


# Plot the figure.

# Indices of the units to be displayed for each type of cell
chosen_hd_cells = [0, 1, 2, 3, 4]
chosen_ego_cells = [14, 56, 73, 76, 81]
chosen_allo_cells = [57, 64, 87, 104, 119]
chosen_pc_cells = [43, 45, 96, 508, 510]

# Position for each subplot
rnn1_rm_x, rnn1_rm_y = 0.2, 0.2
rnn2_rm_x, rnn2_rm_y = 0.2, 1.5
rnn3_rm_x, rnn3_rm_y = 0.2, 2.2
pc_rm_x, pc_rm_y = 0.2, 2.9
spat_stab_x, spat_stab_y = 0.35, 3.6
all_rv_x, all_rv_y = 1.9, 3.6
rnn1_rv_x, rnn1_rv_y = 0.35, 5.0
ego_vs_allo_x, ego_vs_allo_y = 1.9, 5.0

panel_w, panel_h = 1.0, 1.0

# Create the figure
fig = plot_utils.SuperFigure(3.0, 7.5, dpi=180)
cmap = "jet"

# Plot some HD cells, their polar plot and ratemaps to show spatial uniformity
rnn_polars(fig, rnn1_rm_x, rnn1_rm_y, "A)", ds, rnn_index=0, chosen_units=chosen_hd_cells)
rnn_ratemaps(fig, rnn1_rm_x, rnn1_rm_y + 0.6, "B)", ds, rnn_index=0, chosen_units=chosen_hd_cells, cmap=cmap)
# Plot some ego-BVC cell ratemaps
rnn_ratemaps(fig, rnn2_rm_x, rnn2_rm_y, "C)", ds, rnn_index=1, chosen_units=chosen_ego_cells, cmap=cmap)
# Plot some allo-BVC cell ratemaps
rnn_ratemaps(fig, rnn3_rm_x, rnn3_rm_y, "D)", ds, rnn_index=2, chosen_units=chosen_allo_cells, cmap=cmap)
# Plot some place-cell ratemaps from memory activations
place_cell_ratemaps(fig, pc_rm_x, pc_rm_y, "E)", ds, mmap_index=1, chosen_units=chosen_pc_cells, cmap=cmap)

# Plot a histogram of spatial stability for each RNN
stability_ax = fig.add_subplots_to_figure(1, 1,
                                          spat_stab_x, spat_stab_y, panel_w, panel_h,
                                          text="F)", text_offset=-0.15)[0, 0]
bins = np.linspace(0, 1, 11)
plot_utils.prop_hist(stability_ax, rnn0_stability, bins, color="b", alpha=0.7, label=rnn_label(0))
plot_utils.prop_hist(stability_ax, rnn1_stability, bins, color="g", alpha=0.7, label=rnn_label(1))
plot_utils.prop_hist(stability_ax, rnn2_stability, bins, color="r", alpha=0.7, label=rnn_label(2))
stability_ax.set_xlim(0, 1)
stability_ax.set_ylim(0, 1)
stability_ax.set_xlabel("Spatial stability")
stability_ax.set_ylabel("Proportion of cells")
stability_ax.legend(loc="upper left")
stability_ax.grid(linewidth=1, linestyle="-", color="#333333", alpha=0.1)
plot_utils.setup_spines(stability_ax)

## Plot a histogram of directional resultant-vector length for each RNN
prop_hds_ax = fig.add_subplots_to_figure(1, 1,
                                         all_rv_x, all_rv_y, panel_w, panel_h,
                                         text="G)", text_offset=-0.15)[0, 0]
bins = np.linspace(0, 1, 11)
plot_utils.prop_hist(prop_hds_ax, hd_rv_len[:32], bins, color="b", alpha=0.7, label=rnn_label(0))
plot_utils.prop_hist(prop_hds_ax, hd_rv_len[32:160], bins, color="g", alpha=0.7, label=rnn_label(1))
plot_utils.prop_hist(prop_hds_ax, hd_rv_len[160:], bins, color="r", alpha=0.7, label=rnn_label(2))
prop_hds_ax.set_ylim(0, 1)
prop_hds_ax.set_xlim(0, 1.0)
prop_hds_ax.set_xlabel("Resultant vector length")
prop_hds_ax.set_ylabel("Proportion of cells")
prop_hds_ax.legend(loc="upper right")
prop_hds_ax.grid(linewidth=1, linestyle="-", color="#333333", alpha=0.1)
plot_utils.setup_spines(prop_hds_ax)


## RNN 1 - RV polar plot
ax = fig.add_subplots_to_figure(1, 1,
                                rnn1_rv_x, rnn1_rv_y, panel_w, panel_h,
                                text="H)", text_offset=-0.15, projection="polar")[0, 0]

rnn0_rv_angles, rnn0_rv_lengths = hd_rv_angles[:32], hd_rv_len[:32]

for rvang, rvlen in zip(rnn0_rv_angles, rnn0_rv_lengths):
  ax.plot((rvang, rvang), (0, rvlen), color="#4C72B0")
ax.grid(linewidth=1, linestyle=":", color="#333333")
ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
ax.set_xticklabels(["E", "N", "W", "S"])
ax.set_ylim(0, 1.0)
ax.set_yticklabels(["", "%.2f" % hd_threshold, "1"])
ax.set_yticks([0, hd_threshold])
ax.grid(linewidth=1, linestyle="-", color="#333333", alpha=0.1)
plot_utils.setup_spines(ax)


# egoBCV scores vs BVC (allocentric) scores
ego_vs_allo_ax = fig.add_subplots_to_figure(1, 1,
                                            ego_vs_allo_x, ego_vs_allo_y, panel_w, panel_h,
                                            text="I)", text_offset=-0.15)[0, 0]
ego_vs_allo_ax.scatter(ego_score[32:160], allo_score[32:160], color="g", label=rnn_label(1), alpha=0.6, marker=".")
ego_vs_allo_ax.scatter(ego_score[160:], allo_score[160:], color="r", label=rnn_label(2), alpha=0.6, marker=".")
ego_vs_allo_ax.scatter(ego_score[:32], allo_score[:32], color="b", label=rnn_label(0), alpha=0.6, marker=".")
ego_vs_allo_ax.axvline(ego_threshold, color="k", alpha=0.8, linestyle="dashed")
ego_vs_allo_ax.axhline(allo_threshold, color="k", alpha=0.8, linestyle="dashed")
ego_vs_allo_ax.set_xlim(0, 0.5)
ego_vs_allo_ax.set_ylim(0, 0.5)

ego_vs_allo_ax.set_xticks(np.arange(0., 0.51, 0.1))
ego_vs_allo_ax.set_yticks(np.arange(0., 0.51, 0.1))

ego_vs_allo_ax.set_xlabel("Egocentric-BVC score")
ego_vs_allo_ax.set_ylabel("Allocentric-BVC score")
ego_vs_allo_ax.legend()
ego_vs_allo_ax.grid(linewidth=1, linestyle="-", color="#333333", alpha=0.1)
plot_utils.setup_spines(ego_vs_allo_ax)


# 
