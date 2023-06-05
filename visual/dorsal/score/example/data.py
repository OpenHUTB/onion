#!/usr/bin/env python
# coding: utf-8

# This example is primarily concerned with loading raw data. This data is often not fully pre-processed (e.g. neuroids that we don't trust are not filtered, repetitions are not averaged, hard stimuli are not pre-selected etc.).
# 
# If you only want to compare data with each other, you are probably better off 
# using benchmarks directly (e.g. `from brainscore import benchmarks; benchmarks.load('dicarlo.Majaj2015')`) or
# loading the data through benchmarks (e.g. `from brainscore import benchmarks; benchmarks.load_assembly('dicarlo.Majaj2015')`).
# 
# `jupyter nbconvert --to script data.ipynb`


#%% ### Neural assembly
# We can load data (called "assembly") using the `get_assembly` method.
# In the following, we load neural data from the DiCarlo lab, published in Majaj2015.
#
import brainscore
import pickle
import os

filename = 'dicarlo.MajajHong2015.public'
# filename = 'dicarlo.Majaj2015'

# saved_filename = os.path.join('/data2/whd/workspace/sot/hart/score/data', filename)
# if os.path.exists(saved_filename) == False:
#     os.mkdir('data')

# if os.path.exists(saved_filename) == False:
#     neural_data = brainscore.get_assembly(filename)
#     with open(saved_filename, 'wb') as f:
#         pickle.dump(neural_data, f, protocol=pickle.HIGHEST_PROTOCOL)
# else:
#     with open(saved_filename, 'rb') as f:
#         neural_data = pickle.load(f)
neural_data = brainscore.get_assembly(name=filename)
# neural_data.load()


# This gives us a NeuronRecordingAssembly, a sub-class of xarray DataArray.
# The behavioral and neural assemblies are always handled with the xarray framework.
# xarray data is basically a multi-dimensional table with annotating coordinates, similar to pandas. 
# More info here: http://xarray.pydata.org.
# Coming back to the neural assembly `dicarlo.Majaj2015`, 
# it is structured into the dimensions `neuroid x presentation x time_bin`.
# `neuroid` is a MultiIndex containing information about the recording site, such as the animal and the region.
# `presentation` refers to the single presentation of a stimulus with coords annotating 
# e.g. the image_id and the repetition.
# Finally, `time_bin` informs us about the time in milliseconds from when neural responses were collected.
# At the current stage, `dicarlo.Majaj2015` contains only the 70-170 ms time bin.

# The data is in a raw format, but typically we use a pre-processed version.
# We want to:
compact_data = neural_data.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')  # (1) average across repetitions
compact_data = compact_data.sel(region='IT')  # (2) filter neuroids from IT region
compact_data = compact_data.squeeze('time_bin')  # (3) get rid of the scalar time_bin dimension
compact_data = compact_data.T  # (4) and reshape into presentation x neuroid


# The data now contains 2560 images and the responses of 296 neuroids.
print(compact_data.shape)


# Note that the data used for benchmarking is typically already pre-processed.
# For instance, the target assembly for the `dicarlo.Majaj2015` benchmark is the same as our pre-processed version here:

from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark

benchmark = MajajHongITPublicBenchmark()
benchmark_assembly = benchmark._assembly
print(benchmark_assembly.shape)


# We can also easily filter neuroids from a specific region, such as IT.
# By selecting only that region, we keep only the 168 neuroids from that region.
# print(compact_data.sel(region='IT').shape)


#%% ### Stimulus Set

# You may have noticed the attribute `stimulus_set` in the previous assembly.
# A stimulus set contains the stimuli that were shown to measure the neural recordings.
# Specifically, this entails e.g. the image_id and the object_name, packaged in a pandas DataFrame.
stimulus_set = neural_data.attrs["stimulus_set"]
print(stimulus_set[:3])

# We can also directly retrieve the image using the `get_image` method.
image_path = stimulus_set.get_image(stimulus_set['image_id'][0])
print(image_path)


# Images are automatically downloaded locally and can thus be loaded and displayed directly.
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot, image
img = image.imread(image_path)
pyplot.imshow(img)
pyplot.show()

