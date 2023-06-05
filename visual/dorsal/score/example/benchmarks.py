# coding: utf-8

# Benchmarks consist of a target assembly and a metric to compare assemblies.
# They accept a source assembly to compare against and yield a score.

#%% Pre-defined benchmarks
# Brainscore defines benchmarks, which can be run on brain models.
# To implement a model, the BrainModel interface has to be implemented by the model to be tested.
# A very simple implementation could look like this:

import numpy as np
from typing import List, Tuple
from brainscore.benchmarks.screen import place_on_screen

from brainscore.model_interface import BrainModel
from brainio_base.assemblies import DataAssembly

class RandomITModel(BrainModel):
    def __init__(self):
        self._num_neurons = 50
        # to note which time we are recording
        self._time_bin_start = None
        self._time_bin_end = None
    
    def look_at(self, stimuli, **kwargs):
        print("Looking at {len(stimuli)} stimuli")
        rnd = np.random.RandomState(0)
        recordings = DataAssembly(rnd.rand(len(stimuli), self._num_neurons, 1),
                              coords={'image_id': ('presentation', stimuli['image_id']),
                                      'object_name': ('presentation', stimuli['object_name']),
                                      'neuroid_id': ('neuroid', np.arange(self._num_neurons)),
                                      'region': ('neuroid', ['IT'] * self._num_neurons),
                                      'time_bin_start': ('time_bin', [self._time_bin_start]),
                                      'time_bin_end': ('time_bin', [self._time_bin_end])},
                              dims=['presentation', 'neuroid', 'time_bin'])
        recordings.name = 'random_it_model'
        return recordings
    
    def start_task(self, task, **kwargs):
        print("Starting task {task}")
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target=BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
        print("Recording from {recording_target} during {time_bins} ms")
        if str(recording_target) != "IT":
            raise NotImplementedError("RandomITModel only supports IT, not {recording_target}")
        if len(time_bins) != 1:
            raise NotImplementedError("RandomITModel only supports a single start-end time-bin, not {time_bins}")
        time_bins = time_bins[0].tolist()
        self._time_bin_start, self._time_bin_end = time_bins[0], time_bins[1]
    
    def visual_degrees(self):
        print("Declaring model to have a visual field size of 8 degrees")
        return 8

model = RandomITModel()
# The implementation maps a given brain region to a neural network layer.
# In the look_at method, the class just creates a mock result and returns it.
# The other two methods only check for correctness of the input values.

# The following lines load the public benchmark `MajajHong2015public.IT-pls`,
# consisting of neural recordings in macaque IT from `Majaj, Hong et al. 2015`
# and a neural predictivity metric based on PLS regression 
# to compare between model predictions and actual data. 
# Running the benchmark with the `RandomITModel` 
# then returns a score of the model's brain-likeness under this particular benchmark.

from brainscore import score_model
score = score_model(model_identifier='mymodel', model=model, benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)

# The benchmark
# (1) recorded from the model as its response to 2560 stimuli,
# (2) applied the neural predictivity metric to compare the predicted model recordings with the actual primate recordings to yield a score,
# (3) normalized the score by the ceiling.
# Since the benchmark already cross-validated results,
# the resulting score now contains the center (i.e. the average of the splits, in this case the mean)
# and the error (in this case standard-error-of-the-mean).

center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
print("score: {center.values:.3f}+-{error.values:.3f}")
# The score tells us that random features don't predict IT recordings well.
# We can also check the raw unceiled values...
unceiled_scores = score.raw
print(unceiled_scores)


# ...as well as the per-neuroid, per-split correlations.
raw_scores = score.raw.raw
print(raw_scores)


#%% Custom benchmarks
# We can also define our own benchmarks.
# The benchmark fulfills two purposes:
# 1. reproduce the primate experiment on the model
# 2. apply a similarity metric to compare predictions with actual measurements
# 3. normalize the match with the ceiling, i.e. an upper bound on how well a model could do
# The following example implements a simple benchmark that show-cases these three steps.
import numpy as np
import brainscore
from brainscore.benchmarks import Benchmark
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.benchmarks._neural_common import explained_variance, average_repetition

# Let's say, we want to test the model's match to IT recordings between 100-120 ms.
# We'll use the same Majaj et al. 2015 data from primates passively fixating.
class MyBenchmark(Benchmark):
    def __init__(self):
        # both StimulusSets as well as assemblies are packaged through https://github.com/brain-score/brainio_contrib
        assembly = brainscore.get_assembly('dicarlo.MajajHong2015.temporal.public')  # this will take a while to download and open
        assembly = assembly[{'time_bin': [start == 100 for start in assembly['time_bin_start'].values]}]
        # also, let's only look at a subset of the images
        image_ids = np.unique(assembly['image_id'].values)[:1000]
        assembly = assembly.loc[{'presentation': [image_id in image_ids for image_id in assembly['image_id'].values]}]
        stimulus_set = assembly.stimulus_set  # assemblies always have a StimulusSet attached to them
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(image_ids)]
        assembly.attrs['stimulus_set'] = stimulus_set
        # reduce to presentation x neuroid for simplicity (we only have one time_bin)
        assembly = assembly.squeeze('time_bin')
        self._assembly = assembly  # note that this assembly still has repetitions which we need for the ceiling
        self._similarity_metric = CrossRegressedCorrelation(
                                       regression=pls_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=3, stratification_coord='object_name'))
        self._ceiler = InternalConsistency()
    
    @property
    def identifier(self):  # for storing results
        return "my-dummy-benchmark"
    
    def __call__(self, candidate: BrainModel):
        # since the candidate follows the BrainModel interface, we can easily treat all models the same way.
        # (1) reproduce the experiment on the model. 
        candidate.start_task(task=BrainModel.Task.passive)
        candidate.start_recording(recording_target="IT", time_bins=[np.array((100, 120))])
        # since different models can have different fields of view, we adjust the image sizes accordingly.
        # for instance, a stimulus of 2 degree should take up little space for a model with a field of view of 10 degree
        # while the same stimulus would take up much more space for a model of 4 degrees.
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       # for reference, we know this experiment was run at 8 degrees for the primates.
                                       source_visual_degrees=8)
        predictions = candidate.look_at(stimuli=stimulus_set)
        # (2) compute similarity between predictions and measurements
        assembly = average_repetition(self._assembly)  # average over repetitions
        predictions = predictions.squeeze('time_bin')
        print("Computing model-match")
        unceiled_score = self._similarity_metric(predictions, assembly)
        # (3) normalize by our estimate of how well the ideal model could do
        ceiled_score = explained_variance(unceiled_score, self.ceiling)
        return ceiled_score
        
    
    @property
    def ceiling(self):
        print("Computing ceiling")
        return self._ceiler(self._assembly)

my_benchmark = MyBenchmark()
model = RandomITModel()  # we'll use the same model from before
score = my_benchmark(model)
print(score)

# We can also create a custom benchmark from scratch, using our own methods.
# To interface with the rest of Brain-Score, it is easiest if we just provide those to the Benchmark class.
# (But we could also not inherit and define the `__call__` method ourselves).
