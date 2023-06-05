#!/usr/bin/env python
# coding: utf-8

# This framework is concerned with comparing two sets of data, for instance source brain and target brain.
# It does not take care of trying multiple combinations of source data (such as multiple layers in models),
# but only makes direct comparisons.

#%% Metrics
# A metric tells us how similar to assemblies (sets of data) are to each other.
# For comparison, they might be re-mapped (neural predictivity) or compared in sub-spaces (RDMs).

#%% Pre-defined metrics
# Brain-Score comes with many standard metrics used in the field.
# One standardly used metric is neural predictivity:
# (1) it uses linear regression to linearly map between two systems (e.g. from model activations to neural firing rates),
# (2) it computes the correlation between predicted firing rates on held-out images,
# and (3) wraps all of that in cross-validation to estimate generalization.

#%% Neural Predictivity with Pearson Correlation
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation

regression = pls_regression()  # 1: define the regression
correlation = pearsonr_correlation()  # 2: define the correlation
metric = CrossRegressedCorrelation(regression, correlation)  # 3: wrap in cross-validation

# We can then run this metric on some datasets to obtain a score:
import numpy as np
from numpy.random import RandomState

from brainio_base.assemblies import NeuroidAssembly

rnd = RandomState(0)  # for reproducibility
assembly = NeuroidAssembly((np.arange(30 * 25) + rnd.standard_normal(30 * 25)).reshape((30, 25)),
                           coords={'image_id': ('presentation', np.arange(30)),
                                   'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                   'neuroid_id': ('neuroid', np.arange(25)),
                                   'region': ('neuroid', [0] * 25)},
                           dims=['presentation', 'neuroid'])
prediction, target = assembly, assembly  # we're testing how well the metric can predict the dataset itself
score = metric(source=prediction, target=target)
print(score)

# The score values above are aggregates over splits and neuroids.
# We can also check the raw values, i.e. the value per split and per neuroid.
print(score.raw)


#%% RDM
# Brain-Score also includes comparison methods not requiring any fitting, such as the Representational Dissimilarity Matrix (RDM).
from brainscore.metrics.rdm import RDMCrossValidated

metric = RDMCrossValidated()
rdm_score = metric(assembly1=assembly, assembly2=assembly)
print(rdm_score)

#%% Custom metrics
# A metric simply returns a Score for the similarity of two assemblies.
# For instance, the following computes the Euclidean distance of regressed and target neuroids.
from brainio_base.assemblies import DataAssembly
from brainscore.metrics.transformations import CrossValidation
from brainscore.metrics.xarray_utils import XarrayRegression
from brainscore.metrics.regression import LinearRegression

class DistanceMetric:
    def __init__(self):
        regression = LinearRegression()
        self._regression = XarrayRegression(regression=regression)
        self._cross_validation = CrossValidation()

    def __call__(self, source, target):
        return self._cross_validation(source, target, apply=self._apply, aggregate=self._aggregate)
        
    def _apply(self, source_train, target_train, source_test, target_test):
        self._regression.fit(source_train, target_train)
        prediction = self._regression.predict(source_test)
        score = self._compare(prediction, target_test)
        return score
    
    def _compare(self, prediction, target):
        prediction, target = prediction.sortby('image_id').sortby('neuroid_id'), target.sortby('image_id').sortby('neuroid_id')
        assert all(prediction['image_id'].values == target['image_id'].values)
        assert all(prediction['neuroid_id'].values == target['neuroid_id'].values)
        difference = np.abs(target.values - prediction.values)  # lower is better
        return DataAssembly(difference, coords=target.coords, dims=target.dims)
    
    def _aggregate(self, scores):
        return scores.median('neuroid').mean('presentation')
    

metric = DistanceMetric()
score = metric(assembly, assembly)
print(score)

