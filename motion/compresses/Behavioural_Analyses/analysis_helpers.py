#!/usr/bin/env python3

# Four rooms paper helper functions.
# This script contains some useful functions for navigation data analysis.
# Author: Hannah Sheahan
# Date: 30/11/2018
# Issues: N/A

#-----------------------------------------------------------------------------#
from typing import Any, List, Mapping, Sequence, Tuple

import constants as cs
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

import seaborn as sns
import copy
import pandas as pd
import sys
import random

# file i/o
import os

# for adding text description next to plots
from io import BytesIO
import base64

#-------------------------------------------------------------------------------

def find_data(data_path: str, sessions: List[str], exclusions: List[str]) -> Tuple[List[List[str]], int]:
  """Returns the Google Drive file paths of all data files in each session."""
  file_paths = []
  for session in sessions:
    session_path = os.path.join(data_path, session)
    files = os.listdir(session_path)

    # Exclude .DS_Store and subjects who's experiments crashed.
    included_files = []
    for idx, file in enumerate(files):
      include = True
      if file[0]!='.':  # Exclude .DS_Store.
        for exclude_string in exclusions:
          if exclude_string in file:
            include = False
        if include:
          included_files.append(file)
    paths = [os.path.join(session_path, file) for file in included_files]
    file_paths.append(paths)

  assert len(file_paths[0]) == len(file_paths[1])
  num_subjects = len(file_paths[0])
  print('Data files found from both sessions for {} participants.'.format(num_subjects))
  return file_paths, num_subjects

#-------------------------------------------------------------------------------

def load_data(paths: List[List[str]], num_subjects: int, sessions: List[str]) -> Tuple[List[List[Mapping[str, Any]]], Any]:
  """Loads the data from each session and matches it to the right participant.

  Args:
    paths: List of filepaths for each of the two sessions (in and out of scanner).
    num_subjects: The number of participants.
  Returns:
    All participant behavioural data, listed by session, then participant. The
    indices of each participant within each session is matched across the two
    sessions to refer to the same participant, and both are in chronological
    order.
  """
  # Use this to save the player/subject main learning data, with each element being a different subject
  all_data = [[None] * num_subjects for _ in sessions]
  file_names = [[None] * num_subjects for _ in sessions]

  for session_idx, session in enumerate(sessions):
    file_paths = paths[session_idx]
    # Sort the data in each session chronologically by participant index.
    file_order = np.argsort([a[-14:] for a in file_paths])
    file_paths = [file_paths[i] for i in file_order]
    for subject_idx in range(num_subjects):
      path = file_paths[subject_idx]
      with open(path, "r") as f:
        data = json.load(f)
      all_data[session_idx][subject_idx] = data
      file_names[session_idx][subject_idx] = path

  return all_data, file_names

#-------------------------------------------------------------------------------

def calculate_quiz_score(questionaire_data: List[Mapping[str, str]]) -> List[int]:
  """Calculate the quiz scores and order them by participant ID."""
  subject_quiz_scores = {}
  for subject_idx in range(len(questionaire_data)):
    quiz_score = 0
    sub_question_data = questionaire_data[subject_idx]
    if sub_question_data['Q1'] == 'top left':
      quiz_score += 1
    if sub_question_data['Q2'] == 'bottom right':
      quiz_score += 1
    if sub_question_data['Q3'] == 'bottom left':
      quiz_score += 1
    if sub_question_data['Q4'] == 'top right':
      quiz_score += 1
    if sub_question_data['Q5'] == 'bottom left, bottom right':
      quiz_score += 1
    if sub_question_data['Q6'] == 'top left, top right':
      quiz_score += 1
    if sub_question_data['Q7'] == 'top left, bottom left':
      quiz_score += 1
    if sub_question_data['Q8'] == 'top right, bottom right':
      quiz_score += 1
    subject_quiz_scores[sub_question_data['ID']] = quiz_score

  # Order the quiz scores by subject index.
  subject_ids = []
  for key, value in subject_quiz_scores.items():
    subject_ids.append(int(key))
  sorted_indices = sorted(range(len(subject_ids)), key=lambda k: subject_ids[k])
  subject_ordered_quiz_scores = []
  for idx in sorted_indices:
    subject_ordered_quiz_scores.append(subject_quiz_scores[str(subject_ids[idx])])

  return subject_ordered_quiz_scores

#-------------------------------------------------------------------------------

def plot_cov_training(ax, allPlayerLearningData, colours, bunch_factor=1, title=''):
    """Plot the covariance learning results for training session."""
    metric = 'correctChoices'
    whichSession = cs.SESSION_TRAINING
    nsubjects = len(allPlayerLearningData[whichSession])
    # We want the human choices only.
    allChoices = [allPlayerLearningData[whichSession][i]["roomChoicesData"][metric] for i in range(len(allPlayerLearningData[whichSession]))]
    allChoices = np.asarray(allChoices, dtype=np.float)
    allChoices = allChoices[~np.isnan(allChoices)]  # remove nans.

    # Average across adjacent trials using a bunch_factor.
    allChoices = np.reshape(allChoices, (bunch_factor, -1, nsubjects), order='F')
    allChoices = np.mean(allChoices, axis=0)

    # Mean and sem.
    meanChoices = 100 * np.nanmean(allChoices, axis=1)
    seChoices = 100 * np.divide(np.nanstd(allChoices, axis=1), np.sqrt(nsubjects) )
    ax.axhline(y=50, color='gray', linestyle=':')

    subtrials = np.linspace(1,bunch_factor*meanChoices.size, meanChoices.size)
    ax.scatter(subtrials, meanChoices, color=colours[0])
    ax.errorbar(subtrials, meanChoices, seChoices, color=colours[0])

    ax.set_ylabel('Correct first room choice (%)')
    ax.set_ylim([40,100])
    ax.set_xlabel('Trials')
    ax.set_title('Human behaviour, day 1')
    return allChoices

#-------------------------------------------------------------------------------

def plot_cov_scanner(ax, allPlayerLearningData, colours, bunch_factor=1, title=''):
    """Plot the covariance learning results for scanner session."""
    metric = 'humanCorrectChoices'
    whichSession = cs.SESSION_SCANNER
    nsubjects = len(allPlayerLearningData[whichSession])

    # We want the human choices only.
    allChoices = [allPlayerLearningData[whichSession][i]["roomChoicesData"][metric] for i in range(len(allPlayerLearningData[whichSession]))]
    allChoices = np.asarray(allChoices, dtype=np.float)
    allChoices = allChoices[~np.isnan(allChoices)]  # remove nans.

    # Average across adjacent trials using a bunch_factor.
    allChoices = np.reshape(allChoices, (bunch_factor, -1, nsubjects), order='F')
    allChoices = np.mean(allChoices, axis=0)

    # Draw context colour rectangles.
    errorboxes = [
      [Rectangle((1, 0), 32, 100)],
      [Rectangle((33, 0), 32, 100)],
      [Rectangle((33+32, 0), 32, 100)],
    ]
    edgecolor = 'white'
    facecolors = ['gold', 'gold', 'gold']
    alphas = [0.1, 0.25, 0.4]
    for box, facecolor, alpha in zip(errorboxes, facecolors, alphas):
      pc = PatchCollection(box, facecolor=facecolor, alpha=alpha,
                         edgecolor=facecolor)
      ax.add_collection(pc)

    # Mean and sem.
    meanChoices = 100 * np.nanmean(allChoices, axis=1)
    seChoices = 100 * np.divide(np.nanstd(allChoices, axis=1), np.sqrt(nsubjects) )
    ax.axhline(y=50, color='gray', linestyle=':')
    # Note that here we artificially double the trial indices to account for
    # these decisions being from humans only, as in half of trials this choice
    # is made by the computer agent.
    subtrials = np.linspace(bunch_factor/2, 2* bunch_factor*meanChoices.size, meanChoices.size)

    # Plot the learning.
    ax.scatter(subtrials, meanChoices, color=colours[1])
    ax.errorbar(subtrials, meanChoices, seChoices, color=colours[1])

    ax.set_ylim([40,100])
    ax.set_xlabel('Trials')
    ax.set_title('Human behaviour, day 2')

    return allChoices

#-------------------------------------------------------------------------------

def insert_rests(trial_maps: List[str], which_session):
  """Inserts the rests back into the sequence of trial maps."""
  if which_session == cs.SESSION_TRAINING:
    rest_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 25, 42, 59, 76]
  else:
    rest_idx = [0, 1, 2, 3, 4, 5, 6, 23, 40, 57, 74, 91, 108]
  corrected_maps = trial_maps[:]
  for trial in rest_idx:
    corrected_maps.insert(trial, 'rest         ')
  return corrected_maps

#-------------------------------------------------------------------------------

def plot_scores_training_context(ax, allPlayerLearningData, contexts, colour, bunch_factor=1, title='', ylim=(15,30)):
    """Plot the score learning results for training session, by context."""
    metric = 'allScores'
    whichSession = cs.SESSION_TRAINING
    nsubjects = len(allPlayerLearningData[whichSession])
    # if you want the human choices only, only average across subs over those
    allChoices = [allPlayerLearningData[whichSession][i][metric] for i in range(len(allPlayerLearningData[whichSession]))]

    contextChoices = []
    for subject_idx, subjectChoices in enumerate(allChoices):
      contextMaps = allPlayerLearningData[whichSession][subject_idx]["contextMapList"]
      #print([trial_idx for trial_idx, context in enumerate(contextMaps) if context[10:] in contexts])
      contextMaps = insert_rests(contextMaps, cs.SESSION_TRAINING)
      contextTrials = [subjectChoices[trial_idx] for trial_idx, context in enumerate(contextMaps) if context[10:] in contexts]
      contextTrials = np.asarray(contextTrials, dtype=np.float)
      contextTrials = contextTrials[~np.isnan(contextTrials)]  # remove nans.
      contextChoices.append(contextTrials)

    contextChoices = np.asarray(contextChoices, dtype=np.float)
    contextChoices = np.transpose(contextChoices)

    # Average across adjacent trials using a bunch_factor
    contextChoices = np.reshape(contextChoices, (bunch_factor, -1, nsubjects), order='F')
    contextChoices = np.mean(contextChoices, axis=0)

    # mean and sem
    meanChoices = np.nanmean(contextChoices, axis=1)
    seChoices = np.divide(np.nanstd(contextChoices, axis=1), np.sqrt(nsubjects) )
    if 'peanut' in contexts: # vertical. Was tested first.
      subtrials = [list(range(16)), list(range(32, 48))]
      subtrials = [element for sublist in subtrials for element in sublist]
    else:
      subtrials = [list(range(16, 32)), list(range(48, 64))]
      subtrials = [element for sublist in subtrials for element in sublist]

    # First block in this context.
    trials = np.reshape(subtrials[:len(subtrials)//2], (bunch_factor, -1), order='F')
    trials = np.mean(trials, axis=0)
    ax.scatter(trials, meanChoices[:len(meanChoices)//2], color=colour)
    ax.errorbar(trials, meanChoices[:len(meanChoices)//2], seChoices[:len(seChoices)//2], color=colour)

    # Second block in this context.
    trials = np.reshape(subtrials[len(subtrials)//2:], (bunch_factor, -1), order='F')
    trials = np.mean(trials, axis=0)
    ax.scatter(trials, meanChoices[len(meanChoices)//2:], color=colour)
    handle = ax.errorbar(trials, meanChoices[len(meanChoices)//2:], seChoices[len(seChoices)//2:], color=colour)

    ax.set_ylabel('Trial score')
    ax.set_xlabel('Trials')
    ax.set_title('Human behaviour, day 1')
    ax.set_ylim(ylim)
    return allChoices, handle

#-------------------------------------------------------------------------------

def plot_scores_scanner_context(ax, allPlayerLearningData, contexts, colour, bunch_factor=1, title='', ylim=(15,30)):
    """Plot the score learning results for scanner session, by context.
    """
    metric = 'allScores'
    whichSession = cs.SESSION_SCANNER
    nsubjects = len(allPlayerLearningData[whichSession])
    # if you want the human choices only, only average across subs over those
    allChoices = [allPlayerLearningData[whichSession][i][metric] for i in range(len(allPlayerLearningData[whichSession]))]

    contextChoices = []
    for subject_idx, subjectChoices in enumerate(allChoices):
      contextMaps = allPlayerLearningData[whichSession][subject_idx]["contextMapList"]
      contextMaps = insert_rests(contextMaps, cs.SESSION_SCANNER)
      contextTrials = [subjectChoices[trial_idx] for trial_idx, context in enumerate(contextMaps) if context[10:] in contexts]
      contextTrials = [i for i in contextTrials if ~np.isnan(np.asarray(i, dtype=np.float))]  # remove nans.
      while len(contextTrials) != 48:
        contextTrials.append(np.nan)
      contextTrials = np.asarray(contextTrials, dtype=np.float)
      contextChoices.append(contextTrials)

    contextChoices = np.asarray(contextChoices, dtype=np.float)
    contextChoices = np.transpose(contextChoices)

    # Average across adjacent trials using a bunch_factor
    contextChoices = np.reshape(contextChoices, (bunch_factor, -1, nsubjects), order='F')
    contextChoices = np.nanmean(contextChoices, axis=0)

    # Draw context colour rectangles
    errorboxes = [
      [Rectangle((1, 0), 32, 100)],
      [Rectangle((33, 0), 32, 100)],
      [Rectangle((33+32, 0), 32, 100)],
    ]
    edgecolor = 'white'
    facecolors = ['gold', 'gold', 'gold']
    alphas = [0.1, 0.25, 0.4]
    for box, facecolor, alpha in zip(errorboxes, facecolors, alphas):
      pc = PatchCollection(box, facecolor=facecolor, alpha=alpha,
                         edgecolor=facecolor)
      ax.add_collection(pc)

    # mean and sem
    meanChoices = np.nanmean(contextChoices, axis=1)
    seChoices = np.divide(np.nanstd(contextChoices, axis=1), np.sqrt(nsubjects) )
    # Manually increase span over which trials are plotted to reflect interleaved contexts.
    subtrials = np.linspace(bunch_factor/2, 2* bunch_factor*meanChoices.size, meanChoices.size)

    # Plot the learning.
    ax.scatter(subtrials, meanChoices, color=colour)
    handle = ax.errorbar(subtrials, meanChoices, seChoices, color=colour)

    ax.set_xlabel('Trials')
    ax.set_title('Human behaviour, day 2')
    ax.set_ylim(ylim)

    return allChoices, handle

#-------------------------------------------------------------------------------

def plot_cov_training_context(ax, allPlayerLearningData, contexts, colour, bunch_factor=1, title=''):
    """Plot the covariance learning results for training session, by context."""
    metric = 'correctChoices'
    whichSession = cs.SESSION_TRAINING
    nsubjects = len(allPlayerLearningData[whichSession])
    # if you want the human choices only, only average across subs over those
    allChoices = [allPlayerLearningData[whichSession][i]["roomChoicesData"][metric] for i in range(len(allPlayerLearningData[whichSession]))]

    contextChoices = []
    for subject_idx, subjectChoices in enumerate(allChoices):
      contextMaps = allPlayerLearningData[whichSession][subject_idx]["contextMapList"]
      contextMaps = insert_rests(contextMaps, cs.SESSION_TRAINING)
      contextTrials = [subjectChoices[trial_idx] for trial_idx, context in enumerate(contextMaps) if context[10:] in contexts]
      contextTrials = np.asarray(contextTrials, dtype=np.float)
      contextTrials = contextTrials[~np.isnan(contextTrials)]  # remove nans.
      contextChoices.append(contextTrials)

    contextChoices = np.asarray(contextChoices, dtype=np.float)
    contextChoices = np.transpose(contextChoices)

    # Average across adjacent trials using a bunch_factor
    contextChoices = np.reshape(contextChoices, (bunch_factor, -1, nsubjects), order='F')
    contextChoices = np.mean(contextChoices, axis=0)

    # mean and sem
    meanChoices = 100 * np.nanmean(contextChoices, axis=1)
    seChoices = 100 * np.divide(np.nanstd(contextChoices, axis=1), np.sqrt(nsubjects) )
    ax.axhline(y=50, color='gray', linestyle=':')
    if 'peanut' in contexts: # vertical. Was tested first.
      subtrials = [list(range(16)), list(range(32, 48))]
      subtrials = [element for sublist in subtrials for element in sublist]
    else:
      subtrials = [list(range(16, 32)), list(range(48, 64))]
      subtrials = [element for sublist in subtrials for element in sublist]

    # First block in this context.
    trials = np.reshape(subtrials[:len(subtrials)//2], (bunch_factor, -1), order='F')
    trials = np.mean(trials, axis=0)
    ax.scatter(trials, meanChoices[:len(meanChoices)//2], color=colour)
    ax.errorbar(trials, meanChoices[:len(meanChoices)//2], seChoices[:len(seChoices)//2], color=colour)

    # Second block in this context.
    trials = np.reshape(subtrials[len(subtrials)//2:], (bunch_factor, -1), order='F')
    trials = np.mean(trials, axis=0)
    ax.scatter(trials, meanChoices[len(meanChoices)//2:], color=colour)
    handle = ax.errorbar(trials, meanChoices[len(meanChoices)//2:], seChoices[len(seChoices)//2:], color=colour)

    ax.set_ylabel('Correct first room choice (%)')
    ax.set_ylim([40,100])
    ax.set_xlabel('Trials')
    ax.set_title('Human behaviour, day 1')
    return contextChoices, handle

#-------------------------------------------------------------------------------

def plot_cov_scanner_context(ax, allPlayerLearningData, contexts, colour, bunch_factor=1, title=''):
    """Plot the covariance learning results for scanner session, by context."""
    metric = 'humanCorrectChoices'
    whichSession = cs.SESSION_SCANNER
    nsubjects = len(allPlayerLearningData[whichSession])
    # if you want the human choices only, only average across subs over those
    allChoices = [allPlayerLearningData[whichSession][i]["roomChoicesData"][metric] for i in range(len(allPlayerLearningData[whichSession]))]

    contextChoices = []
    for subject_idx, subjectChoices in enumerate(allChoices):
      contextMaps = allPlayerLearningData[whichSession][subject_idx]["contextMapList"]
      contextMaps = insert_rests(contextMaps, cs.SESSION_SCANNER)
      contextTrials = [subjectChoices[trial_idx] for trial_idx, context in enumerate(contextMaps) if context[10:] in contexts]
      contextTrials = [i for i in contextTrials if ~np.isnan(np.asarray(i, dtype=np.float))]  # remove nans.
      while len(contextTrials) != 24:
        contextTrials.append(np.nan)
      contextTrials = np.asarray(contextTrials, dtype=np.float)
      contextChoices.append(contextTrials)

    contextChoices = np.asarray(contextChoices, dtype=np.float)
    contextChoices = np.transpose(contextChoices)

    # Average across adjacent trials using a bunch_factor
    contextChoices = np.reshape(contextChoices, (bunch_factor, -1, nsubjects), order='F')
    contextChoices = np.nanmean(contextChoices, axis=0)

    # Draw context colour rectangles
    errorboxes = [
      [Rectangle((1, 0), 32, 100)],
      [Rectangle((33, 0), 32, 100)],
      [Rectangle((33+32, 0), 32, 100)],
    ]
    edgecolor = 'white'
    facecolors = ['gold', 'gold', 'gold']
    alphas = [0.1, 0.25, 0.4]
    for box, facecolor, alpha in zip(errorboxes, facecolors, alphas):
      pc = PatchCollection(box, facecolor=facecolor, alpha=alpha,
                         edgecolor=facecolor)
      ax.add_collection(pc)

    # mean and sem
    meanChoices = 100 * np.nanmean(contextChoices, axis=1)
    seChoices = 100 * np.divide(np.nanstd(contextChoices, axis=1), np.sqrt(nsubjects) )
    ax.axhline(y=50, color='gray', linestyle=':')
    # Manually increase span over which trials are plotted to reflect interleaved contexts.
    subtrials = np.linspace(bunch_factor/2, 2*2* bunch_factor*meanChoices.size, meanChoices.size)

    # Plot the learning.
    ax.scatter(subtrials, meanChoices, color=colour)
    handle = ax.errorbar(subtrials, meanChoices, seChoices, color=colour)

    ax.set_ylim([40,100])
    ax.set_xlabel('Trials')
    ax.set_title('Human behaviour, day 2')

    return contextChoices, handle

#-------------------------------------------------------------------------------

def moved_which_way(current_position: List[float], previous_position: List[float]) -> str:
  """Determine the direction of travel."""
  if current_position[0] > previous_position[0]: # moved to right.
    return 'right'
  elif current_position[0] < previous_position[0]: # moved left.
    return 'left'
  elif current_position[1] > previous_position[1]: # moved up.
    return 'up'
  elif current_position[1] < previous_position[1]: # moved down.
    return 'down'
  else:
    return 'no_move'

#-------------------------------------------------------------------------------

def format_room_choices(ax, allPlayerLearningData, title=''):
    """Format the covariance learning results for scanner session."""
    metric = 'humanCorrectChoices'
    whichSession = cs.SESSION_SCANNER
    nsubjects = len(allPlayerLearningData[whichSession])
    # if you want the human choices only, only average across subs over those
    allChoices = [allPlayerLearningData[whichSession][i]["roomChoicesData"][metric] for i in range(len(allPlayerLearningData[whichSession]))]
    allChoices = np.asarray(allChoices, dtype=np.float)
    allChoices = allChoices[~np.isnan(allChoices)]  # remove nans.
    allChoices = np.reshape(allChoices, (-1, nsubjects), order='F')
    return allChoices

#-------------------------------------------------------------------------------

def reformatControlStrings(controlStateTransitions):
    for trial in range(len(controlStateTransitions)):
        times = []
        controls = []
        if trial is not None:
            for entry in range(len(controlStateTransitions[trial])):
                time, control = controlStateTransitions[trial][entry].split(" ")
                time = float(time.replace(",","."))
                times.append(time)
                controls.append(control)
            controlStateTransitions[trial] = {"timestamps":times, "controlmode":controls}
    return controlStateTransitions

#-------------------------------------------------------------------------------

def defineFilepaths():
    # Specify paths for data and generated figures
    # Note that the filepath has been updated to the processed data one because otherwise we don't include the 2 manually combined subject behavioural logs
    savepath = '/Users/hsheahan/Documents/Four_rooms/2d_behavioural_data/generated_figures/'
    #filepath = '/Users/hannahsheahan/Documents/Postdoc/Experiments/MartiniTask/2D_version/Data/scanner/behaviour/originalData/'
    filepath = '/Users/hsheahan/Documents/Four_rooms/2d_behavioural_data/hannah_preprocessed/'
    filepaths = [filepath + 'session_1/', filepath + 'session_2/']

    session1 = [f for f in os.listdir(filepaths[0]) if os.path.isfile(os.path.join(filepaths[0], f))]
    session2 = [f for f in os.listdir(filepaths[1]) if os.path.isfile(os.path.join(filepaths[1], f))]
    session1 = [session1[file] for file in range(len(session1)) if ("_ignore"  not in session1[file] and "DS_"  not in session1[file])]
    session2 = [session2[file] for file in range(len(session2)) if ("_ignore"  not in session2[file] and "DS_"  not in session2[file])]
    sessions = [session1, session2]
    nSubjects = len(session1) if len(session1) > len(session2) else len(session2)

    return sessions, nSubjects, filepaths, savepath

#-------------------------------------------------------------------------------

def find_train_test_deltatime(sessions):
  """Find time between the training and scanning sessions for each participant."""
  dt = [[None for i in range(len(sessions[cs.SESSION_TRAINING]))] for j in range(2)]
  for i in range(len(sessions[cs.SESSION_TRAINING])):

    # Find the date and time for the training session.
    file = sessions[cs.SESSION_TRAINING][i]
    subject = file[-14:-11]
    trainingdate = file[9:23]
    traindatetime = datetime.strptime(trainingdate, '%d-%m-%y_%H-%M')
    dt[cs.SESSION_TRAINING][i] = traindatetime

    # Find the associated date and time for the scanning session for that subjects.
    for j in range(len(sessions[cs.SESSION_SCANNER])):
      scannerfile = sessions[cs.SESSION_SCANNER][j]
      scannersubject = scannerfile[-14:-11]
      if scannersubject == subject:
        testingdate = scannerfile[9:23]
        testdatetime = datetime.strptime(testingdate, '%d-%m-%y_%H-%M')
        dt[cs.SESSION_SCANNER][i] = testdatetime

  timediff = []
  for i in range(len(sessions[cs.SESSION_TRAINING])):
    diff = dt[cs.SESSION_SCANNER][i] - dt[cs.SESSION_TRAINING][i]
    hoursdiff = (diff.days * 24) + (diff.seconds / (60 * 60))
    timediff.append(hoursdiff)
  timediff = [diff for diff in timediff if diff > 0.01] # Remove the spurious 'zero' diffs.
  mean = np.mean(timediff)
  err = np.std(timediff) / np.sqrt(len(timediff))
  print('On average train and test (scanner) sessions were: {:.1f} +- {:.1f} hours apart (mean +- SEM).'.format(mean, err))

  return timediff

#-------------------------------------------------------------------------------

def compute_duration_per_agent(raw_data, num_subjects: int, allowed_states: Tuple[int, ...]) -> Tuple[List[float], List[float]]:
  """Calculate how much time is spent under human vs agent control per trial."""
  durations = []
  players_human_time = []
  players_computer_time = []
  for player_id in range(num_subjects):
    num_trials = len(raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'])
    human_durations = []
    computer_durations = []
    for trial in range(num_trials):
      human_durations_per_trial = []
      computer_durations_per_trial = []
      previous_game_state = cs.STATE_GO
      control_idx = 0
      control_order = raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['controlStateOrder']
      if control_order:
        for i in range(2):  # Ensure there are enough switches in control order to handle the data.
          control_order.append(control_order[0])
          control_order.append(control_order[1])
      times = raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['timeStepTrackingData']
      first_timestep, time = 0., 0. # Default
      first_step = True
      for step_ind, step in enumerate(times):
        game_state = raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['stateTransitions'][step_ind]
        if step[0] != 'T':
          if (int(game_state) in allowed_states):
            time = step[:6]
            time = time.replace(',', '.')
            time = float(time)
            if first_step:
              first_timestep = time
              first_step = False
          duration = time - first_timestep
          if int(game_state) == cs.STATE_SHOWREWARD and int(previous_game_state) != cs.STATE_SHOWREWARD:
            # A control transition has occurred.
            if control_idx < len(control_order):
              if control_order[control_idx] == 'Human':
                human_durations_per_trial.append(duration)
              else:
                computer_durations_per_trial.append(duration)
              control_idx += 1
              first_timestep = copy.copy(time)
        previous_game_state = copy.copy(game_state)
      human_durations.append(np.sum(human_durations_per_trial))
      computer_durations.append(np.sum(computer_durations_per_trial))
    players_human_time.append(np.nanmean([i for i in human_durations if i]))
    players_computer_time.append(np.nanmean([i for i in computer_durations if i]))

  return players_human_time, players_computer_time

#-------------------------------------------------------------------------------

def player_in_which_square(xpos: float, zpos: float) -> int:
    """Takes a momentary player x,z position and outputs a gridsquare index.

    The player can only be in one square at a time.
    """
    squarelabel = 1
    squarepositions = [None]*(9*9)
    count = 0
    deltaSquare = 1

    for i in np.linspace(-4.5,3.5,9): # Note that these are the 'lower-left' corner positions for each of the squares
        for j in np.linspace(-4.5,3.5,9):
            # this is a list carrying coordinates of each square in the environment. Each index is the label of the square.
            squarepositions[count] = [i,j]
            envx = squarepositions[count][0]
            envz = squarepositions[count][1]

            if ( xpos >= envx ) and (xpos < (envx + deltaSquare)):
                if (zpos >= envz ) and (zpos < (envz + deltaSquare)):
                    squarelabel = count

            count += 1   # ready for evaluating the next square

    return squarelabel

#-------------------------------------------------------------------------------

def compute_occupancy_and_transitions(
    raw_data,
    num_subjects: int,
    allowed_states: Tuple[int, ...],
    *,
    vertical_maps: Tuple[str, ...],
    horizontal_maps: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, List[int]], Mapping[str, List[int]]]:
  """Calculates the duration and transitions per gridsquare in each context."""
  gridsize = 9
  dataFrequency = 0.04
  horizontal_transitions = {
      'right':[0]*(gridsize*gridsize),
      'left':[0]*(gridsize*gridsize),
      'up':[0]*(gridsize*gridsize),
      'down':[0]*(gridsize*gridsize),
      'no_move':[0]*(gridsize*gridsize),
  }
  vertical_transitions = {
      'right':[0]*(gridsize*gridsize),
      'left':[0]*(gridsize*gridsize),
      'up':[0]*(gridsize*gridsize),
      'down':[0]*(gridsize*gridsize),
      'no_move':[0]*(gridsize*gridsize),
  }
  vertical_heatmap_counts = [0]*(gridsize*gridsize)
  horizontal_heatmap_counts = [0]*(gridsize*gridsize)

  timestep_positions = []
  for player_id in range(num_subjects):
    player_positions = []
    num_trials = len(raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'])
    for trial in range(num_trials):
      trial_positions = []
      positions = raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['timeStepTrackingData']
      for step_ind, step in enumerate(positions):
        game_state = raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['stateTransitions'][step_ind]
        if step[0] != 'T':
          if (int(game_state) in allowed_states):
            position = step.replace(',', '.')
            position = position.split(" ")
            position = [float(i) for i in position]
            position = position[1:] # ignore the time info and just retain position.
            square_label = player_in_which_square(position[0], position[1])
            if trial_positions:
              transition = moved_which_way(position, trial_positions[-1])
            else:
              prev_square_label = copy.copy(square_label)
              transition = 'no_move'
            trial_positions.append(position)

            if raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['mapName'] in vertical_maps:
              vertical_heatmap_counts[square_label] +=1
              vertical_transitions[transition][prev_square_label] += 1
            elif raw_data[cs.SESSION_SCANNER][player_id]['allTrialData'][trial]['mapName'] in horizontal_maps:
              horizontal_heatmap_counts[square_label] +=1
              horizontal_transitions[transition][prev_square_label] += 1
            prev_square_label = copy.copy(square_label)

  # Convert to units of time (s).
  horizontal_heatmap_counts = [float(horizontal_heatmap_counts[i])*dataFrequency for i in range(len(horizontal_heatmap_counts))]
  horizontal_counts = np.array(horizontal_heatmap_counts).reshape(gridsize,gridsize)
  vertical_heatmap_counts = [float(vertical_heatmap_counts[i])*dataFrequency for i in range(len(vertical_heatmap_counts))]
  vertical_counts = np.array(vertical_heatmap_counts).reshape(gridsize,gridsize)

  return horizontal_counts, vertical_counts, horizontal_transitions, vertical_transitions

#-------------------------------------------------------------------------------

def normalise_transitions(horizontal_transitions: Mapping[str, List[int]], vertical_transitions: Mapping[str, List[int]]) -> Mapping[str, Mapping[str, List[float]]]:
  """Calculate transition probabilities for each grid-square and context."""
  gridsize = 9
  transition_probs = {'horizontal':[], 'vertical':[]}
  for context in ['horizontal', 'vertical']:
    normalised_transitions = {
      'right':[0]*(gridsize*gridsize),
      'left':[0]*(gridsize*gridsize),
      'up':[0]*(gridsize*gridsize),
      'down':[0]*(gridsize*gridsize),
      'no_move':[0]*(gridsize*gridsize),
    }
    if context == 'horizontal':
      transitions = copy.copy(horizontal_transitions)
    elif context == 'vertical':
      transitions = copy.copy(vertical_transitions)
    else:
      raise ValueError('Unrecognised context.')

    for square_label in range(gridsize*gridsize):
      total_transition_from_square = 0
      for key in ['right', 'left', 'down', 'up']:
        total_transition_from_square += transitions[key][square_label]

      for key in ['right', 'left', 'down', 'up']:
        if total_transition_from_square == 0:
          normalised_transitions[key][square_label] = 0
        else:
          normalised_transitions[key][square_label] = float(transitions[key][square_label]) / float(total_transition_from_square)
    transition_probs[context] = normalised_transitions

  return transition_probs

#-------------------------------------------------------------------------------

def xy_direction(label):
  """Map the string label to a direction for plotting."""
  mapping = {'right':(1,0), 'left':(-1,0), 'up':(0,-1), 'down':(0,1), 'none':(0,0)}
  return mapping[label]

#-------------------------------------------------------------------------------

def vectorise_transitions(transition_probs: Mapping[str, Mapping[str, List[float]]]) -> Mapping[str, List[float]]:
  """Compute the vector of the summed transitions per gridsquare."""
  gridsize = 9
  direction_label = {0:'right', 1:'left', 2:'up', 3:'down', 4:'none'}
  transition_vectors = {'horizontal':[], 'vertical':[]}

  for context in ['horizontal', 'vertical']:
    direction_vector = [0] * (gridsize*gridsize)
    for square_label in range(gridsize*gridsize):
      normed_transitions = transition_probs[context]
      all_directions = np.array([normed_transitions[key][square_label] for key in ['right', 'left', 'up', 'down']])
      if np.all(np.array([i==all_directions[0] for i in all_directions])):
        direction = 4
      else:
        direction = np.argmax(all_directions)
      vector = all_directions[0]*np.array([1., 0.]) + all_directions[1]*np.array([-1., 0.]) + all_directions[2]*np.array([0.,-1.]) + all_directions[3]*np.array([0.,1.])
      direction_vector[square_label] = vector
    transition_vectors[context] = direction_vector
  return transition_vectors

#-------------------------------------------------------------------------------

def plot_transition_heatmap(horiz_counts, vert_counts, transition_vectors: Mapping[str, Any]) -> None:
  """Plot the occupancy heatmap and vector transitions."""
  plt.figure(figsize=(10,4))
  plt.subplot(1,2,1)
  axes = sns.heatmap(np.rot90(horiz_counts/(27*48), k=1), linewidths=.5, square=True, yticklabels=False,
                    xticklabels=False, cmap=cm.gist_heat_r, cbar_kws={"label":"Time spent per trial (sec)"})
  plt.title('Horizontal contexts')

  count = 0
  x,y = np.meshgrid(np.arange(0.5,9.5),np.arange(0.5,9.5))
  test = np.asarray([transition_vectors['horizontal']]).reshape((9,9,2), order='F')
  u, v = test[:,:,0], test[:,:,1]
  u[u==0] = np.nan
  axes.quiver(x,y,u,v, scale=7.)

  plt.subplot(1,2,2)
  axes = sns.heatmap(np.rot90(vert_counts/(27*48), k=1), linewidths=.5, square=True, yticklabels=False, xticklabels=False, cmap=cm.gist_heat_r,
                    cbar_kws={"label":"Time spent per trial (sec)"})
  plt.title('Vertical contexts')

  x,y = np.meshgrid(np.arange(0.5,9.5),np.arange(0.5,9.5))
  test = np.asarray(transition_vectors['vertical']).reshape((9,9,2), order='F')
  u, v = test[:,:,0], test[:,:,1]
  u[u==0] = np.nan
  axes.quiver(x,y,u,v, scale=7.)
  axes.set_aspect('equal')

#-------------------------------------------------------------------------------

def visualise_training_metadata(allPlayerLearningData, sessions, savepath, plotcolours, saveFig):
  """Plot across-subject metadata for the training session."""

  # Histogram of age.
  plt.figure(figsize=(12,3))
  subjectAges = [float(allPlayerLearningData[cs.SESSION_SCANNER][i]["participantAge"]) for i in range(len(allPlayerLearningData[cs.SESSION_SCANNER])) if allPlayerLearningData[cs.SESSION_SCANNER][i] is not None]
  plt.subplot(1,4,1)
  plt.hist(subjectAges, color=plotcolours[cs.SESSION_TRAINING])
  plt.xlabel('Age')
  plt.ylabel('Number of subjects')
  print('Mean age: {}, +- {} std'.format(np.mean(subjectAges), np.std(subjectAges)))

  # Total practice scores.
  trainingScore = [(allPlayerLearningData[cs.SESSION_TRAINING][i]["totalScore"]) for i in range(len(allPlayerLearningData[cs.SESSION_TRAINING])) if allPlayerLearningData[cs.SESSION_TRAINING][i] is not None]
  plt.subplot(1,4,2)
  plt.hist(trainingScore, color=plotcolours[cs.SESSION_TRAINING])
  plt.xlabel('Training score')

  # Determine how much time was between the two experiment sessions for pairs of datafiles by the same subject
  timeBetweenSessions = find_train_test_deltatime(sessions)
  plt.subplot(1,4,3)
  plt.hist(timeBetweenSessions, bins=15, color=plotcolours[cs.SESSION_TRAINING])
  plt.xlabel('Time between training and test sessions')

  # Practice duration.
  practiceDuration = [(allPlayerLearningData[cs.SESSION_TRAINING][i]["experimentDuration"]) for i in range(len(allPlayerLearningData[cs.SESSION_TRAINING])) if allPlayerLearningData[cs.SESSION_TRAINING][i] is not None]
  plt.subplot(1,4,4)
  plt.hist(practiceDuration, color=plotcolours[cs.SESSION_TRAINING])
  plt.xlabel('Practice duration (s)')

  plt.suptitle('Histograms of Training Session Metadata')
  if saveFig:
    plt.savefig('Training_metadata.pdf', bbox_inches='tight')
    files.download("Training_metadata.pdf")

#-------------------------------------------------------------------------------

def visualise_testing_metadata(allPlayerLearningData, sessions, savepath, plotcolours, saveFig):
  """Plot the across-subject metadata for the scanning session."""
  # how many subjects on each ordering of the stimuli
  experimentVersions = [(allPlayerLearningData[cs.SESSION_SCANNER][i]["experimentVersion"][16:])
   for i in range(len(allPlayerLearningData[cs.SESSION_SCANNER])) if allPlayerLearningData[cs.SESSION_SCANNER][i] is not None]
  for i in range(len(experimentVersions)):
    version = experimentVersions[i]
    version = version.replace("cheese", "c")
    version = version.replace("mushroom", "m")
    version = version.replace("pineapple","p")
    experimentVersions[i] = version

  plt.subplots_adjust(wspace=0.5)
  fig, axes = plt.subplots(1,3, figsize=(14,4))
  axes[0].hist(experimentVersions, color=plotcolours[cs.SESSION_SCANNER])
  axes[0].set_xlabel('Block test ordering')
  axes[0].set_ylabel('Number of subjects')
  axes[0].set_title('Different reward type orderings')

  # Plot an example trial sequence for a single sub to see the switching
  numericalMapSequence = []
  randsub = random.randint(0,len(allPlayerLearningData[cs.SESSION_SCANNER])-1)
  mapSequence = allPlayerLearningData[cs.SESSION_SCANNER][randsub]["mapName"][7:]
  for map in mapSequence:
    if "banana" in map:
      numericalMapSequence.append(1)
    elif "mushroom" in map:
      numericalMapSequence.append(2)
    elif "pineapple" in map:
      numericalMapSequence.append(3)
    elif "avocado" in map:
      numericalMapSequence.append(4)
    elif "watermelon" in map:
      numericalMapSequence.append(5)
    elif "cheese" in map:
      numericalMapSequence.append(6)
    else:
      numericalMapSequence.append(None)

  axes[1].plot(numericalMapSequence, color=plotcolours[cs.SESSION_SCANNER])
  for i in range(len(numericalMapSequence)):
    if numericalMapSequence[i] is None:
        axes[1].axvline(x=i, color='grey', linestyle=':')
  axes[1].set_yticks([i for i in range(1,7)])
  axes[1].set_yticklabels(['banana','mushroom','pineapple','avocado','watermelon','cheese'], rotation=60)
  axes[1].set_xlabel('Trial')
  axes[1].set_title("Context sequence for example subject #{0}".format(randsub))

  # dist of test duration
  testDuration = [(allPlayerLearningData[cs.SESSION_SCANNER][i]["experimentDuration"]) for i in range(len(allPlayerLearningData[cs.SESSION_SCANNER])) if allPlayerLearningData[cs.SESSION_SCANNER][i] is not None]
  axes[2].hist(testDuration, color=plotcolours[cs.SESSION_SCANNER])
  axes[2].set_xlabel('Scanner test duration (s)')
  axes[2].set_ylabel('Number of subjects')

  fig.suptitle('Scanning session metadata')
  if saveFig:
    plt.savefig('Testing_metadata.pdf', bbox_inches='tight')
    files.download("Testing_metadata.pdf")

#-------------------------------------------------------------------------------

def plot_shaded_curve(ax, trials, means, sems, colourname='dodgerblue', title='', linelabel=''):
    """Plot a pretty shaded learning curve."""
    # Offset the x-values by the number of trials at the start that are just for setup
    offset = 0
    for i in range(len(means)):
        if not math.isnan(means[i]):
            break
        offset += 1
    means = means[offset:]
    sems = sems[offset:]
    trials = trials[offset:]-trials[offset]

    # plot the rest breaks
    for i in range(len(means)):
        if math.isnan(means[i]):
            ax.axvline(x=trials[i], color='lightgray', linestyle=':')

    # plot the actual data
    ax.plot(trials, means, color=colourname, label=linelabel)
    ax.fill_between(trials, means-sems, means+sems, color=colourname, alpha=0.25)
    ax.set_xlim([0, np.max(trials)])
    ax.set_xlabel('Trial')
    ax.set_title(title)

#-------------------------------------------------------------------------------

def plot_scores_learning(
    ax,
    allPlayerLearningData,
    whichSession,
    savepath,
    colours,
    bunch_factor=1,
    plot_individuals=False,
    saveFig=False,
    title='',
    ylim=(0,32),
    ylabel='Trial score',
):
  """Plots trial score across the session."""
  nsubjects = len(allPlayerLearningData[whichSession])
  if plot_individuals:
    for i in range(nsubjects):
      scores = allPlayerLearningData[whichSession][i]["allScores"]
      ax.plot(scores, alpha=0.1)

  subScores = [allPlayerLearningData[whichSession][i]["allScores"]
    for i in range(len(allPlayerLearningData[whichSession]))]
  subScores = np.asarray(subScores, dtype=np.float)
  subScores = subScores[~np.isnan(subScores)]  # remove nans.

  # Average across adjacent trials using a bunch_factor
  subScores = np.reshape(subScores, (bunch_factor, -1, nsubjects), order='F')
  subScores = np.mean(subScores, axis=0)
  # Mean and sem
  meanScores =  np.nanmean(subScores, axis=1)
  seScores = np.nanstd(subScores, axis=1) / np.sqrt(nsubjects)
  subtrials = np.linspace(1,bunch_factor*meanScores.size, meanScores.size)

  # Plot the learning.
  ax.scatter(subtrials, meanScores, color=colours[whichSession])
  ax.errorbar(subtrials, meanScores, seScores, color=colours[whichSession])
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.set_ylim(ylim)
  ax.set_xlabel('Trials')
  return subScores

#-------------------------------------------------------------------------------

def plot_duration_learning(
    ax,
    allPlayerLearningData,
    whichSession,
    savepath,
    colours,
    bunch_factor=1,
    plot_individuals=False,
    saveFig=False,
    title='',
    ylim=(0,32),
    ylabel='Trial duration (s)',
):
  """Plot every subject's trial duration across the session."""
  nsubjects = len(allPlayerLearningData[whichSession])
  if plot_individuals:
    for i in range(nsubjects):
      scores = allPlayerLearningData[whichSession][i]["timesToGoal"]
      if whichSession != cs.SESSION_SCANNER:
        scores = [i-10 for i in scores]
      ax.plot(scores, alpha=0.1)

  subScores = [allPlayerLearningData[whichSession][i]["timesToGoal"]
    for i in range(len(allPlayerLearningData[whichSession]))]
  subScores = np.asarray(subScores, dtype=np.float)
  subScores = subScores[~np.isnan(subScores)]  # remove nans.

  # Average across adjacent trials using a bunch_factor
  subScores = np.reshape(subScores, (bunch_factor, -1, nsubjects), order='F')
  subScores = np.mean(subScores, axis=0)
  # Mean and sem
  meanScores =  np.nanmean(subScores, axis=1)
  seScores = np.nanstd(subScores, axis=1) / np.sqrt(nsubjects)
  subtrials = np.linspace(1,bunch_factor*meanScores.size, meanScores.size)

  if whichSession != cs.SESSION_SCANNER:
    meanScores -= 10  # Correct for a data preprocessing error. Training and test sessions have different durations.

  # Plot the learning.
  ax.scatter(subtrials, meanScores, color=colours[whichSession])
  ax.errorbar(subtrials, meanScores, seScores, color=colours[whichSession])
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.set_ylim(ylim)
  ax.set_xlabel('Trials')
  return subScores

#-------------------------------------------------------------------------------
