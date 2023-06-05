#First we import some libraries
#Json for loading and saving the model (optional)
import json
#Pandas for saving datasets
import pandas as pd
#matplotlib for rendering
import matplotlib.pyplot as plt
#numpy for handeling matrix operations
import numpy as np
#time, to, well... keep track of time
import time
#iPython display for making sure we can render the frames
from IPython import display
#seaborn for rendering
import seaborn
#pympler to find memory leaks
# from pympler import muppy, summary, refbrowser
# import objgraph
#Keras is a deep learning libarary
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import tf_agents
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import nest_utils

import PIL.Image
import copy
from os import path, mkdir, chdir

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

import os
from datetime import datetime
import math
import pprint

#Setup for memory profiling
import sys
# from memory_profiler import profile
# %load_ext memory_profiler
#Setup matplotlib so that it runs nicely in iPython
# %matplotlib inline
#setting up seaborn
seaborn.set()

print("The 'ball' is a leaf that drifts downward, randomly moving lateraly as well as descending. Attention starts on the ball at the beginning of each trial.")
print("In this version, the agent is trained without any primary information on the location of attention")


## Hyperparameters

# How many trials back to store in each observation
LOOK_BACK = 10

# Percentage of inserted white pixel noise. 0=no noise, 1=opaque background
NOISE = 0.5

ITERS = 1500
N = 5


#%%
class trainCatchwithAttention(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(self):
        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(8 * 2) - 1,
            name='action')  # The number of possible actions is equal to the number of grid tiles times 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32, minimum=0, maximum=1,
            name='observation')  # The observation space is equal to two stacked game grids

        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col

        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2))
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self.step_count = 0  # internal counter for the current step

        self._state = self._append_to_memory(self._draw_state())

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col
        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land
        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self._state = self._append_to_memory(self._draw_state())
        self.step_count = 0
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle
        if action in np.arange(0, 8):  # left
            move = -1
        elif action in np.arange(8, 16):
            move = 1  # right

        # Here we define how action selection affects the locus of attention
        # Rescale action selection to exclude the chosen move
        temp_vec = np.array([0, 8])
        temp_mat = np.array(
            [temp_vec, temp_vec + 1, temp_vec + 2, temp_vec + 3, temp_vec + 4, temp_vec + 5, temp_vec + 6,
             temp_vec + 7])
        attn_action = np.argwhere(temp_mat == action)[0][0]
        # Attention movement options are stationary or 8 possible directions
        attn_moves = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
        delta_col, delta_row = attn_moves[attn_action]

        # Apply the change in attention locus
        self.attn_row = self.attn_row + delta_row
        self.attn_col = self.attn_col + delta_col

        # Calculate direction unit vector between attn and ball
        row_diff = self.attn_row - self.ball_row
        col_diff = self.attn_col - self.attn_col
        distance = math.sqrt(row_diff ** 2 + col_diff ** 2)

        if distance != 0:
            unit_dir_row = row_diff / distance
            unit_dir_col = col_diff / distance
        else:
            unit_dir_row = 0
            unit_dir_col = 0

        if self.attn_row < 1:  # Check to make sure attention field is within bounds
            self.attn_row = 1  # undo the mistake
        if self.attn_row > self.grid_size - 2:
            self.attn_row = self.grid_size - 2
        if self.attn_col < 1:  # Check to make sure attention field is within bounds
            self.attn_col = 1  # undo the mistake
        if self.attn_col > self.grid_size - 2:
            self.attn_col = self.grid_size - 2

            # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2));
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        # Update the positions of the moving pieces
        self.paddle_loc = self.paddle_loc + move
        if self.paddle_loc < 1 or self.paddle_loc > self.grid_size - 2:  # Check to make sure paddle is within bounds
            self.paddle_loc = self.paddle_loc - move  # undo the mistake

        # Update ball position
        self.ball_row = self.ball_row + 1  # ball decends one space per timestep

        if self.ball_col < self.landing:  # adjust to the right if ball is left of landing zone
            self.ball_col = self.ball_col + 1
        elif self.ball_col > self.landing:  # adjust to the left if ball is right of landing zone
            self.ball_col = self.ball_col - 1

        # Don't let the ball leave the playing field
        if self.ball_col < 0:  # Check to make sure the ball is within bounds
            self.ball_col = 0  # undo the mistake
        if self.ball_col > self.grid_size - 1:
            self.ball_col = self.grid_size - 1

            # Update the game state in the model
        self._state = self._append_to_memory(self._draw_state())

        # Scoring
        if self.ball_row == self.grid_size - 1:  # Check if the ball has hit the bottom
            self._episode_ended = True
            if abs(self.ball_col - self.paddle_loc) <= 1:  # Did the player catch the ball
                return ts.termination(self._state, reward=2)  # Good!
            else:
                return ts.termination(self._state, reward=-2)  # Bad!
        elif self.ball_row in self.attn_rowspan and self.ball_col in self.attn_colspan:  # small reward for attending the ball
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.5, discount=0.4)
        else:  # Punishment for attending empty space
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=-0.5, discount=0.4)

    def _draw_state(self):
        attentional_canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Draw attention on the attentional space
        #         attentional_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 1 #attention locus is a 3 by 3 square

        # Draw a noisy visual space
        noise_level = NOISE  # between 0 and 0.5, gives the percentage of playing feild to be filled with inserted while pixels
        noise_array = np.concatenate((np.repeat(1, noise_level * (self.grid_size ** 2)),
                                      np.repeat(0, (1 - noise_level) * (self.grid_size ** 2))))

        visual_canvas = np.random.permutation(noise_array).reshape((self.grid_size, self.grid_size))
        visual_canvas = visual_canvas.astype('int32')
        visual_canvas[self.grid_size - 1, :] = 0  # Paddle row has no noise

        # Remove noise from attended spotlight
        visual_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1] + 1,
        self.attn_colspan[0]:self.attn_colspan[-1] + 1] = 0

        # Draw objects
        visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        visual_canvas[self.grid_size - 1,
        self.paddle_loc - 1:self.paddle_loc + 2] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

        return canvas

    def _append_to_memory(self, state):
        memory = copy.deepcopy(self.memory_buffer)
        memory = np.delete(memory, np.arange(self.grid_size * 2), 0)  # Delete the oldest memory
        updated_memory = np.append(memory, state, axis=0)  # Append most recent observation
        self.memory_buffer = updated_memory  # Update the memory buffer
        return updated_memory


#%%
class impotentAttention(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(self):
        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(8 * 2) - 1,
            name='action')  # The number of possible actions is equal to the number of grid tiles times 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32, minimum=0, maximum=1,
            name='observation')  # The observation space is equal to two stacked game grids

        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col

        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2))
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self.step_count = 0  # internal counter for the current step

        self._state = self._append_to_memory(self._draw_state())

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col
        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land
        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self._state = self._append_to_memory(self._draw_state())
        self.step_count = 0
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle
        if action in np.arange(0, 8):  # left
            move = -1
        elif action in np.arange(8, 16):
            move = 1  # right

        # Here we define how action selection affects the locus of attention
        # Rescale action selection to exclude the chosen move
        temp_vec = np.array([0, 8])
        temp_mat = np.array(
            [temp_vec, temp_vec + 1, temp_vec + 2, temp_vec + 3, temp_vec + 4, temp_vec + 5, temp_vec + 6,
             temp_vec + 7])
        attn_action = np.argwhere(temp_mat == action)[0][0]
        # Attention movement options are stationary or 8 possible directions
        attn_moves = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
        delta_col, delta_row = attn_moves[attn_action]

        # Apply the change in attention locus
        self.attn_row = self.attn_row + delta_row
        self.attn_col = self.attn_col + delta_col

        # Calculate direction unit vector between attn and ball
        row_diff = self.attn_row - self.ball_row
        col_diff = self.attn_col - self.attn_col
        distance = math.sqrt(row_diff ** 2 + col_diff ** 2)

        if distance != 0:
            unit_dir_row = row_diff / distance
            unit_dir_col = col_diff / distance
        else:
            unit_dir_row = 0
            unit_dir_col = 0

        if self.attn_row < 1:  # Check to make sure attention field is within bounds
            self.attn_row = 1  # undo the mistake
        if self.attn_row > self.grid_size - 2:
            self.attn_row = self.grid_size - 2
        if self.attn_col < 1:  # Check to make sure attention field is within bounds
            self.attn_col = 1  # undo the mistake
        if self.attn_col > self.grid_size - 2:
            self.attn_col = self.grid_size - 2

            # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2));
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        # Update the positions of the moving pieces
        self.paddle_loc = self.paddle_loc + move
        if self.paddle_loc < 1 or self.paddle_loc > self.grid_size - 2:  # Check to make sure paddle is within bounds
            self.paddle_loc = self.paddle_loc - move  # undo the mistake

        # Update ball position
        self.ball_row = self.ball_row + 1  # ball decends one space per timestep

        if self.ball_col < self.landing:  # adjust to the right if ball is left of landing zone
            self.ball_col = self.ball_col + 1
        elif self.ball_col > self.landing:  # adjust to the left if ball is right of landing zone
            self.ball_col = self.ball_col - 1

        # Don't let the ball leave the playing field
        if self.ball_col < 0:  # Check to make sure the ball is within bounds
            self.ball_col = 0  # undo the mistake
        if self.ball_col > self.grid_size - 1:
            self.ball_col = self.grid_size - 1

            # Update the game state in the model
        self._state = self._append_to_memory(self._draw_state())

        # Scoring
        if self.ball_row == self.grid_size - 1:  # Check if the ball has hit the bottom
            self._episode_ended = True
            if abs(self.ball_col - self.paddle_loc) <= 1:  # Did the player catch the ball
                return ts.termination(self._state, reward=0)  # Good!
            else:
                return ts.termination(self._state, reward=-0)  # Bad!
        elif self.ball_row in self.attn_rowspan and self.ball_col in self.attn_colspan:  # small reward for attending the ball
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.5, discount=0.4)
        else:  # Punishment for attending empty space
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=-0.5, discount=0.4)

    def _draw_state(self):
        attentional_canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        #         #Draw attention on the attentional space
        #         attentional_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 1 #attention locus is a 3 by 3 square

        # Draw a noisy visual space
        noise_level = NOISE  # between 0 and 0.5, gives the percentage of playing feild to be filled with inserted while pixels
        noise_array = np.concatenate((np.repeat(1, noise_level * (self.grid_size ** 2)),
                                      np.repeat(0, (1 - noise_level) * (self.grid_size ** 2))))

        visual_canvas = np.random.permutation(noise_array).reshape((self.grid_size, self.grid_size))
        visual_canvas = visual_canvas.astype('int32')
        visual_canvas[self.grid_size - 1, :] = 0  # Paddle row has no noise

        # DON'T Remove noise from attended spotlight
        #         visual_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 0

        # Draw objects
        visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        visual_canvas[self.grid_size - 1,
        self.paddle_loc - 1:self.paddle_loc + 2] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

        return canvas

    def _append_to_memory(self, state):
        memory = copy.deepcopy(self.memory_buffer)
        memory = np.delete(memory, np.arange(self.grid_size * 2), 0)  # Delete the oldest memory
        updated_memory = np.append(memory, state, axis=0)  # Append most recent observation
        self.memory_buffer = updated_memory  # Update the memory buffer
        return updated_memory


#%%
class trainCatch(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(self):
        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(8 * 2) - 1,
            name='action')  # The number of possible actions is equal to the number of grid tiles times 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32, minimum=0, maximum=1,
            name='observation')  # The observation space is equal to two stacked game grids

        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col

        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2))
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self.step_count = 0  # internal counter for the current step

        self._state = self._append_to_memory(self._draw_state())

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col
        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land
        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self._state = self._append_to_memory(self._draw_state())
        self.step_count = 0
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle
        if action in np.arange(0, 8):  # left
            move = -1
        elif action in np.arange(8, 16):
            move = 1  # right

        # Here we define how action selection affects the locus of attention
        # Rescale action selection to exclude the chosen move
        temp_vec = np.array([0, 8])
        temp_mat = np.array(
            [temp_vec, temp_vec + 1, temp_vec + 2, temp_vec + 3, temp_vec + 4, temp_vec + 5, temp_vec + 6,
             temp_vec + 7])
        attn_action = np.argwhere(temp_mat == action)[0][0]
        # Attention movement options are stationary or 8 possible directions
        attn_moves = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
        delta_col, delta_row = attn_moves[attn_action]

        # Apply the change in attention locus
        self.attn_row = self.attn_row + delta_row
        self.attn_col = self.attn_col + delta_col

        # Calculate direction unit vector between attn and ball
        row_diff = self.attn_row - self.ball_row
        col_diff = self.attn_col - self.attn_col
        distance = math.sqrt(row_diff ** 2 + col_diff ** 2)

        if distance != 0:
            unit_dir_row = row_diff / distance
            unit_dir_col = col_diff / distance
        else:
            unit_dir_row = 0
            unit_dir_col = 0

        if self.attn_row < 1:  # Check to make sure attention field is within bounds
            self.attn_row = 1  # undo the mistake
        if self.attn_row > self.grid_size - 2:
            self.attn_row = self.grid_size - 2
        if self.attn_col < 1:  # Check to make sure attention field is within bounds
            self.attn_col = 1  # undo the mistake
        if self.attn_col > self.grid_size - 2:
            self.attn_col = self.grid_size - 2

            # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2));
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        # Update the positions of the moving pieces
        self.paddle_loc = self.paddle_loc + move
        if self.paddle_loc < 1 or self.paddle_loc > self.grid_size - 2:  # Check to make sure paddle is within bounds
            self.paddle_loc = self.paddle_loc - move  # undo the mistake

        # Update ball position
        self.ball_row = self.ball_row + 1  # ball decends one space per timestep

        if self.ball_col < self.landing:  # adjust to the right if ball is left of landing zone
            self.ball_col = self.ball_col + 1
        elif self.ball_col > self.landing:  # adjust to the left if ball is right of landing zone
            self.ball_col = self.ball_col - 1

        # Don't let the ball leave the playing field
        if self.ball_col < 0:  # Check to make sure the ball is within bounds
            self.ball_col = 0  # undo the mistake
        if self.ball_col > self.grid_size - 1:
            self.ball_col = self.grid_size - 1

            # Update the game state in the model
        self._state = self._append_to_memory(self._draw_state())

        # Scoring
        if self.ball_row == self.grid_size - 1:  # Check if the ball has hit the bottom
            self._episode_ended = True
            if abs(self.ball_col - self.paddle_loc) <= 1:  # Did the player catch the ball
                return ts.termination(self._state, reward=2)  # Good!
            else:
                return ts.termination(self._state, reward=-2)  # Bad!
        elif self.ball_row in self.attn_rowspan and self.ball_col in self.attn_colspan:  # small reward for attending the ball
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=0.4)
        else:  # Punishment for attending empty space
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=-0.0, discount=0.4)

    def _draw_state(self):
        attentional_canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Draw attention on the attentional space
        #         attentional_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 1 #attention locus is a 3 by 3 square

        # Draw a noisy visual space
        noise_level = NOISE  # between 0 and 0.5, gives the percentage of playing feild to be filled with inserted while pixels
        noise_array = np.concatenate((np.repeat(1, noise_level * (self.grid_size ** 2)),
                                      np.repeat(0, (1 - noise_level) * (self.grid_size ** 2))))

        visual_canvas = np.random.permutation(noise_array).reshape((self.grid_size, self.grid_size))
        visual_canvas = visual_canvas.astype('int32')
        visual_canvas[self.grid_size - 1, :] = 0  # Paddle row has no noise

        # Remove noise from attended spotlight
        visual_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1] + 1,
        self.attn_colspan[0]:self.attn_colspan[-1] + 1] = 0

        # Draw objects
        visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        visual_canvas[self.grid_size - 1,
        self.paddle_loc - 1:self.paddle_loc + 2] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

        return canvas

    def _append_to_memory(self, state):
        memory = copy.deepcopy(self.memory_buffer)
        memory = np.delete(memory, np.arange(self.grid_size * 2), 0)  # Delete the oldest memory
        updated_memory = np.append(memory, state, axis=0)  # Append most recent observation
        self.memory_buffer = updated_memory  # Update the memory buffer
        return updated_memory


#%%
class impotentCatch(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(self):
        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(8 * 2) - 1,
            name='action')  # The number of possible actions is equal to the number of grid tiles times 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32, minimum=0, maximum=1,
            name='observation')  # The observation space is equal to two stacked game grids

        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col

        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2))
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self.step_count = 0  # internal counter for the current step

        self._state = self._append_to_memory(self._draw_state())

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col
        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land
        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self._state = self._append_to_memory(self._draw_state())
        self.step_count = 0
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle
        if action in np.arange(0, 8):  # left
            move = -1
        elif action in np.arange(8, 16):
            move = 1  # right

        # Here we define how action selection affects the locus of attention
        # Rescale action selection to exclude the chosen move
        temp_vec = np.array([0, 8])
        temp_mat = np.array(
            [temp_vec, temp_vec + 1, temp_vec + 2, temp_vec + 3, temp_vec + 4, temp_vec + 5, temp_vec + 6,
             temp_vec + 7])
        attn_action = np.argwhere(temp_mat == action)[0][0]
        # Attention movement options are stationary or 8 possible directions
        attn_moves = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
        delta_col, delta_row = attn_moves[attn_action]

        # Apply the change in attention locus
        self.attn_row = self.attn_row + delta_row
        self.attn_col = self.attn_col + delta_col

        # Calculate direction unit vector between attn and ball
        row_diff = self.attn_row - self.ball_row
        col_diff = self.attn_col - self.attn_col
        distance = math.sqrt(row_diff ** 2 + col_diff ** 2)

        if distance != 0:
            unit_dir_row = row_diff / distance
            unit_dir_col = col_diff / distance
        else:
            unit_dir_row = 0
            unit_dir_col = 0

        if self.attn_row < 1:  # Check to make sure attention field is within bounds
            self.attn_row = 1  # undo the mistake
        if self.attn_row > self.grid_size - 2:
            self.attn_row = self.grid_size - 2
        if self.attn_col < 1:  # Check to make sure attention field is within bounds
            self.attn_col = 1  # undo the mistake
        if self.attn_col > self.grid_size - 2:
            self.attn_col = self.grid_size - 2

            # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2));
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        # Update the positions of the moving pieces
        self.paddle_loc = self.paddle_loc + move
        if self.paddle_loc < 1 or self.paddle_loc > self.grid_size - 2:  # Check to make sure paddle is within bounds
            self.paddle_loc = self.paddle_loc - move  # undo the mistake

        # Update ball position
        self.ball_row = self.ball_row + 1  # ball decends one space per timestep

        if self.ball_col < self.landing:  # adjust to the right if ball is left of landing zone
            self.ball_col = self.ball_col + 1
        elif self.ball_col > self.landing:  # adjust to the left if ball is right of landing zone
            self.ball_col = self.ball_col - 1

        # Don't let the ball leave the playing field
        if self.ball_col < 0:  # Check to make sure the ball is within bounds
            self.ball_col = 0  # undo the mistake
        if self.ball_col > self.grid_size - 1:
            self.ball_col = self.grid_size - 1

            # Update the game state in the model
        self._state = self._append_to_memory(self._draw_state())

        # Scoring
        if self.ball_row == self.grid_size - 1:  # Check if the ball has hit the bottom
            self._episode_ended = True
            if abs(self.ball_col - self.paddle_loc) <= 1:  # Did the player catch the ball
                return ts.termination(self._state, reward=2)  # Good!
            else:
                return ts.termination(self._state, reward=-2)  # Bad!
        elif self.ball_row in self.attn_rowspan and self.ball_col in self.attn_colspan:  # small reward for attending the ball
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=0.4)
        else:  # Punishment for attending empty space
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=-0.0, discount=0.4)

    def _draw_state(self):
        attentional_canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        #         #Draw attention on the attentional space
        #         attentional_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 1 #attention locus is a 3 by 3 square

        # Draw a noisy visual space
        noise_level = NOISE  # between 0 and 0.5, gives the percentage of playing feild to be filled with inserted while pixels
        noise_array = np.concatenate((np.repeat(1, noise_level * (self.grid_size ** 2)),
                                      np.repeat(0, (1 - noise_level) * (self.grid_size ** 2))))

        visual_canvas = np.random.permutation(noise_array).reshape((self.grid_size, self.grid_size))
        visual_canvas = visual_canvas.astype('int32')
        visual_canvas[self.grid_size - 1, :] = 0  # Paddle row has no noise

        # DO NOT Remove noise from attended spotlight
        #         visual_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 0

        # Draw objects
        visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        visual_canvas[self.grid_size - 1,
        self.paddle_loc - 1:self.paddle_loc + 2] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

        return canvas

    def _append_to_memory(self, state):
        memory = copy.deepcopy(self.memory_buffer)
        memory = np.delete(memory, np.arange(self.grid_size * 2), 0)  # Delete the oldest memory
        updated_memory = np.append(memory, state, axis=0)  # Append most recent observation
        self.memory_buffer = updated_memory  # Update the memory buffer
        return updated_memory


#%%
class trainAttention(py_environment.PyEnvironment):
    """
    Class catch is the actual game.
    In the game, balls, represented by white tiles, fall from the top.
    The goal is to catch the balls with a paddle
    """

    def __init__(self):
        self.grid_size = 10
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(8 * 2) - 1,
            name='action')  # The number of possible actions is equal to the number of grid tiles times 3
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32, minimum=0, maximum=1,
            name='observation')  # The observation space is equal to two stacked game grids

        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col

        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land

        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2))
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self.step_count = 0  # internal counter for the current step

        self._state = self._append_to_memory(self._draw_state())

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.ball_row = 0  # Start the ball at the top of the screen
        self.ball_col = int(np.random.randint(0, self.grid_size, size=1))  # What column does the ball appear in
        self.paddle_loc = math.floor(
            self.grid_size / 2 - 1)  # Starts paddle in the middle, column of leftmost corner of paddle
        self.attn_row = self.ball_row + 1  # Attention starts fixated on the ball
        self.attn_col = self.ball_col
        self.landing = np.random.randint(0, 10)  # Randomly predetermine where the ball will land
        self.memory_buffer = np.zeros((self.grid_size * 2 * LOOK_BACK, self.grid_size), dtype=np.int32)
        self._state = self._append_to_memory(self._draw_state())
        self.step_count = 0
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.step_count += 1  # Increment the step counter

        # here we define how action selection affects the location of the paddle
        if action in np.arange(0, 8):  # left
            move = -1
        elif action in np.arange(8, 16):
            move = 1  # right

        # Here we define how action selection affects the locus of attention
        # Rescale action selection to exclude the chosen move
        temp_vec = np.array([0, 8])
        temp_mat = np.array(
            [temp_vec, temp_vec + 1, temp_vec + 2, temp_vec + 3, temp_vec + 4, temp_vec + 5, temp_vec + 6,
             temp_vec + 7])
        attn_action = np.argwhere(temp_mat == action)[0][0]
        # Attention movement options are stationary or 8 possible directions
        attn_moves = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
        delta_col, delta_row = attn_moves[attn_action]

        # Apply the change in attention locus
        self.attn_row = self.attn_row + delta_row
        self.attn_col = self.attn_col + delta_col

        # Calculate direction unit vector between attn and ball
        row_diff = self.attn_row - self.ball_row
        col_diff = self.attn_col - self.attn_col
        distance = math.sqrt(row_diff ** 2 + col_diff ** 2)

        if distance != 0:
            unit_dir_row = row_diff / distance
            unit_dir_col = col_diff / distance
        else:
            unit_dir_row = 0
            unit_dir_col = 0

        if self.attn_row < 1:  # Check to make sure attention field is within bounds
            self.attn_row = 1  # undo the mistake
        if self.attn_row > self.grid_size - 2:
            self.attn_row = self.grid_size - 2
        if self.attn_col < 1:  # Check to make sure attention field is within bounds
            self.attn_col = 1  # undo the mistake
        if self.attn_col > self.grid_size - 2:
            self.attn_col = self.grid_size - 2

            # Represent attention location:
        self.attn_rowspan = list(range(self.attn_row - 1, self.attn_row + 2));
        self.attn_colspan = list(range(self.attn_col - 1, self.attn_col + 2))

        # Update the positions of the moving pieces
        self.paddle_loc = self.paddle_loc + move
        if self.paddle_loc < 1 or self.paddle_loc > self.grid_size - 2:  # Check to make sure paddle is within bounds
            self.paddle_loc = self.paddle_loc - move  # undo the mistake

        # Update ball position
        self.ball_row = self.ball_row + 1  # ball decends one space per timestep

        if self.ball_col < self.landing:  # adjust to the right if ball is left of landing zone
            self.ball_col = self.ball_col + 1
        elif self.ball_col > self.landing:  # adjust to the left if ball is right of landing zone
            self.ball_col = self.ball_col - 1

        # Don't let the ball leave the playing field
        if self.ball_col < 0:  # Check to make sure the ball is within bounds
            self.ball_col = 0  # undo the mistake
        if self.ball_col > self.grid_size - 1:
            self.ball_col = self.grid_size - 1

            # Update the game state in the model
        self._state = self._append_to_memory(self._draw_state())

        # Scoring
        if self.ball_row == self.grid_size - 1:  # Check if the ball has hit the bottom
            self._episode_ended = True
            if abs(self.ball_col - self.paddle_loc) <= 1:  # Did the player catch the ball
                return ts.termination(self._state, reward=0)  # Good!
            else:
                return ts.termination(self._state, reward=-0)  # Bad!
        elif self.ball_row in self.attn_rowspan and self.ball_col in self.attn_colspan:  # small reward for attending the ball
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.5, discount=0.4)
        else:  # Punishment for attending empty space
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=-0.5, discount=0.4)

    def _draw_state(self):
        attentional_canvas = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Draw attention on the attentional space
        #         attentional_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1]+1, self.attn_colspan[0]:self.attn_colspan[-1]+1] = 1 #attention locus is a 3 by 3 square

        # Draw a noisy visual space
        noise_level = NOISE  # between 0 and 0.5, gives the percentage of playing feild to be filled with inserted while pixels
        noise_array = np.concatenate((np.repeat(1, noise_level * (self.grid_size ** 2)),
                                      np.repeat(0, (1 - noise_level) * (self.grid_size ** 2))))

        visual_canvas = np.random.permutation(noise_array).reshape((self.grid_size, self.grid_size))
        visual_canvas = visual_canvas.astype('int32')
        visual_canvas[self.grid_size - 1, :] = 0  # Paddle row has no noise

        # Remove noise from attended spotlight
        visual_canvas[self.attn_rowspan[0]:self.attn_rowspan[-1] + 1,
        self.attn_colspan[0]:self.attn_colspan[-1] + 1] = 0

        # Draw objects
        visual_canvas[self.ball_row, self.ball_col] = 1  # draw ball
        visual_canvas[self.grid_size - 1,
        self.paddle_loc - 1:self.paddle_loc + 2] = 1  # draw paddle, which always attended

        canvas = np.concatenate((visual_canvas, attentional_canvas), axis=0)

        return canvas

    def _append_to_memory(self, state):
        memory = copy.deepcopy(self.memory_buffer)
        memory = np.delete(memory, np.arange(self.grid_size * 2), 0)  # Delete the oldest memory
        updated_memory = np.append(memory, state, axis=0)  # Append most recent observation
        self.memory_buffer = updated_memory  # Update the memory buffer
        return updated_memory


#%%
# Verify that the environment is coded properly.
attention_environment = trainAttention()
catch_environment = trainCatch()
impotent_catch_environment = impotentCatch()
impotent_attention_environment = impotentAttention()
catch_w_attention_environment = trainCatchwithAttention()
utils.validate_py_environment(attention_environment, episodes=5)
utils.validate_py_environment(catch_environment, episodes=5)
utils.validate_py_environment(impotent_catch_environment, episodes=5)
utils.validate_py_environment(impotent_attention_environment, episodes=5)
utils.validate_py_environment(catch_w_attention_environment, episodes=5)

# Convert to tf environment so that we can run this shit in parallel
catch_env = tf_py_environment.TFPyEnvironment(catch_environment)
impotent_catch_env = tf_py_environment.TFPyEnvironment(impotent_catch_environment)
attention_env = tf_py_environment.TFPyEnvironment(attention_environment)
impotent_attention_env = tf_py_environment.TFPyEnvironment(impotent_attention_environment)
catch_attention_env = tf_py_environment.TFPyEnvironment(catch_w_attention_environment)


#%% Setup to train
for subject in np.arange(N):

    q_net = q_network.QNetwork(
        attention_env.time_step_spec().observation,
        attention_env.action_spec(),
        fc_layer_params=(200,200,200))

    agent = dqn_agent.DqnAgent(
        time_step_spec=attention_env.time_step_spec(),
        action_spec=attention_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=0.2,
        optimizer=tf.optimizers.Adam(0.001))

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=catch_attention_env.batch_size,
        max_length=10000)

    def collect_training_data():
      dynamic_step_driver.DynamicStepDriver(
        env=catch_attention_env,
        policy=agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=1000).run()

    def train_agent():
      dataset = replay_buffer.as_dataset(
          sample_batch_size=100,
          num_steps=2)

      iterator = iter(dataset)

      loss = None
      for _ in range(100):
        trajectories, _ = next(iterator)
        loss = agent.train(experience=trajectories)

      print('Training loss: ', loss.loss.numpy())
      return loss.loss.numpy()

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        print('Average Return: {0}'.format(avg_return.numpy()[0]))
        return avg_return.numpy()[0]

    TESTING_ITERS = 50

    training_loss = []
    training_return = []
    attention_return = []
    catch_return = []
    impotent_catch_return = []
    impotent_attention_return = []

    for i in range(ITERS):
        i += 1
        collect_training_data()
        print('Step {}'.format(str(subject)+"."+str(i)))
        training_loss.append(train_agent())
        training_return.append(compute_avg_return(catch_attention_env, agent.policy, TESTING_ITERS)) # Compute average return for actual environment
        attention_return.append(compute_avg_return(attention_env, agent.policy, TESTING_ITERS))
        catch_return.append(compute_avg_return(catch_env, agent.policy, TESTING_ITERS))
        impotent_catch_return.append(compute_avg_return(impotent_catch_env, agent.policy, TESTING_ITERS))
        impotent_attention_return.append(compute_avg_return(impotent_attention_env, agent.policy, TESTING_ITERS))

    # Save results to dataframe
    if subject == 0:
        results = pd.DataFrame()
    results['combined_return'+str(subject)] = training_return;
    results['attention_return'+str(subject)] = attention_return; results['catch_return'+str(subject)] = catch_return;
    results['impotent_attention_return'+str(subject)] = impotent_attention_return; results['impotent_catch_return'+str(subject)] = impotent_catch_return;
    results['loss'+str(subject)] = training_loss
    results

# Save results df to file
results.sort_index(axis=1, inplace =True)
results.to_excel('unaware_catch_1500_group.xlsx', index=True)
