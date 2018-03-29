import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from scipy.spatial.distance import euclidean
import uuid
import pickle
from scipy.spatial import Delaunay
import itertools
from sklearn.neighbors import NearestNeighbors
import tqdm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os
import subprocess
import sys
from scipy import signal

#RUNNING PARAMETER:
new = False# True: generate a new membrane, False: use membrane called from 'membrane_id'
membrane_id = "e795a811-6cca-464c-bdaf-3d91d27c592e.pkl" # e795a811-6cca-464c-bdaf-3d91d27c592e'8f561921-c1c2-4a1b-9b5a-2281f1f12aa1.pkl' # name of membrane to use if new = False

read_from_dna =False #True: generate new membrane from DNA, False Ignore DNA
dna_net_dim = 2 #number of dimensions of input DNA network
input_dna = '1101100101111100111011001001111000010'#'1101100101111100111011001001111000010' #input DNA

cplusplus = True # Run in fast mode using C++ binaries
graph = False #Plot graph in real time, only when cplusplus is False (2d only)

#LEARNING PARAMETERS:
train_batch_size = 200 #Size of learning batch
test_batch_size = 40 #Size of testing batch
nb_classes = 12 #Number of classes to learn/test from

#NETWORK GENERATION PARAMETERS
net_dim = 3#dimension of the membrane (set to 2 or 3)
g = 0.0 #gravitational constant
dt = 0.001 #time-step
nb_row = 5 #number of masses/row
nb_col = 5 #number of masses/column
nb_hei = 5 #number of masses/height (if 3d)
x_axis_length = 10*nb_row #size of x axis
y_axis_length = 10*nb_col #size of y axis
z_axis_length = 10*nb_hei #size of z axis (if 3d)
delaunay=True #True: Delaunay, False: Gaussian Connectivity
quadratic_spring = True#True: quadratic springs, False: Linear springs
fixed_plate = False #Fixed Plate at the bottom (if 3d)
random_input = False # random input nodes
gradientMatrix = False #learn the gradient of the spring length (True: learn gradient, False: learn length)

#NETWORK ASSESSMENT PARAMETERS
washout_criteria = 0.15 #Criteria for Washout Assessment standard deviation
washout_max_permitted_time = 3 #Maximum time allowed for Membrane to Washout

#REGULATING COEFFICIENT (Values to change the Ratio input/feedback, set both to 1.0 if not needed)
w_input_over_coef = 1000.0
w_feed_over_coef = 1.0

#FORCE PARAM
force_param = False #Force Selection of Parameters when generating new Membrane
damp_gen_force = 1 #Membrane Dampering Generation Parameter
sig_force = 0.01 #Membrane Spring Coefficients Standard Deviation
stiff_force = 1000 #Membrane Stiffness Generation Parameter


