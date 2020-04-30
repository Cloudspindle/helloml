
# !pip install tensorflow==2.0.0-beta0
#--------------------------------------------------------------------------------
#  Copyright (C) 2015 - 2020 Cloudspindle Inc.
#  HelloMotion.ipynb: - 'hello world' example for Deep Learning using
#                     - motion data generated from the MeKit platform.
#                     - Learn to classify clockwise and anticlockwise motions
#  License Terms:
#  Permission is hereby granted to use this software freely under the terms of
#  the free bsd license: https://www.freebsd.org/copyright/freebsd-license.html
#--------------------------------------------------------------------------------

!git clone https://github.com/Cloudspindle/helloml.git
import helloml.ml
import csv
import matplotlib.pyplot as plt
from google.colab import files
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import tensorflow as tf
from keras import layers
from keras import utils


"""#load the data (input training, output and test) NB - You manually #now need to load the following files in this order (ClockwiseZero_accel.200.csv, AntiClockwiseZero_accel.200.csv, ChloeClockwise_accel.200.csv, ChloeAntiClock_accel.200.csv). The first two are the training examples from which a full 100 example #training set will be built. The second two and the two examples #that will be tested after training."""

(input_set,output_set,test_set)=load_data()

"""#show an example of the data (input training v test) - rerun to #randomly select and example"""

show_input_example()

"""define the neural network"""

net=define_neural_net(num_inputs,num_hidden,num_outputs)

"""train the neural network"""

train_neural_net(net,training_cycles)

"""test the neural network with the two test examples (the blue signal line is the clockwise test example and the orange signal line is the anticlockwise test example)"""

test_neural_net()
