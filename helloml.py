#--------------------------------------------------------------------------------
#  Copyright (C) 2015 - 2020 Cloudspindle Inc.
#  HelloMotion.ipynb: - 'hello world' example for Deep Learning using
#                     - motion data generated from the MeKit platform.
#                     - Learn to classify clockwise and anticlockwise motions
#  License Terms:
#  Permission is hereby granted to use this software freely under the terms of
#  the free bsd license: https://www.freebsd.org/copyright/freebsd-license.html
#--------------------------------------------------------------------------------
#!git clone https://github.com/Cloudspindle/helloml.git
import matplotlib.pyplot as plt

from ml import init,load_y,load_data,show_input_example,define_neural_net,show_input_example,define_neural_net,train_neural_net,test_neural_net
input_set_size=100
test_set_size=2
num_samples=200               # 200 samples are in the motion examples

# neural network node parameters
num_inputs=num_samples        # we make the number of neural inputs equal to the number of samples in the motion examples
num_hidden=512
num_outputs=2
training_cycles=100

init() #define global variables
# (input_set,output_set,test_set)=load_data() #load the data (input training, output and test) NB - You manually #now need to load the following files in this order (ClockwiseZero_accel.200.csv, AntiClockwiseZero_accel.200.csv, ChloeClockwise_accel.200.csv, ChloeAntiClock_accel.200.csv). The first two are the training examples from which a full 100 example #training set will be built. The second two and the two examples #that will be tested after training.
base = "https://raw.githubusercontent.com/Cloudspindle/helloml/master/"
base = "./"
training_clock_wise      = load_y(base+"ClockwiseZero_accel.200.csv")
training_anti_clock_wise = load_y(base+"AntiClockwiseZero_accel.200.csv")
test_clock_wise          = load_y(base+"ChloeClockwise_accel.200.csv")
test_anti_clock_wise     = load_y(base+"ChloeAntiClock_accel.200.csv")

raw_fig, [a,b,c,d] = plt.subplots(4,figsize=(30,15))  
raw_fig.suptitle('This is a the raw input data', fontsize=24,color='red')


a.plot(training_clock_wise,color='blue')
a.set_title("training_clock_wise")
a.set_ylabel("G")

b.plot(training_anti_clock_wise,color='green')
b.set_title("training_anti_clock_wise")
b.set_ylabel("G")

c.plot(test_clock_wise,color='orange')
c.set_title("test_clock_wise")
c.set_ylabel("G")

d.plot(test_anti_clock_wise,color='red')
d.set_title("test_anti_clock_wise")
d.set_ylabel("G")

plt.show()
# show_input_example() #show an example of the data (input training v test) - rerun to #randomly select and example

# net=define_neural_net(num_inputs,num_hidden,num_outputs) #define the neural network
# train_neural_net(net,training_cycles) #train the neural network
# test_neural_net(net) #test the neural network with the two test examples (the blue signal line is the clockwise test example and the orange signal line is the anticlockwise test example)
#!rm -rf helloml
