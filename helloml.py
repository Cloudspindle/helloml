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

from ml import init,load_data,define_neural_net,train_neural_net,test_neural_net
training_set_size=100
test_set_size=2
num_samples=200               # 200 samples are in the motion examples

# neural network node parameters
num_inputs=num_samples        # we make the number of neural inputs equal to the number of samples in the motion examples
num_hidden=512
num_outputs=2
training_cycles=100

init() #define global variables
base = "https://raw.githubusercontent.com/Cloudspindle/helloml/master/"
base = "./"

training_clock_wise      = load_data(base+"ClockwiseZero_accel.200.csv",fuzz=training_set_size,fuzzsize=0.01)
training_anti_clock_wise = load_data(base+"AntiClockwiseZero_accel.200.csv",fuzz=training_set_size,fuzzsize=0.01)
test_clock_wise          = load_data(base+"ChloeClockwise_accel.200.csv")
test_anti_clock_wise     = load_data(base+"ChloeAntiClock_accel.200.csv")

raw_fig, [a,b,c,d] = plt.subplots(4,figsize=[25, 6])  

raw_fig.suptitle('This is a the raw input data', fontsize=24,color='red')
mixcolors = ['b','g','r','c','m','y'] 
linestyles = [ '--', '-.', ':']

for i in range(0,training_set_size):
  a.plot(training_clock_wise[i],linestyle=linestyles[i%len(linestyles)],color=mixcolors[i%len(mixcolors)])
a.set_title("training_clock_wise")
a.set_ylabel("G")

for i in range(0,training_set_size):
  b.plot(training_anti_clock_wise[i],linestyle=linestyles[i%len(linestyles)],color=mixcolors[i%len(mixcolors)])
b.set_title("training_anti_clock_wise")
b.set_ylabel("G")

c.plot(test_clock_wise,color='orange')
c.set_title("test_clock_wise")
c.set_ylabel("G")

d.plot(test_anti_clock_wise,color='red')
d.set_title("test_anti_clock_wise")
d.set_ylabel("G")

plt.show()

# net=define_neural_net(num_inputs,num_hidden,num_outputs) #define the neural network
# train_neural_net(net,training_cycles) #train the neural network
# test_neural_net(net) #test the neural network with the two test examples (the blue signal line is the clockwise test example and the orange signal line is the anticlockwise test example)
#!rm -rf helloml
