import csv
import matplotlib.pyplot as plt
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

input_set_size=100
test_set_size=2
num_samples=200               # 200 samples are in the motion examples

# neural network node parameters
num_inputs=num_samples        # we make the number of neural inputs equal to the number of samples in the motion examples
num_hidden=512
num_outputs=2
training_cycles=100

# gfig, gaxis = plt.subplots(5,figsize=(30,15))  

def init():
  global input_set,output_set,test_set
  global input_set_size,test_set_size,num_samples 

  input_set=np.arange(input_set_size*num_samples)
  input_set=input_set.reshape(input_set_size,num_samples)
  input_set=input_set.astype('float32')

  output_set=np.arange(input_set_size)
  output_set=output_set.astype('int')

  test_set=np.arange(test_set_size*num_samples)
  test_set=test_set.reshape(test_set_size,num_samples)
  test_set=test_set.astype('float32')

def load_y(url):
  headers = ['ts', 'x', 'y', 'z','m'] 
  data=pd.read_csv(url,sep=',',names=headers,header=None,parse_dates=True,index_col=0,infer_datetime_format=True )
  return data.y.to_numpy()
  

def load_data():
  global input_set,output_set,test_set
  
  url = "https://raw.githubusercontent.com/Cloudspindle/helloml/master/ClockwiseZero_accel.200.csv"

  headers = ['ts', 'x', 'y', 'z','m'] 
  data=pd.read_csv(url,sep=',',names=headers,header=None,parse_dates=True,index_col=0,infer_datetime_format=True )
  
  url =  "https://raw.githubusercontent.com/Cloudspindle/helloml/master/AntiClockwiseZero_accel.200.csv"
  # load the anticlockwise gesture AntiClockwiseZero_accel.200.csv as the second training example
  
  anti_data=pd.read_csv(url,sep=',',names=headers,header=None,parse_dates=True,index_col=0,infer_datetime_format=True )
  anti_clock_y=anti_data.y.to_numpy()
  clock_y=data.y.to_numpy()

  url = "https://raw.githubusercontent.com/Cloudspindle/helloml/master/ChloeClockwise_accel.200.csv"
  # upload the data from the file ChloeClockwise_accel.200.csv - real test data
  
  test_clock_data=pd.read_csv(url,sep=',',names=headers,header=None,parse_dates=True,index_col=0,infer_datetime_format=True )
  test_clock_y=test_clock_data.y.to_numpy()

  url = "https://raw.githubusercontent.com/Cloudspindle/helloml/master/ChloeAntiClock_accel.200.csv"
  # upload the dat from the file ChloeAntiClock_accel.200.csv file - real test data
 
  test_anti_clock_data=pd.read_csv(url,sep=',',names=headers,header=None,parse_dates=True,index_col=0,infer_datetime_format=True )
  test_anti_clock_y=test_anti_clock_data.y.to_numpy()


  for i in range(0,input_set_size):
    if (i%2==0):
      # create the noisy clockwise signal
      clock_y += 0.01*np.random.randn(*clock_y.shape)
      input_set[i]=clock_y
      output_set[i]=1  # 1 means it's a clockwise signal
    else:
      # create the noisy anticlockwise signal
      anti_clock_y += 0.01*np.random.randn(*anti_clock_y.shape)
      input_set[i]=anti_clock_y
      output_set[i]=0  # zero means that it's an anticlockwise signal

  # assign the test set
  test_set[0]=test_clock_y
  test_set[1]=test_anti_clock_y

  # reshape
  input_set=input_set.reshape(input_set.shape[0],num_samples)
  input_set=input_set.astype('float32')
  #input_set/=1
  output_set=utils.to_categorical(output_set,num_outputs)

  return input_set,output_set,test_set

def show_input_example():
  global input_set,output_set,test_set

  random_selection=round(random.uniform(0, input_set_size))
  print(random_selection)
  print(random_selection%2)
  # fig=plt.figure("Input Dataset",figsize=(10,5))
  gaxis[0].plot(input_set[random_selection],color='red',label="Clockwise")
  gaxis[0].set_title("Training Clockwise")
  gaxis[0].set_ylabel('Acceleration Magniture Gs', fontsize=18)
  gaxis[0].set_xlabel('Sample #', fontsize=18)

  gaxis[1].plot(test_set[random_selection%2],color='blue',label="Anti Clockwise")
  gaxis[1].set_title("Training Anti Clockwise")

  # fig.suptitle('Input  Dataset', fontsize=22)
  gaxis[1].set_xlabel('Sample #', fontsize=18)
  gaxis[1].set_ylabel('Acceleration Magniture Gs', fontsize=18)
  # plt.legend()
  # mng = plt.get_current_fig_manager()
  # mng.window.showMaximized()
  # plt.show()

def define_neural_net(num_inputs,num_hidden_nodes,num_outputs):
  global input_set,output_set,test_set

  net = tf.keras.Sequential()
  net.add(tf.keras.layers.Dense(num_hidden_nodes, activation='sigmoid', input_shape=(num_inputs,)))
  net.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))
  net.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy']) 
  net.summary()
  return net

def train_neural_net(net,training_cycles):
  global input_set,output_set,test_set
  
  log=net.fit(input_set, output_set, epochs=training_cycles)
  loss = log.history['loss']
  epochs = range(1,len(loss)+1)
  # fig=plt.figure('Training',figsize=(10,5))
  gaxis[2].set_xlabel('Epoch', fontsize=18)
  gaxis[2].set_ylabel('Loss', fontsize=18)

  gaxis[2].plot(epochs,loss,'g.',label='Training loss')
  gaxis[2].set_title("Training loss")

  # plt.legend()
  # mng = plt.get_current_fig_manager()
  # mng.window.showMaximized()
  # plt.show()

def test_neural_net(net):
  global input_set,output_set,test_set
  # fig=plt.figure('Test Data',figsize=(10,5))
  gaxis[3].plot(test_set[0],color='red',label="Clockwise")
  gaxis[3].set_title("Test Data ")
  gaxis[3].text(0.5, 0.2, 'Classified as Clockwise', size=30,color='red', 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square',       ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))
  gaxis[4].set_xlabel('Sample #', fontsize=18)
  gaxis[4].set_ylabel('Acceleration Magniture Gs', fontsize=18)
  # plt.legend()

  gaxis[4].plot(test_set[1],color='blue',label="Anti clockwise")
  gaxis[4].text(0.5, 0.2, 'Classified as Anti Clockwise',size=30, color='blue', 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='square',       ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))

  gaxis[4].set_title("Test Data ")
  # fig.suptitle('Test Data', fontsize=22)
  gaxis[3].set_xlabel('Sample #', fontsize=18)
  gaxis[3].set_ylabel('Acceleration Magniture Gs', fontsize=18)
  # plt.legend()
  # mng = plt.get_current_fig_manager()
  # mng.window.showMaximized()
  # plt.show()
  # predictions=net.predict(test_set,batch_size=10,verbose=10)
  #print(predictions)
  rounded_predictions=net.predict_classes(test_set,batch_size=10,verbose=10)
  #print(rounded_predictions)
  i=0
  while i < rounded_predictions.size:
    if rounded_predictions[i] == 1:
      print("clockwise")
   
    else:
      print("anticlockwise")
   
    i+=1
  gfig.tight_layout()
  plt.show()

