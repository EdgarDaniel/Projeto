#coding=utf-8
#COPYRIGHT EDGAR DANIEL
#THIS IS THE MAIN SCRIPT FOR TRAIN THE VGG16 NETWORK FOR GENDER RECOGNITION
#THIS TRAIN USES BATCHES AND EPOCHS FOR OPTIMIZATION
#IN THE END OF EACH EPOCH THE EVALUATE FUNCTION RUNS ALL THE VALIDATION DATASET TO METRICS
#IN COMMENT ARE FUNCTIONS THAT CALCULATES THE TIME EXECUTION FOR BENCHMARK TESTS
#THE LAST FUNCTION IN THIS FILE RELOAD THE CHECKPOINT FILE FROM THE SAVED NETWORK AND EXECUTES VALIDATION AND
#TEST DATASETS.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.utils import shuffle
import tensorflow as tf
import tflearn
from createData import *
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

#IMPORT IMPORTANT FUNCTIONS TO RTX GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#IMPORT MODELS DEFINITON FROM TF-SLIM lIGHTWEIGHT HIGH-LEVEL API OF TENSORFLOW
import tensorflow.contrib.slim as slim

#IMPORT VGG16 MODEL
from vgg import vgg_16

from time import time

from itertools import chain

##!!!!REMOVE BEFORE GIT
##from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

#VARIABLES WITH PATH FOR IMAGES TRAIN/VALIDATION/TEST
TRAIN_DIR = '/home/edgardaniel/Downloads/train'
VAL_DIR = '/home/edgardaniel/Downloads/validation'
TEST_DIR = '/home/edgardaniel/Downloads/test'


#VARIABLE WITH NUMBER OF CLASSES 
TOT_CLASSES = 2

#VARIBLES THAT OLD'S THE SIZE OF IMAGE
HEIGHT_IMGS = 224
WIDTH_IMGS = 224

#VARIABLE THAT EXPECIFY THE IMAGE OF COLOR: 1-GRAYSCALE / 3-COLOR
DEPTH_IMGS = 3

#VARIABLE IMG SIZE
IMG_SIZE = 224

#VARIABLE BATCH SIZE
BATCH_SIZE = 12

#VARIABLE EPOCH'S NUMBER
EPOCHS = 5

##!!!!!VARIABLES FOR DISCOVER FOR THE REPORT
learning_rate = 1e-3
lr_decay = 0.9
decay_epochs = 10

#PLACEHOLDER'S FOR IMAGE
x_input_shape = (None, HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS)
x_inputs = tf.placeholder(tf.float32, shape=x_input_shape)

#PLACEHOLDER'S FOR LABLES
y_targets = tf.placeholder(tf.int32, shape=None)
y_model = tf.placeholder(tf.float32, shape=(None, TOT_CLASSES))

##!!!!VERIFY PLACEHOLDERS
generation_num = tf.Variable(0, trainable=False)

#FUNCTION FOR CALCULATE LOST
def loss_gtsrb(logits, targets):
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name='xentropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	return cross_entropy_mean

#FUNCTION FOR TRAIN 
def train_step(loss_value, gn):

    model_learning_rate = tf.train.exponential_decay(learning_rate, gn, decay_epochs, lr_decay, staircase=True)

    # CREATE OPTIMIZER
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    #INITIALIZE TRAIN STEP
    train_step = my_optimizer.minimize(loss_value)

    return train_step

#FUNCTION FOR CALCULATE ACCURACY FOR BATCH
def accuracy_of_batch(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))

    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy

#FUNCTION THAT CREATE NETWORK
with tf.variable_scope('model_definition') as scope:

	model_outputs, end_points = vgg_16(x_inputs,TOT_CLASSES)
	#print(end_points)
	scope.reuse_variables()

#LOSS OPERATION
loss = loss_gtsrb(model_outputs, y_targets)
#TRAIN OPERATION
train_op = train_step(loss, generation_num)
#PREDICTIONS OPERATION
predictions_batch = tf.cast(tf.argmax(model_outputs,1), tf.int32)
#ACCURACY OPERATION
accuracy = accuracy_of_batch(y_model, y_targets)


#CREATE A LIST OF TUPLES WITH THE FIRST POSITION THE PATH TO THE IMAGE AND THE SECOND POSITION THE LABEL
all_images_train = list_all_data(TRAIN_DIR)
all_images_validation = list_all_data(VAL_DIR)
all_images_test = list_all_data(TEST_DIR)


#SHUFFLE ALL PATH IMAGES FOR RANDOMIZES THE DATA
shuffle(all_images_train)
shuffle(all_images_validation)
shuffle(all_images_test)

#CREATE PLT
plt.ion()

#CREATE OBJECT THAT HANDLE THE SAVER MODE OF THE MODEL TRAINED
saver = tf.train.Saver()

#IMPLEMENTATION OF FUNCTION THAT READS THE BATCHES AND CALCULATE THE ACCURACY
def evaluate(VAL_PATH,batch_size):

	#VARIABLES FOR CONTROL
	num_examples = int (len(VAL_PATH)/batch_size)
	total_accuracy = 0
	total_loss = 0
	cont_images = 0
	results = []
	labelsa = []
	
	#RESTORE THE ACTUAL SESSION
	sess = tf.get_default_session()

	for j in range(num_examples):

		#LOAD BATCH OF DATA
		val,cont = create_data(VAL_PATH,batch_size,cont_images)
		
		#VARIABLE THAT CONTROLS THE PROGRESSION OF IMAGES READ
		cont_images = cont

		#SEPARATE LABELS FROM FEATURES
		val_x = np.array([i[0] for i in val])

		#NORMALIZED THE DATA
		val_x = val_x / 255
		
		#LABELS
		val_y = np.array([i[1] for i in val])


		#INITIALIZE VARIABLE FOR CONTROL OF TIME FOR BENCHMARKS
		#timeN = time()

		#CALCULATE THE LOSS AND THE RESULTS OF THE NETWORK FOR BATCH
		t_loss, temp_validation_y = sess.run([loss, model_outputs],feed_dict={x_inputs: val_x, y_targets: val_y})

		#SAVE RESULTS FOR CALCULATE AUC
		#CREATES THE VARIABLE THAT STORES THE PROBABILIST OF THE NETWORK
		for i in range(batch_size):
			results.append(temp_validation_y[i])

		#CREATES THE VARIABLE THAT STORES THE LABELS
		#label = np.array([[0,1]]*batch_size)
		for i in range(batch_size):
			labelsa.append(val_y[i])

		#temp_validation_y = sess.run(model_outputs, feed_dict={x_inputs: val_x})
		#print(temp_validation_y)

		#CALCULARE THE ACCURACY FOR BATCH
		t_acc = sess.run(accuracy,feed_dict={y_model: temp_validation_y, y_targets: val_y})


		#FUNCTIONS TO CALCULATE TIME TO BENCHMARKS
		#timeF = time()
		#timeR = timeF - timeN

		#f = open("GENDER_RECONIGTION.txt", "a")
		#f.write(str(timeR*1000) + "\n")
		#f.close()
       
		#accuracy = sess.run(accuracy_operation, feed_dict={x: val_x, y: temp_validation_y})

		#VARIABLES THAT SAVE THE VALUES OF ACCURACY AND LOST FOR FINAL GRAPHICS
		total_accuracy += t_acc
		total_loss += t_loss


	return (total_accuracy / num_examples),(total_loss / num_examples),results,labelsa

#CREATE A SESSION TO RUN THE MODEL
#FUNCITONS FOR RTX GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#CREATE SESSION
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())

	#CALCULATE THE NUMBER OF ITERATIONS DIVIDE THE NUMBER OF ALL IMAGES ON
	#TRAIN DATASET FOR THE NUMBER OF IMAGES BATCH SIZE
	num_iteration = int (len(all_images_train) / BATCH_SIZE)
	loss_epoch = []
	train_epoch = []

	#INITIALIZE VARIABLES THAT SAVE THE DATA TO GRAPHICS OF PERFORMANCE
	accuracy_train_data = []
	loss_train_data = []
	accuracy_validation_data= []
	loss_validation_data = []

	#INITIALIZE VARIABLES THAT SAVE THE DATA TO AUC GRAPHIC
	labelsOF = []
	resultsOF = []

	print("Training...")
	for i in range(EPOCHS):
		value1 = 0
		value2 = 0
		num_batch = 0

		for j in range(num_iteration):

			#LOAD THE IMAGES OF DATA WITH THE SIZE OF BATCH_SIZE AND RETURN THE IMAGES AND THE COUNT NUMBER FOR LOAD THE
			#NEXT IMAGES FOR DIFERENTE BATCHES
			train,res1= create_data(all_images_train,BATCH_SIZE,value1)
			value1 = res1
			num_batch = num_batch +1

			#EXTRACT FEATURES
			X = np.array([i[0] for i in train])

			#NORMALIZED THE DATA
			X = X/255
			
			#EXTRACT LABELS
			Y = np.array([i[1] for i in train])
		
			#SESSION RUN
			sess.run(train_op, feed_dict={x_inputs: X, y_targets: Y})
			[t_loss, y_out] = sess.run([loss, model_outputs],feed_dict={x_inputs: X, y_targets: Y})
			t_acc = sess.run(accuracy, feed_dict={y_model: y_out, y_targets: Y})

			#SHOW INFORMATION
			#print("TRAINING STEP CONCLUDED numb_batch:{},batch_num:{}, loss:{}, accuracy:{}".format(num_batch,num_iteration,t_loss,t_acc))

			#APPEND THE INFORMATION ABOUT THE ACCURACY AND LOSS OF EACH BATCH
			loss_epoch.append(t_loss)
			train_epoch.append(t_acc)


		#CAlCULATE THE LOSS AND ACCURACY IN THE END OF EACH EPOCH
		loss_final = sum(loss_epoch) / len(loss_epoch)
		accuracy_final = sum(train_epoch) / len(train_epoch)
		validation_accuracy,validation_loss,resultO,labelsO = evaluate(all_images_validation,BATCH_SIZE)

		#SAVE ALL THE RESULTS AND LABELS OF EACH EPOCH TO THE AUC GRAPHIC
		resultsOF.append(resultO)
		labelsOF.append(labelsO)

		#LOG INFORMATION ABOUT EACH EPOCH
		print("EPOCH = {}" .format(i+1))
		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
		print("Validation Loss = {:.3f}".format(validation_loss))
		print("Accuracy = {}".format(accuracy_final))
		print("Loss = {}" .format(loss_final))
		print()

		#APPEND ALL DATA ABOUT THE ACCURACY AND LOSS OF EACH EPOCH
		loss_train_data.append(loss_final)
		accuracy_train_data.append(accuracy_final)
		accuracy_validation_data.append(validation_accuracy)
		loss_validation_data.append(validation_loss)

	#SAVE THE ACCURACY AND LOSS OF TRAIN AND VALITION TO CSV FILES
	numpy.savetxt("loss_train_data.csv", loss_train_data, delimiter=",")
	numpy.savetxt("accuracy_train_data.csv",accuracy_train_data,delimiter=",")
	numpy.savetxt("accuracy_validation_data.csv",accuracy_validation_data,delimiter=",")
	numpy.savetxt("loss_validation_data.csv",loss_validation_data,delimiter=",")

	#!!! CAUTION, THIS RESULTS AND LABELS STORES THE INFORMATION ABOUT ALL THE EPOCHS
	#MAKE THE LISTS OF LABELS AND RESULTS IN NUMPY ARRAYS
	resultsOF = np.asarray(resultsOF)
	labelsOF = np.asarray(labelsOF)

	#SAVE THE RESULTS AND LABELS TO CSV FILES
	numpy.savetxt("labelsOF.csv",labelsOF, delimiter=",")
	numpy.savetxt("resultsOF.csv",resultsOF, delimiter=",")
	
	#SAVE THE MODEL
	saver.save(sess, 'VGG16')
	print("Model saved")

"""
#FUNTION TO TEST THE SAVED MODEL WITH VALITION OR TEST IMAGES    
with tf.Session() as sess:
	
	#RESTORE NEW SESSION
	saver.restore(sess, tf.train.latest_checkpoint('/home/edgardaniel/Desktop/Linux_LAST_FILES/Desktop/NewModel'))
	
	#INITIALIZE VARIABLES FOR BENCHMARK TESTS
	#timeInicial = time()
	#timeFinal = time() - timeInicial

	#VARIABEL THAT CONTROLS THE NUMBER OF ITERATIONS
	#cont = 0
	
	#WHILE LOOP THAT EXECUTES SEVERAL TIMES THE EVALUATION OF THE IMAGES IN ONE MINUTE
	#while timeFinal <= 60:

		#EXECUTES THE FUNTION FOR EVALUATE THE NETWORK // INPUT : VARIABLE WITH ALL DATA IMAGES, THE SIZE OF BATCH
	test_accuracy,test_loss,resultsOF,labelsOF = evaluate(all_images_validation, BATCH_SIZE)

		#CALCULATES THE EXECUTION TIME
		#timeFinal = time() - timeInicial
		#cont = cont+1

	#WRITE TIME OF EXECUTION TO FILE "GENDER_RECONIGTION.txt"
	#f = f = open("GENDER_RECONIGTION.txt", "a")
	#f.write(str(cont) + '\n')
	#f.close()
	
	#TURN THE LIST OF RESULT AND LABELS IN NUMPY ARRAYS
	labelsOF = np.asarray(labelsOF)
	resultsOF = np.asarray(resultsOF)

	#print(resultsOF)
	#print(labelsOF)

	#SAVES THE RESULTS AND LABELS FROM THE VALIDATION DATASET IN FILES
	np.savetxt("labelsOF.csv", labelsOF, delimiter=",")
	np.savetxt("resultsOF.csv", resultsOF, delimiter=",")


	#PRINT THE RESULTS TO THE CLI
	#print("Número de Iterações", cont)

	#PRINT THE ACCURACY NUMBER FOR THE VALIDATION DATASET IN THE CLI
	print("Test Accuracy = {:.3f}".format(test_accuracy))
"""