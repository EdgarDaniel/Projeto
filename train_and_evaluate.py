#coding=utf-8

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
# roc curve and auc score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#IMPORT IMPORTANT FUNCTIONS TO RTX GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#IMPORT MODELS DEFINITON FROM TF-SLIM lIGHTWEIGHT HIGH-LEVEL API OF TENSORFLOW
import tensorflow.contrib.slim as slim
#IMPORT VGG16 MODEL
from vgg import vgg_16

from time import time

import auc_roc as graphic

from itertools import chain

##!!!!REMOVE BEFORE GIT
##from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope


#TEST_DIR = '/home/edgardaniel/Desktop/Linux_LAST_FILES/Desktop/Teste'

#VARIABLES WITH PATH FOR IMAGES TRAIN/VALIDATION/TEST
TRAIN_DIR = '/home/edgardaniel/Downloads/train'
VAL_DIR = '/home/edgardaniel/Downloads/validation'
#TEST_DIR = '/home/edgardaniel/Downloads/test'

# THE PATH FOR ALL THE TRAIN,VAL AND TEST DATA TO LAB COMPUTER

#TRAIN_DIR = '/home/jcneves/Desktop/NEW CODE/ModelF/train'
#VAL_DIR = '/home/jcneves/Desktop/NEW CODE/ModelF/validation'
#TEST_DIR = '/home/jcneves/Desktop/NEW CODE/ModelF/test'

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

##!!!!!VARIABLES FOR DISCOVER 
DROPOUT_PROB = 1
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
dropout_prob = tf.placeholder(tf.float32)
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
	print(end_points)
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
#all_images_test = list_all_data(TEST_DIR)


#SHUFFLE ALL PATH IMAGES FOR RANDOMIZES THE DATA
shuffle(all_images_train)
shuffle(all_images_validation)
#shuffle(all_images_test)

#CREATE PLT
plt.ion()

#CREATE OBJECT THAT HANDLE THE SAVER MODE OF THE MODEL TRAINED
saver = tf.train.Saver()

#IMPLEMENTATION OF FUNCTION THAT READS THE BATCHES AND CALCULATE THE ACCURACY
def evaluate(VAL_PATH,batch_size):

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

		#timeN = time()

		#CALCULATE THE ACCURACY FOR BATCH
		t_loss, temp_validation_y = sess.run([loss, model_outputs],feed_dict={x_inputs: val_x, y_targets: val_y, dropout_prob: DROPOUT_PROB})


		results.append(temp_validation_y)
		label = np.array([0,1]*batch_size)
		labelsa.append(label)

		#temp_validation_y = sess.run(model_outputs, feed_dict={x_inputs: val_x, dropout_prob: DROPOUT_PROB})
		#print(temp_validation_y)
		t_acc = sess.run(accuracy,feed_dict={y_model: temp_validation_y, y_targets: val_y})


		#timeF = time()
		#timeR = timeF - timeN

		#f = open("GENDER_RECONIGTION.txt", "a")
		#f.write(str(timeR*1000) + "\n")
		#f.close()
       
		#accuracy = sess.run(accuracy_operation, feed_dict={x: val_x, y: val_y})
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
    	
	num_iteration = int (len(all_images_train) / BATCH_SIZE)
	loss_epoch = []
	train_epoch = []

	accuracy_train_data = []
	loss_train_data = []
	accuracy_validation_data= []
	loss_validation_data = []

	labelsOF = []
	resultsOF = []

	print("Training...")
	for i in range(EPOCHS):
		value1 = 0
		value2 = 0
		num_batch = 0

		for j in range(num_iteration):

			train,res1= create_data(all_images_train,BATCH_SIZE,value1)
			value1 = res1
			num_batch = num_batch +1

			#FEATURES
			X = np.array([i[0] for i in train])

			#NORMALIZED THE DATA
			X = X/255
			
			#LABELS
			Y = np.array([i[1] for i in train])
		
			#SESSION RUN
			sess.run(train_op, feed_dict={x_inputs: X, y_targets: Y, dropout_prob: DROPOUT_PROB})
			[t_loss, y_out] = sess.run([loss, model_outputs],feed_dict={x_inputs: X, y_targets: Y, dropout_prob: DROPOUT_PROB})
			t_acc = sess.run(accuracy, feed_dict={y_model: y_out, y_targets: Y})

			#SHOW INFORMATION
			#print("TRAINING STEP CONCLUDED numb_batch:{},batch_num:{}, loss:{}, accuracy:{}".format(num_batch,num_iteration,t_loss,t_acc))
		
			loss_epoch.append(t_loss)
			train_epoch.append(t_acc)

			#print(loss_epoch)
			#print(train_epoch)

		#CAlCULATE THE LOSS AND ACCURACY IN THE END OF EACH EPOCH
		loss_final = sum(loss_epoch) / len(loss_epoch)
		accuracy_final = sum(train_epoch) / len(train_epoch)
		validation_accuracy,validation_loss,resultO,labelsO = evaluate(all_images_validation,BATCH_SIZE)

		resultsOF.append(resultO)
		labelsOF.append(labelsO)

		#LOG INFORMATION ABOUT EACH BATCH
		print("EPOCH = {}" .format(i+1))
		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
		print("Validation Loss = {:.3f}".format(validation_loss))
		print("Accuracy = {}".format(accuracy_final))
		print("Loss = {}" .format(loss_final))
		print()
		
		loss_train_data.append(loss_final)
		accuracy_train_data.append(accuracy_final)
		accuracy_validation_data.append(validation_accuracy)
		loss_validation_data.append(validation_loss)


	test4 = np.array(labelsOF).flatten()
	test5 = np.array(resultsOF).flatten()
	print(test4)
	print(test5)
	auc = roc_auc_score(test4,test5)
	print(auc)

	fpr, tpr, thresholds = roc_curve(test4, test5)

	graphic.plot_roc_curve(fpr,tpr)

	eval_indices = range(1, EPOCHS+1)
	#print(eval_indices)
	#print(loss_data)
	#print(accuracy_data)

	plt.clf()
	plt.subplot(211)
	plt.plot(eval_indices, accuracy_train_data, 'k--', label='TREINO')
	plt.plot(eval_indices, accuracy_validation_data, 'g-x', label='VALIDAÇÃO')
	plt.legend(loc='upper right')
	plt.xlabel('Épocas')
	plt.ylabel('ACERTO')
	plt.grid(which='major', axis='both')


	plt.subplot(212)
	#plt.plot(eval_indices, train, 'g-x', label='Train Set Accuracy')
	plt.plot(eval_indices,loss_train_data, 'r-x', label='TREINO')
	#plt.plot(eval_indices, np.ones(len(eval_indices))/TOT_CLASSES, 'k--')
	plt.plot(eval_indices,loss_validation_data,'k--',label='VALIDAÇÃO')
	plt.legend(loc="upper right")
	plt.xlabel('Épocas')
	plt.ylabel('ERRO')
	plt.ylim(0, 1)
	plt.grid(which='both', axis='y')

	plt.subplots_adjust(left=0.2, wspace=0.2, hspace=0.3)

	plt.show()
	plt.pause(0.01)

	
	#SAVE THE MODEL
	saver.save(sess, 'VGG16')
	print("Model saved")
	plt.savefig('Learning.png')

"""
#FUNTION TO TEST THE SAVED MODEL WITH TEST IMAGES    
with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint('/home/edgardaniel/Desktop/Linux_LAST_FILES/Desktop/NewModel'))
	timeInicial = time()
	timeFinal = time() - timeInicial
	cont = 0
	while timeFinal <= 60:
		test_accuracy = evaluate(all_images_test, 30)
		#print("Teste Acerto:", test_accuracy)
		timeFinal = time() - timeInicial
		cont = cont+1
	f = f = open("GENDER_RECONIGTION.txt", "a")
	f.write(str(cont) + '\n')
	f.close()
	print("Número de Iterações", cont)
	print("Test Accuracy = {:.3f}".format(test_accuracy))
"""
