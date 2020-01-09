
#COPYRIGHT EDGAR DANIEL
#THIS SCRIPT CREATES THE DATA FOR THE NETWORK


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks.

IMG_SIZE = 224

# FUNCTION THAT CREATES A LIST OF TUPLES WITH THE FIRST POSITION THE PATH TO THE IMAGE AND THE SECOND POSITION THE LABEL
def list_all_data(path):
	#VARIABLE THAT WILL STORES THE TUPLES OF IMAGES AND LABELS
	all_images = []
	foldersPATH = os.listdir(path)
	for i in range(len(foldersPATH)):	
		#CREATES THE LABEL DEPENDING THE NUMBER OF FOLDER
		label = i
		folder_images = os.path.join(path,foldersPATH[i])
		images = os.listdir(folder_images)
		for j in range(len(images)):
			path_image = os.path.join(folder_images,images[j])
			all_images.append((path_image,label))
	return all_images

#FUNCTION THAT LOAD THE IMAGES FROM THE PATH WITH THE BATCHSIZE
#THIS FUNCTION USES ONE COUNTER FOR CONTROL THAT ALLOW THE PROGRESSION OF LOADING DIFERENT IMAGES
def create_data(all_data_path,batchsize,cont_images):
	all_data = []
	all_len = len(all_data_path)
	cont = 0
	
	for i in range(cont_images,all_len):
		path_image,label_image = all_data_path[i]
		imgF = cv2.imread(path_image, cv2.IMREAD_COLOR)
		#cv2.imshow("IMAGE",imgF)
		#cv2.waitKey()
		imgF = cv2.resize(imgF,(IMG_SIZE,IMG_SIZE))
		all_data.append([np.array(imgF), np.array(label_image)])
		cont = cont+ 1
		if(cont == batchsize):
			cont_images = cont_images + batchsize
			break
		
	return all_data, cont_images


