
import cv2
import os
import math
import csv
import pixelDensity
from random import shuffle
import numpy as np
from pathlib import Path


#Reads all of the images and xml files
def readData(path):
	paired_data = []
	num_files = len(os.listdir(path))
	print('reading new data')
	for i, file in enumerate(os.listdir(path)):
		print(round((i/num_files)*100,2),  end="\r")
		if file.endswith('.pgm'):
			img = cv2.imread(path+'/' + file, 0)
			if Path(path+'/'+ file[:-4] + '.xml').is_file():
				label = open(path+'/'+ file[:-4] + '.xml', 'r').read()
				paired_data.append((img, label))
	return paired_data


#Reads all of the labelled images
def readLabelledData(path):
	labelled_data = []
	num_files = len(os.listdir(path))
	print('Reading labelled data:')
	for i, file in enumerate(os.listdir(path)):
		print(round((i/num_files)*100,2),  end="\r")
		if file.endswith('.pgm'):
			img = cv2.imread(path+'/' + file, 0)
			utf = file[:4]
			labelled_data.append((utf,img))
	return labelled_data


#Parses the information from an xml file (String) to a dictionary
#with the useful information (x, y, w, h and utf)
def getLabelInfo(label):
	label_info = {}
	label_info['utf'] = None
	for i, c in enumerate(label):
		if(c == '-' and label[i+1]=='x' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['x'] = value
		if(c == '-' and label[i+1]=='y' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['y'] = value
		if(c == '-' and label[i+1]=='w' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['w'] = value
		if(c == '-' and label[i+1]=='h' and label[i+2]=='='):
			value =''
			for a in label[i+3:]:
				if(a.isdigit()):
					value += a
				else:
					break
			label_info['h'] = value
		if(c == '<' and label[i+1]=='u' and label[i+2]=='t'and label[i+3]=='f' and label[i+4]=='>' and label[i+5] == ' '):
			value =''
			for a in label[i+6:]:
				if(a != ' '):
					value += a
				else:
					break
			label_info['utf'] = value
	return label_info

def showImage(img):
	cv2.imshow('image',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()



if __name__ == "__main__":
	#The training data (with labels)
	labelled = readLabelledData('labelled')
	new_data = readData('project/test')

	for i, pair in enumerate(new_data):
		image =  pair[0]
		cv2.imshow('totalimage',image)
		raw_xml = pair[1]
		#There can be multiple labelled characters in one xml/image
		split_xml = raw_xml.splitlines()
		#For each labelled character
		tagged_image_list = []
		for x, xml in enumerate(split_xml):
			tagged_image = image
			xml_info = getLabelInfo(xml)
			label = xml_info['utf']
			for (old_xml, old_image) in labelled:
				if old_xml == label:
					tagged_image_list.append(old_image)
					break

		#New combined image		
		vis = np.zeros((128, 128*len(tagged_image_list)), np.uint8)
		for i, iimage in enumerate(tagged_image_list):
			vis[:128, 128*i:128*i+128] = iimage


		cv2.imshow('taggedImage',vis)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
