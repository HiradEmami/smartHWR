import cv2
import os
import pixelDensity
import numpy as np


def binarize(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return otsu

#Reads all of the images and xml files
def readData():
	paired_data = []
	img = cv2.imread('Train/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm',0)
	for i, file in enumerate(os.listdir('Train')):
		print(i,  end="\r")
		if file.endswith('.pgm'):
			img = cv2.imread('Train/' + file, 0)
		if file.endswith('.xml'):
			label = open('Train/'+ file, 'r').read()
			paired_data.append((img, label))
	return paired_data

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

#Cuts an image based on x, y, w and h values
def cutImage(image, label_info):
	x = int(label_info['x']);
	y = int(label_info['y']);
	w = int(label_info['w']);
	h = int(label_info['h']);
	cropped_image = image[y:y+h, x:x+w]
	#cv2.imshow("cropped", cropped_image)
	#cv2.waitKey(0)
	return cropped_image


#Extracts the annotated characters from the images and pairs them with
#their labels to create labelled_data
def extractAnnotatedSegments():
	print('Reading data:')
	paired_data = readData()
	data_size = len(paired_data)
	labelled_data = []
	fail_counter = 0
	print('All data read\nExtracting segments:')
	for i, pair in enumerate(paired_data):
		print(repr(round((i/len(paired_data)*100),2)) + '%', end="\r")
		image =  pair[0]
		raw_xml = pair[1]
		#There can be multiple labelled characters in one xml/image
		split_xml = raw_xml.splitlines()
		#For each labelled character
		for xml in split_xml:
			xml_info = getLabelInfo(xml)
			#If there is no utf code, skip this xml line
			if(xml_info['utf'] is None):
				fail_counter+=1
				continue
			else:
				label = xml_info['utf']
				#Cut the image using the xml info
				cut_image = cutImage(image, xml_info)
				#Binarize this cut image and crop it such that it fits in a 128x128 square
				cut_image = pixelDensity.cropSquare(pixelDensity.binarize(cut_image))
				#Store the image with its label in labelled_data
				labelled_data.append([cut_image, label])
				
				cv2.imwrite('Labelled/'+label+'_'+str(i) +'.pgm',cut_image)
	print('All segments extracted\nNumber of labels without utf:')
	print(fail_counter)

extractAnnotatedSegments()