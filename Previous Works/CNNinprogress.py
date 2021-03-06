from __future__ import print_function
from random import shuffle
import cv2
import math
import os
import numpy as np

def binarize(img):
  # Performs gaussian blurring with a kernel size of (5,5)
  blur = cv2.GaussianBlur(img,(5,5),0)
  # Performs Otsu thresholding (binarization) on the blurred image
  ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # switch black/white index (such that white=0), and set black to 1 instead of 256
  otsu[otsu==0] = 1
  otsu[otsu==255] = 0
  return otsu

#Reads all of the labelled images, and determined all different classes
def readLabelledData(maxFiles):
  lbls = []
  for i, file in enumerate(sorted(os.listdir('Labelled'))):
    if file.endswith('.pgm'):
      utf = file[:4]
      if utf not in lbls:
        lbls.append(utf)
      if i==maxFiles:
        break;

  numlbl = len(lbls) # number of classes in the data set
  labelled_data = []
  for i, file in enumerate(sorted(os.listdir('Labelled'))):
    print(i,  end="\r")
    if file.endswith('.pgm'):
      img = cv2.imread('Labelled/' + file, 0)
      bin_img = binarize(img) # for now binarize image here (must already be done, but not)
      #print(np.unique(bin_img))
      utf = file[:4]
      output = getDesiredOutput(numlbl, lbls.index(utf))
      # labelled_data.append((utf,img, output))
      labelled_data.append((utf,bin_img, output))
    if len(labelled_data)==maxFiles:
      break;
  return labelled_data, lbls

# Generate target output for an image
# Args: length of target vector, index in vector that must have value 1
# Returns vector of zeros of given length, with 1 at given position
def getDesiredOutput(length, onepos):
  out = np.zeros(shape=(1,length))
  out[0,onepos] = 1
  return out

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

# Function for the ReLu layer, implemented using the Softplus function
# Args: input matrix/vector, boolean for derivative
def reLuLayer(x, deriv = False):
  # set every negative value in x to zero
  # use softplus method to be able to find derivative, which is nonlin with deriv=true (needed for learning)
  if(deriv == True):
    return nonlin(x, True)

  # return np.log(1+np.exp(x))
  return x.clip(min=0)


# Function used for backpropagation through a pooling layer
# invert pooling by reconstructing original matrix, but only keeping values selected as local maxima
# Reshape to vector, and construct another vector with the errors at the right places
# Args: original input matrix to pool layer, matrix with indications of max places, vector of errors
# Returns reconstructed matrix as vector, and another vector of same size with errors
def invertPool(original, maxplaces, errors):
  only_maxes = np.multiply(original,maxplaces) # replace all irrelevant values (nonmax) with zero
  only_maxes_valuevec = only_maxes.reshape(1,sum(len(x) for x in only_maxes)) # reshape to vector

  # construct vector of size only_maxes_vec, with errors at max places
  maxplaces_vec = maxplaces.reshape(1,sum(len(x) for x in maxplaces))
  only_maxes_errorvec = np.zeros(shape=(1,len(only_maxes_valuevec)))
  error_indx = 0
  for indx in range(len(maxplaces_vec)):
    if maxplaces_vec[0, indx]==1: # indx is a local maxima: next error in errors belongs to indx
      only_maxes_errorvec[0, indx] = errors[0, error_indx]
      error_indx +=1
  return only_maxes_valuevec, only_maxes_errorvec
  

# Function for the max pooling layer. Args = dat-matrix, window size (square), stride size
# slide window over dat in steps of stride, take in each window the maximum value
# return matrix with max values, and a matrix of same size as dat with indications of max places
def maxPoolLayer(dat, window, stride):
  nrow, ncol = np.shape(dat)
  prow = nrow/stride + (1 if nrow%stride>0 else 0)
  pcol = ncol/stride + (1 if ncol%stride>0 else 0)
  out = np.zeros(shape=(prow, pcol))
  places = np.zeros(shape=(nrow, ncol)) # dat matrix of zeros, with 1 at places of max values
  for r in range(0,prow):
    for c in range(0,pcol):
      # make sure that at the end no indexes occur outside the shape of the matrix
      sub = dat[r*stride:min(r*stride+window, nrow), c*stride:min(c*stride+window, ncol)]
      out[r,c] = np.max(sub)
      max_r, max_c = np.unravel_index(sub.argmax(), np.shape(sub))
      places[(r*stride)+max_r, (c*stride)+max_c] = 1 # put 1 at place of maximal value
  return out, places

# Performs the actual convolution of dat and mask (assumption = dat and mask have same shape)
def convolution(dat, mask):
  return np.sum(np.multiply(dat, np.flipud(mask))) 
  # return np.mean(np.multiply(dat, np.flipud(mask)))

# Main function of a convolution layer: slides mask over dat, at each place 
# doing a convolution with the part of dat under the mask. ("same" conv)
def convLayerSame(dat, mask):
  nrow, ncol = np.shape(dat)
  rMask, cMask = np.shape(mask)
  out = np.zeros(shape=(nrow, ncol))
  for r in range(0,nrow):
    for c in range(0,ncol):
      sub = np.zeros(shape=(rMask, cMask)) 
      sub[0:min(nrow-r, rMask),0:min(ncol-c, cMask)] = dat[r:min(nrow, r+rMask), c:min(ncol, c+cMask)]
      out[r,c] = convolution(sub, mask)

  return out
  
# Main function of a convolution layer: slides mask over dat, at each place 
# doing a convolution with the part of dat under the mask. ("valid" conv)
def convLayerValid(dat, mask):
  nrow, ncol = np.shape(dat)
  rMask, cMask = np.shape(mask)
  convRow = nrow - (rMask-1)
  convCol = ncol - (cMask-1)
  out = np.zeros(shape=(convRow, convCol))
  for r in range(0,convRow):
    for c in range(0,convCol):
      sub = np.zeros(shape=(rMask, cMask)) 
      sub = dat[r:r+rMask, c:c+cMask]
      out[r,c] = convolution(sub, mask)

  return out

def LoadUTF():
    print ('Reading UTFs: ')
    readMe = open('checkpoints/Allclass_UTF.txt', 'r').readlines()

    # # Creates a list containing 5 lists, each of 8 items, all set to 0
    # w, h = 2, len(readMe)
    # document = [[0 for x in range(w)] for y in range(h)]

    utfs=[]
    for i in range(0, len(readMe)):
       temp=readMe[i].strip()
       utfs.append(temp)
       #print(utfs[i])
    print ('UTFs are Imported')
    print ('Length of UTFs: '+str(len(utfs)))
    print ('example UTF:', utfs[100])
    print('example UTF:', utfs[200])
    print('example UTF:', utfs[300])
    return utfs
  
# Main function of a convolution layer: slides mask over dat, at each place 
# doing a convolution with the part of dat under the mask. ("full" conv)
def convLayerFull(dat, mask):
  nrow, ncol = np.shape(dat)
  rMask, cMask = np.shape(mask)
  convRow = nrow + (rMask-1)*2
  convCol = ncol + (cMask-1)*2
  padded = np.zeros(shape=(convRow, convCol))
  padded[rMask-1:convRow - (rMask-1), cMask-1:convCol - (cMask-1)] = dat
  return convLayerValid(padded, mask)

# Height and width of an input image
imgWidth = imgHeight = 128
# initialize random generator
np.random.seed(1)


# Randomly initialized masks, currently not updated yet (hence bad classifications?):
mask = np.random.random((2, 2)) - 1
mask3 = np.random.random((3, 3)) - 1

# Masks with predetermined values:
maskline = np.matrix([[1,-1,-1],
           [-1,1,-1],
           [-1,-1,1]])
           
maskFlip = np.matrix([[0,1,2],
           [3,4,5],
           [6,7,8]])
mask10 = np.matrix([
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [1,1,1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1,1,1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
# read 1000 images from data set, and get list of all classes (utf codes)
labelled_data, allclasses = readLabelledData(1000)
print("Number of classes: "+str(len(allclasses)))
# shuffle images, such that classes are more evenly distributed across data set
shuffle(labelled_data)

# list of parameters
inputlayer = 1600  # number of input nodes for fully connected part (TODO make dynamic using window size and img size)
numTestData = 100  # number of images to test on
numTrainData = 300 # number of images to train on
maskUsed = mask10   # mask that is used in the convolution
windowsize = 3     # window size of the pooling layer in the CNN
stridesize = 3     # stride size of the pooling layer in the CNN

# randomly initialize our weights with mean 0
# weights from input layer ("output of conv. layer" nodes) to hidden layer (4 nodes) fully connected
syn0 = 2 * np.random.random((inputlayer, 80)) - 1
# biases
syn0_B = 2 * np.random.random((1, 80)) - 1

# weights from hidden layer (4 nodes) to output layer ("number of classes" nodes)
syn1 = 2 * np.random.random((80, len(allclasses))) - 1
# biases
syn1_B = 2 * np.random.random((1, len(allclasses))) - 1


for j in range(numTrainData):

    x = labelled_data[j]
    im = x[1] # input data (image) 
    y = x[2]  # desired (target) output

    conv = convLayerValid(im, maskUsed)
    reLu = reLuLayer(conv)
    pool, places = maxPoolLayer(reLu, windowsize, stridesize)
    #print(places)
    l0 = pool.reshape(1,sum(len(x) for x in pool))
    #print(np.shape(l0)) #TODO use result to make "inputLayer" var dynamic
    l1 = nonlin(np.dot(l0, syn0) + syn0_B)
    l2 = nonlin(np.dot(l1, syn1) + syn1_B)

    l2_error = y - l2

    if (j % 1) == 0:
      tar_class = np.where(y==1)[1]
      print (str(tar_class) + " Error " + str(j) + " :" + str(np.mean(np.abs(l2_error))) + ", certainty: "+str(np.max(l2)) + ", correct: "+ str(np.argmax(l2)==np.argmax(y)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)
    
    # get error of pool layer
    pool_error = l1_delta.dot(syn0.T)

    # map maximal places to input of pooling layer and to pool_error. returns vector representations
    pool_input_vec, pool_input_errorvec = invertPool(reLu, places, pool_error)

    # pool does not contribute on error: pass on to ReLu layer
    relu_delta = pool_input_errorvec * reLuLayer(pool_input_vec, deriv=True)
    
    # no need for dot product: connections 1-1 instead of fully connected
    # reshape to original format
    sqrt_size = int(math.sqrt(relu_delta.size))
    # print(sqrt_size)
    conv_error = relu_delta.reshape(sqrt_size, sqrt_size)
    # conv_delta = ...

    syn1 += l1.T.dot(l2_delta)
    syn1_B += l2_delta
    syn0 += l0.T.dot(l1_delta)
    syn0_B += l1_delta
    # usedMask += ....

totCorrect = 0
tot = 0
for j in range(numTrainData,numTestData+numTrainData):
  x = labelled_data[j]
  tot = tot + 1
  im = x[1]
  y = x[2]
  conv =  convLayerValid(im, maskUsed)
  reLu = reLuLayer(conv)
  pool, places = maxPoolLayer(reLu, windowsize, stridesize)
  l0 = pool.reshape(1,sum(len(x) for x in pool))
  l1 = nonlin(np.dot(l0, syn0))
  l2 = nonlin(np.dot(l1, syn1))
  if(np.argmax(l2)==np.argmax(y)):
    totCorrect = totCorrect + 1
  print("Percentage correct: "+str(float(totCorrect)/float(tot))+" ("+str(totCorrect)+"/"+str(tot)+"), certainty: "+str(np.max(l2)))
