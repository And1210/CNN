import numpy as np
import mnistdb.io as mio

correctNumbers = mio.load() # array with the correct number
data = mio.load(scaled=True) #the data of the images
correctNumbersIndex = mio.load(one_hot=True) #array with a 1 at the index position of the correct number
for i in range(60000):
    trainingImagesArray = data.trainX[i] #the 60 000 training images
    trainingImagesArray = np.array(trainingImagesArray,ndmin=2) #scaling the array so it has 2 dimensions
    trainingIndexArray = correctNumbersIndex.trainY[i] # the index position of the correct number for the 60 000 training images
for i in range(10000):
    testingImagesArray = data.testX[i] #the 10 000 testing images
    testingImagesArray = np.array(testingImagesArray,ndmin=2) #scaling the array so it has 2 dimensions
    testingIndexArray = correctNumbersIndex.testY[i] #the index position of the correct number for the 10 000 testing images


class Filter:
    #filterSize - the size of the filter, the filter is square so it is one side length
    #***filter size must be odd***
    def __init__(self, filterSize):
    	if(filterSize%2 == 0):
    		raise Exception("filterSize [the only parameter to Filter.__init__()] must be odd")
    	else:
        	self.values = np.random.uniform(low=-1, high=1, size=(filterSize, filterSize))
        	self.size = filterSize
        
    #this function convolutes the filter over the given image
    #a is the filter b is the image under convolution
    def convolute(self, b):

    	#the "kernel" of convolution
       	a = self.values

       	#each sum is multiplied by this number to "normalize" it
       	normalizationConstant = 1/((self.size)**2)

       	#this will be the output of the function, it's the array where all the convolution sums in the below for loop will end up 
       	#instead of initializing an array of the same dimensions as b, I just copied all the indexes over. quick and dirty i know
       	output = b[:]

       	#the use of this variable will become clear later, basically it makes the convolution sum to start at one corner of the kernel
       	shift = int((self.size-1)/2)

       	#x and y are the indices of convolution, ie, they're the indices of the cell that the current sum being calculated will end up
       	for x in range(0, len(b)):
       		#it is necessary that b is at least a rectangular matrix so that b[0] is the same length as b[q] for any index q
       		for y in range(0, len(b[0])):
       			output[x][y] = 0

       			#these for loops iterate over the kernel and the corresponding indices in the image
       			for i in range(0, self.size):
       				for j in range(0, self.size):
       					try:
       						#see how the shift variable makes the convolution sum to start at one corner of the kernel
       						output[x][y] += a[x+shift-i][x+shift-j]*b[y+shift-i][x+shift-j]
       					except IndexError:
       						#if the kernel overlaps the edge of b, consider the value of b[y+shift-i][x+shift-j] to be zero
       						output[x][y] += 0
       			#"normalizes" the output
       			output[x][y] *= normalizationConstant

       	return output
        
#Convolution layer
class ConvLayer:
    #neuronNum - the number of neurons on the layer
    #filterSize - the size of the filter, the filter is square so it is one side length
    def __init__(self, neuronNum, filterSize):
        self.filters = [Filter(filterSize) for x in range(neuronNum)]
        
    #imgs - a list of input images to the layer
    def compute(self, imgs):
        output = []
        if (len(imgs) != len(self.filters)): #refactor to accept more images than neurons as well
            mult = int(len(self.filters)/len(imgs))
            images = []
            counter = 0
            for i in range(len(imgs)):
                for j in range(mult):
                    images.push(imgs[counter])
                counter = counter + 1
        else:
            images = imgs
        for i in range(len(self.filters)): #looping through every neuron
            #Convolute old image with filter
            newImg = self.filters[i].convolute(images[i])
            #Apply activation function (ReLU)
            
            #Pooling
            
            output.push(newImg)
        return output
        
class CNN:
    #inputSize - tuple of the width and height of input image
    #hiddenLayers - list of number of neurons on each hidden layer
    #outputSize - the number of outputs (10 for mnist)
    #lr - the learning rate
    def __init__(self, inputSize, hiddenLayers, outputSize, lr):
        self.lr = lr
        self.layers = []
        for neurons in hiddenLayers:
            self.layers.push(ConvLayer(neurons, 3))
    def setLearningRate(self, newRate):
        self.learningRate = newRate
    def getLearningRate(self):
        return self.learningRate
        
    def feedForward(self, inputData):
        pass
    def train(self, inputData, expectedOutput):
        pass
