import numpy as np
import mnistdb.io as mio
import skimage.measure

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
    def __init__(self, filterSize):
        self.values = np.random.uniform(low=-1, high=1, size=(filterSize, filterSize))
        self.w = filterSize
        
    #an image represented as a 2D numpy array
    def convolute(self, image):
        #this is where the filter is convoluted over the entire input image
        #should produce a new image with features highlighted

    #relu activation function
    def relu(self, image):
        for r in numpy.arange(0,image[0]):  
            for c in numpy.arange(0, image[1]):
                if image[r, c] > 0:
                    image[r, c] = image[r, c]
                else:
                    image[r, c] = 0
        return image
    #max pooling function
    def maxPool(self, image, size):
        skimage.measure.block_reduce(image, size, np.max)
        return image

        pass
        
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
            newImg = relu(newImg)
            #Pooling
            newImg = maxPool(newImg)
            output.push(newImg)
        return output

#Each layer of the Neural Network has properties, those are stored here    
class Layer:
    #Layer constructor
    #inputSize ----- the number of inputs (or neurons) from the previous layer
    #selfSize ----- the number of neurons on this layer
    def __init__(self, inputSize, selfSize):
        self.weights = 2 * np.random.rand(selfSize, inputSize) - 1  #giving random weights to start
        self.bias = 2 * np.random.rand(selfSize, 1) - 1 #giving random bias to start
        
    #Feeds an input vector through the layer to produce an output
    #inputData ----- a 1D array or list of input values with same length as inputSize
    def compute(self, inputData):
        out = self.weights.dot(inputData)  #feeding the given inputs through the current layerout)
        out = out + self.bias
        out = 1 / (1 + np.exp(-out))    #sigmoid activation function
        return out
        
class CNN:
    #inputSize - tuple of the width and height of input image
    #hiddenLayers - list of number of neurons on each hidden layer
    #outputSize - the number of outputs (10 for mnist)
    #lr - the learning rate
    def __init__(self, inputSize, hiddenLayers, outputSize, lr=0.05):
        self.lr = lr
        self.layers = []
        self.layerOut = [None for i in range(hiddenLayers + 1)]
        self.flattenOut = None
        for neurons in hiddenLayers:    #Setting up the convolutional layers
            self.layers.append(ConvLayer(neurons, 3))
            
#        flattenNum = int(inputSize[0]*inputSize[1]/(pow(4, len(hiddenLayers))))
        testInput = np.zeros(inputSize) #getting the number of inputs for the flattened layer
        for l in self.layers:
            testInput = l.compute(testInput)
        flattenNum = testInput.shape[0] * testInput.shape[1]
        self.finalLayer = Layer(flattenNum, outputSize)
        
    def setLearningRate(self, newRate):
        self.learningRate = newRate
    def getLearningRate(self):
        return self.learningRate
        
    def feedForward(self, inputData):
        curData = inputData
        index = 0
        for l in self.layers:
            curData = l.compute(curData)
            self.layerOut[index] = curData
            index = index + 1
        curData = curData.flatten()
        self.flattenOut = curData
        probabilities = self.finalLayer.compute(curData)
        return probabilities
    
    def train(self, inputData, expectedOutput):
        curOutput = self.feedForward(inputData)
        cost = expectedOutput - curOutput
        layerOutputs = [inputData] + self.layerOut
        
        gradient = self.lr * cost * curOutput * (1 - curOutput)
#        for i in range(len(layerOutputs)):
            

a = np.zeros((100, 100))
nn = CNN(a.shape, [2, 4], 2)