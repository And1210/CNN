import numpy as np

class Filter:
    #filterSize - the size of the filter, the filter is square so it is one side length
    def __init__(self, filterSize):
        self.values = np.random.uniform(low=-1, high=1, size=(filterSize, filterSize))
        self.w = filterSize
        
    #an image represented as a 2D numpy array
    def convolute(self, image):
        #this is where the filter is convoluted over the entire input image
        #should produce a new image with features highlighted
        pass
    
    #a convoluted image has the relu function applied to it
    def relu(self, image):
        for r in numpy.arange(0,image[0]):  
            for c in numpy.arange(0, image[1]):
                if image[r, c] > 0:
                    image[r, c] = image[r, c]
                else:
                    image[r, c] = 0
        return image
    def maxPool(self, image, size, stride):
        
        
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
            #pushing th
            output.push(newImg)
        return output

    def backpropagation
        
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
        layerOuput = compute(inputData)
        return layerOutput
        
    def train(self, inputData, expectedOutput):

        pass
