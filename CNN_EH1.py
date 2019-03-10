import numpy as np
import mnistdb.io as mio
from math import ceil as ceiling
from math import floor

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

#relu activation function
def relu(image):
    out = image
    out[out<0] = 0
    return out
#max pooling function. The max pooling operation takes the maximum value of a certain region of a given image. This function moves
#a kernel across the given image and max pools the image within the region defined by the kernel. The kernel is just a 2 index array
#that describes the length and width of the rectangular region being pooled. The kernel is moved accross the image and at each
#interval the value produced by the pooling operation is moved into a separate array. The point of this is to reduce the size of the
#size of the given image while still preserving most of the information it holds.
def maxPool(image, kernel):

    #width and height of the kernel
    width = kernel[0]
    height = kernel[1]

    #these define the amount the kernel moves in each direction in between intervals. A lot of maxPooling functions allow for arbitrary
    #movement so I've defined these as separate variables, however for our purposes a step size that is the same as the corresponding
    #dimension of the kernel is used since we neither want to miss any cells in the image or overlap any.
    hStep = kernel[0]
    vStep = kernel[1]

    #this array holds the results of the pooling operation at each interval, as the name suggests, it is what the function will output
    output = []
    #we have to take into account the possibility that the kernel will not evenly cover the image, ie, it's width and heigth might not
    #evenly divide the width and height of the given image respectively. If this occurs, at some iterations the kernel will hang over
    #the edge of the image. This is the reason for using ceiling() when initializing the output array below, we need it
    #to always round up when calculating its dimensions since the iterations when the kernel hangs over the side need a place to put the
    #result of the pooling operation.
    for i in range(0, ceiling(len(image)/hStep)):
        #[0] * ceiling(len(image[0])/vStep) is weird python notation for an array of zeroes of length ceiling(len(image[0])/vStep)
        output.append([0] * ceiling(len(image[0])/vStep))

    #this nested for loop iterates the kernel accross the image, maxPooling at each iteration and throwing the result in output
    #i and j are the index in output the result will be put in. Note that when we index image with i and j.
    for i in range(0, len(output)):
        for j in range(0, len(output[0])):
            #this array represents the values in image covered by the kernel in this iteration
            kernelCover = []
            #these nested for loops append all the values covered by the kernel to kernelCover
            #x and y can be thought of as the indices in the kernel
            for x in range(0, width):
                for y in range(0, height):
                    #this try except statement accounts for when the kernel overhangs the edge of the image. Note that if it does overhang,
                    #an IndexError will be thrown and nothing will be appended to kernelCover.
                    try:
                        kernelCover.append(image[i*hStep + x][j*vStep + y])
                    except IndexError:
                        pass

            output[i][j] = max(kernelCover)

    # returns output as a numpy array since other parts of the program use numpy arrays
    return np.array(output)

def backConvolute(img, kernal):
    output = np.zeros((img.shape[0] - kernal.shape[0] + 1,
                       img.shape[1] - kernal.shape[1] + 1))
    xWidth = img.shape[0] - kernal.shape[0]
    yWidth = img.shape[1] - kernal.shape[1]
    for i in range(xWidth + 1):
        for j in range(yWidth + 1):
            output[i][j] += np.sum(kernal*img[i:img.shape[0]-(xWidth-i),j:img.shape[1]-(yWidth-j)])
    return output

def inverseConvolute(img, kernal):
    output = np.zeros((kernal.shape[0] + floor(img.shape[0]/2) + 1,
                       kernal.shape[1] + floor(img.shape[1]/2) + 1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i:kernal.shape[0]+i,j:kernal.shape[1]+j] += kernal
            output[i+kernal.shape[0]-img.shape[0]:i+kernal.shape[0],j+kernal.shape[1]-img.shape[1]:j+kernal.shape[1]] += kernal[kernal.shape[0]-i-img.shape[0]:kernal.shape[0]-i,kernal.shape[1]-j-img.shape[1]:kernal.shape[1]-j]*img
    return output

class Filter:
    #filterSize - the size of the filter, the filter is square so it is one side length
    #***filter size must be odd***
    def __init__(self, filterSize):
        if(filterSize%2 == 0):
            raise Exception("filterSize [the only parameter to Filter.__init__()] must be odd")
            
        self.values = np.random.uniform(low=-1, high=1, size=(filterSize, filterSize))
        self.size = filterSize
        
    #this function convolutes the filter over the given image
    #a is the filter b is the image under convolution
    def convolute(self, b):

    	#the "kernel" of convolution
       	a = self.values

       	#each sum is multiplied by this number to "normalize" it
       	normalizationConstant = 1/((self.size)**2)
           
        output = backConvolute(b, a)
        output *= normalizationConstant

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
                    images.append(imgs[counter])
                counter = counter + 1
        else:
            images = imgs
        for i in range(len(self.filters)): #looping through every neuron
            #Convolute old image with filter
            newImg = self.filters[i].convolute(images[i])
            #Apply activation function (ReLU)
            newImg = relu(newImg)
            #Pooling
#            newImg = maxPool(newImg, (2, 2))
            output.append(newImg)
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
        self.layerOut = [None for i in range(len(hiddenLayers) + 1)]
        self.flattenOut = None
        for neurons in hiddenLayers:    #Setting up the convolutional layers
            self.layers.append(ConvLayer(neurons, 3))
            
#        flattenNum = int(inputSize[0]*inputSize[1]/(pow(4, len(hiddenLayers))))
        testInput = [np.zeros(inputSize)] #getting the number of inputs for the flattened layer
        for l in self.layers:
            testInput = l.compute(testInput)
        flattenNum = len(testInput) * testInput[0].shape[0] * testInput[0].shape[1]
        self.finalLayer = Layer(flattenNum, outputSize)
        self.finalConvShape = (len(testInput), testInput[0].shape[0], testInput[0].shape[1])
        
    def setLearningRate(self, newRate):
        self.learningRate = newRate
    def getLearningRate(self):
        return self.learningRate
        
    def feedForward(self, inputData):
        curData = [inputData]
        index = 0
        for l in self.layers:
            curData = l.compute(curData)
            self.layerOut[index] = curData
            index = index + 1
        curData = np.array(curData).flatten()
        curData = curData.reshape(curData.shape[0], 1)
        self.flattenOut = curData
        probabilities = self.finalLayer.compute(curData)
        return probabilities
    
    def train(self, inputData, expectedOutput):
        #iterative step for backprop:
        #ultimate goal is to calculate the change in values for every filter (or kernal)
        #for first fully connected layer, do back prop as normal
        #unflatten the resultant cost
        #for each cnn step:
        #   the cost function at each layer should be an image
        #   create the change in filter weights with animation i saw
        #   back propagate cost
        
        curOutput = self.feedForward(inputData)
        cost = expectedOutput - curOutput
        layerOutputs = [inputData] + self.layerOut
        
        #train the fully connected layer at the end
        gradient = self.lr * cost * curOutput * (1 - curOutput)
        weightChange = gradient.dot(self.flattenOut.transpose())
        self.finalLayer.weights = self.finalLayer.weights + weightChange
        self.finalLayer.bias = self.finalLayer.bias + gradient
        cost = self.finalLayer.weights.transpose().dot(cost) 
        
        #unflatten current cost value
        cost = cost.reshape(self.finalConvShape)
        
        #backpropagate through every layer
        for i in range(len(self.layers)):
            curIndex = len(self.layers)-(i+1)
            curLayer = self.layers[curIndex]
            if curIndex >= 1:
                mult = floor(len(layerOutputs[curIndex+1])/len(layerOutputs[curIndex]))
                newCost = [np.zeros(layerOutputs[curIndex][0].shape) for i in range(len(layerOutputs[curIndex]))]
            else:
                mult = 1
            for c in range(len(cost)):
                if isinstance(layerOutputs[curIndex], list):
                    weightChange = backConvolute(layerOutputs[curIndex][floor(c/mult)], cost[c])
                else:
                    weightChange = backConvolute(layerOutputs[curIndex], cost[c])
                weightChange *= self.lr
                curLayer.filters[c].values += weightChange
                if curIndex >= 1:
                    newCost[floor(c/mult)] += inverseConvolute(curLayer.filters[c].values, cost[c])
            if curIndex >= 1:
                cost = newCost
            
            
            
#
#a = np.zeros((100, 100))
#nn = CNN(a.shape, [1, 2], 2)
#out = nn.feedForward(a)
#print(out)
#for i in range(1000):
#    if (i % 100 == 0):
#        print(i)
#    nn.train(a, np.array([[1],[0]]))
#out = nn.feedForward(a)
#print(out)

nn = CNN((28, 28), [1, 1], 10, lr=0.05)
for i in range(60000):
    if (i % 1000 == 0):
        print(i)
    trainData = data.trainX[i].reshape((28, 28))
    label = np.array(correctNumbersIndex.trainY[i], ndmin=2)
    label = label.T
    nn.train(trainData, label)
    
correctNum = 0
for i in range(10000):
    if (i % 1000 == 0):
        print(i)
    testData = data.testX[i].reshape((28, 28))
    label = np.array(correctNumbersIndex.testY[i], ndmin=2)
    label = label.T
    out = nn.feedForward(testData)
    if (np.argmax(out) == np.argmax(label)):
        correctNum = correctNum + 1
print("Result: ")
print(correctNum)
    