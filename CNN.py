import numpy as np
import mnistdb.io as mio
import skimage.measure
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

def reluDerivative(image):
    out = image
    out[out<=0] = 0
    out[out>0] = 1
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
            newImg = maxPool(newImg, (2, 2))
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
            prevImgs = self.layerOut[curIndex]
            prevDerivative = [np.zeros(prevImgs[i].shape) for i in range(len(prevImgs))]
            
            count = 0
            #for every feature map:
            for cur_kernal in range(len(cost)):
                c = cost[cur_kernal]
                for x in range(0, curLayer.filters[cur_kernal].shape[1]):
                    for y in range(0, curLayer.filters[cur_kernal].shape[0]):
                        for x_img in range(prevImgs[floor(count/2)].shape[1]):
                            for y_img in range(prevImgs[floor(count/2)].shape[0]):
                                #adjusting the kernal weights
                                curLayer.filters[cur_kernal][x][y] = curLayer.filters[cur_kernal][x][y] + c[x_img][y_img]*reluDerivative(prevImgs[x_img][y_img])*(prevImgs[curLayer.filters[cur_kernal].shape[1]-1-x_img][curLayer.filters[cur_kernal].shape[0]-1-y_img])
                                
                for x_img in range(prevImgs[floor(count/2)].shape[1]):
                    for y_img in range(prevImgs[floor(count/2)].shape[0]):
                        #backpropagating the cost to the previous layer
#                        cost[cur_kernal][x_img][y_img]*reluDerivative(prevImgs[x_img][y_img])

a = np.zeros((100, 100))
nn = CNN(a.shape, [2, 4], 2)
out = nn.feedForward(a)
nn.train(a, np.array([[1],[0]]))