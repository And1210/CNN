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

def main:
    for i in range(60000)
    convolutionalNetwork = CNN(inputsize, hiddenLayers, ouputSize, lr)
    feedforward
    train