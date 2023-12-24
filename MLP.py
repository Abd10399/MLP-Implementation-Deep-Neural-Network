#Importing the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from math import sqrt

############################### Defining Activation Function Classes and their methods! ##########################################
class Relu_Class:
  def activation(z):
    return np.maximum(0,z)
  def prime(z):
      z[z<=0] = 0
      z[z>0] = 1
      return z

class Leaky_Relu_Class:
  @staticmethod
  def activation(z):
    alpha = 0.1
    return np.where(z<=0,alpha*z,z)
  def prime(z):
    alpha = 0.1
    return np.where(z<=0,alpha,1)

class Sigmoid_Class:
  @staticmethod
  def activation(z):
      return 1 / (1 + np.exp(-z))
  def prime(z):
      return Sigmoid_Class.activation(z) * (1 - Sigmoid_Class.activation(z))

class tanh_Class:
  @staticmethod
  def activation(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
  def prime(z):
    return 1 - np.power(tanh_Class.activation(z), 2)

class softmax_Class:
  @staticmethod
  def activation(x):
    e = np.exp(x-np.max(x))
    s = np.sum(e, axis=1, keepdims=True)
    return e/s
  @staticmethod
  def prime(z):
      return softmax_Class.activation(z)*softmax_Class.activation(1-z)

############################### Defining the Softmax Loss (Cross_Entropy) Class and its related methods #######################################################
class Cross_Entropy:
  def __init__(self, activation_fn):
      self.activation_fn = activation_fn

  def activation(self, z):
    return self.activation_fn.activation(z)

  def loss(y_true, y_pred):
      epsilon=1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      N = y_pred.shape[0]
      loss = -np.sum(y_true*np.log(y_pred+1e-9))/N
      return loss

  @staticmethod
  def prime(Y, AL):
      return AL - Y

  def delta(self, y_true, y_pred):
      return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)
  
  
############################### Defining the MLP Class and its related methods #######################################################
class MultiLayerPerceptron:
  #Constructor
  def __init__(self, dimensions, activations, weight = "uniform"):
    """
    list of dimensions (input, hidden layer(s), output)
    list of activation functions (relu, softmax etc) (in the case of 1 hidden layer)
    """
    self.num_layers = len(dimensions) #Tells us the number of layers in the MLP model
    self.loss = None      #init
    self.alpha = None   #init
    # Weights, biases and the activations will be initialized as dictionaries, with the key being the index and the value being the subsequent Matrix
    self.w = dict()
    self.b = dict()
    self.activations = dict()
    self.lambd = None
    # initial values for the weights as well as the biases according to the number of layers we have in the model
    for i in range(self.num_layers - 1):
      # Starting from index i=1, we initialize weights according to the weight parameter
      if weight == "zeros":
        self.w[i + 1] = np.zeros((dimensions[i],dimensions[i+1]))
      elif weight == "uniform":
        self.w[i+1] = np.random.uniform(-1,1, (dimensions[i], dimensions[i+1]))
      elif weight == "gaussian":
        self.w[i+1] = np.random.normal(0,1, (dimensions[i], dimensions[i+1]))
      elif weight == "xavier":
        std_dev = sqrt(2/(dimensions[i] + dimensions[i+1]))
        self.w[i+1] = np.random.normal(0,std_dev, (dimensions[i], dimensions[i+1]))
      elif weight == "kaiming":
        self.w[i+1] = np.random.normal(0,2/(dimensions[i]), (dimensions[i], dimensions[i+1]))
      else:
        self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])  #Weight matrix set to the |i| X |i+1|

      #Now we init the biases to 0's according to the appropriate dimension
      self.b[i + 1] = np.zeros(dimensions[i + 1])

      #Now we init the activations (starting from 2, since 1 will have the x as input)
      self.activations[i + 2] = activations[i]

  def forward_pass(self, x):
    """
    Allows us to give an input, and go through the whole network till we reach the outputs
    Returns a tuple containing the z's and the activations applied onto the z's according to each activation function we've provided
    """
    z = dict()  #Creating a dictionary for the z's
    a = dict()  #Creating a dictionary for the a's      ##Note, these dictionaries are for a single forward prop instance

    a[1] = x    #Setting as the first activation, the x's since that is what we will use

    for i in range(1,self.num_layers):      #Starting from 1, as all the dictionaries are indexed by 1
      #Calculating the z's and the a's
      z[i+1] = np.dot(a[i], self.w[i]) + self.b[i]     #Applying the weights to the x's/prev activation to get the next set of values
      a[i+1] = self.activations[i+1].activation(z[i+1])   #We apply the next activation function on the z we just calculated and put it in the next activation layer to be
                                                          # calculated in the next iteration for z

    return (z, a)   #Returning the tuple of z's and a's (to be used in the back prop / for prediction)

  def updater(self, i, dw, delta):
    """
    updates the weights/biases according to the index (i), the partial derivative and the error, uses L2 regularization if lambd set to non-zero, else its not regularized
    """
    reg_matrix_w = np.zeros_like(self.w[i])

    if self.regg == 1:
      dw += (self.lambd) * np.sign(self.w[i])
      pass
    elif self.regg == 2:
      dw += 2 * (self.lambd) * self.w[i]

    #If self.regg == 0, then we're not going to be using regularization and hence dw will not be updated
    self.w[i] = self.w[i] - self.alpha * dw
    self.b[i] = self.b[i] - self.alpha * np.mean(delta, 0)

    #The above will be done for each iteration in the gradient descent

  def back_pass(self, z, a, y):
    """
    Compute the partial derivatives and the error for the final layer, does backpropogation, by propogating the gradients back through the network
    """
    delta = self.loss.delta(y, a[self.num_layers])
    dw = np.dot(a[self.num_layers - 1].T, delta)

    derivatives = dict()  #Creating a dictionary for the delta and the derivatives
    derivatives[self.num_layers - 1] = (dw, delta) #Putting the stuff computed above in the dictionary

    #Now we have the back propogation loop!!                        ####################################################################### Very Important!!!!!
    for i in reversed(range(2, self.num_layers)):
      delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i])
      dw = np.dot(a[i - 1].T, delta)
      derivatives[i - 1] = (dw, delta)

    for key, value in derivatives.items():
      self.updater(key, value[0], value[1])       #value[0] contains dw, while value[1] contains delta, we're unpacking the tuple we packed and added to the derivatives dictionary

    ############################        Completed Back Propogation!!!


  def fit(self, x_train, y_train, epochs, mini_batch_size, alpha, x_test, y_test, regularization = 0, lambd = 0):
    """
    main function used to fit the model using the training data, and then used to test using the evac function
    """
    #Converting one_hot encoding for y_train and y_test
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    #Flattening the data for the x:
    n_features= x_train.shape[1]*x_train.shape[2]
    x_train = x_train.reshape(-1, n_features)
    x_test = x_test.reshape(-1, n_features)
    #Reshapes both x_train and x_test from a 3D Matrix to a 2D Matrix

    #Setting up some variables which we skipped over in the initialization which are needed
    self.alpha = alpha
    self.loss = Cross_Entropy(self.activations[self.num_layers])
    # self.lambd = regularization() Need to fix this!!!!!!!             ######## FIX THIS
    if regularization == 0:
      print("No Reg")
      self.regg = 0
      self.lambd = 0
    elif regularization == 1:      ##Doing L1 Regularization
      print("L1 Reg")
      self.regg = 1
      self.lambd = lambd

    else: #Do L2 Regularization       #Regularization = 2,3 4 etc anything
      print("L2 Reg")
      self.regg = 2
      self.lambd = lambd


    #Setting up some miscelaneous parameters for graphing!
    self.train_logger = []
    self.test_logger = []
    self.cost_logger = []

    ##Starting the SGD!!!

    for i in range(epochs):
      #Randomizing the data:
      permutation = np.random.permutation(x_train.shape[0])
      x_S = x_train[permutation]
      y_S = y_train[permutation]

      #Setting up the mini_batches
      num_mini_batches = x_train.shape[0] // mini_batch_size

      for j in range(num_mini_batches):
        start_i = j*mini_batch_size
        end_i = (j+1)*mini_batch_size

        #Given the indices, now we do a forward pass on them, and then a backward pass
        (z_dict, a_dict) = self.forward_pass(x_S[start_i:end_i])

        #Now using the values in z_dict and a_dict, we do a backward pass to propogate the gradients back to the layers
        self.back_pass(z_dict, a_dict, y_S[start_i:end_i])
        #Done training now

      #Now we check the training and testing accuracy and run the evaluate_acc function
      training_acc = self.evaluate_acc(self.predict(x_train), np.argmax(y_train,axis = 1))
      testing_acc = self.evaluate_acc(self.predict(x_test), np.argmax(y_test,axis = 1))

      #training_acc = np.mean(self.predict(x_train) == np.argmax(y_train,axis = 1))
      #testing_acc = np.mean(self.predict(x_test) == np.argmax(y_test,axis = 1))

      #Logging the accuracies
      self.train_logger.append(training_acc)
      self.test_logger.append(testing_acc)

      # print results for monitoring while training
      print("Epoch {0} train data: {1} %".format(i, 100 * (training_acc)))
      print("Epoch {0} test data: {1} %".format(i, 100 * (testing_acc)))

  def predict(self, x):
    """
    Feeds forward the x to get the output, and returns the argmax
    """
    (garbage, a) = self.forward_pass(x)
    return np.argmax(a[self.num_layers], axis = 1)

  def evaluate_acc(self, yhat, y):  ##Used for evaluating the accuracy, given true y's and predicted y's
    return np.mean(yhat == y)


"""
Running the Model
print(fashion_trainset_x.shape, fashion_trainset_y.shape)
print(fashion_testset_x.shape, fashion_testset_y.shape)
net = MultiLayerPerceptron([784,64,64,10],[Relu_Class, Relu_Class, softmax_Class], weight = "kaiming")
net.fit(fashion_trainset_x, fashion_trainset_y, 100, 256, 0.001, fashion_testset_x, fashion_testset_y, regularization = 0, lambd = 0.05)   ##Need to add regularization at the end

For Regularization:
'''
No: regularization = 0, L1: regularization = 1, L2: regularization = 2
For L1, L2 u assign lambd as the regularization strength
Pre test, L1: keep lambd = 0.0005 (Very Weak, Accuracy) -> it implies that the higher most of our features are useful, and we loose information if we force them to go to 0
L2: lambd = 0.01 (Stronger, If stronger needed, must increase the batch size or the learning rate to kill randomization/noise resulting from L2)

Note: It may appear as if accuracy goes down in epochs, but it will pull up eventually, as the neurons get used to the regularization and start making good predictions
For a batch size of 256, can increase the epochs to 500, approx compute time will be 20 mins per fit for MNIST
'''
"""