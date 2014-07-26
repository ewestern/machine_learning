import numpy as np
import scipy.optimize as optimize
import sys
import random
fmin = optimize.fmin_cg
import matplotlib.pyplot as plt
import math

# todo: computeNumericalGradient

def runNN(X, y, lam, input_layer_size, hidden_layer_size, num_labels):
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1)
    vec = unravel(Theta1, Theta2)
    return fmin(nnCostFunction, vec, fprime=nnGrad, args = (input_layer_size, hidden_layer_size, num_labels, X, y, lam), gtol = 1e-4)

def unravel(mat1, mat2):
    n1, m1 = mat1.shape
    n2, m2 = mat2.shape
    return np.append(np.reshape(mat1, (n1 * m1, 1)), np.reshape(mat2, (n2 * m2, 1)))

def reravel(vec, hidden_layer_size, input_layer_size, num_labels):
    mat1 = np.reshape(vec[0: hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    mat2 = np.reshape(vec[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
    return mat1, mat2

def biasTerm (mat):
    m, n = mat.shape
    return np.append(np.ones((m, 1)), mat, axis=1)

def nnGrad(theta, input_size, hidden_size, num_labels, X, y, lam):
    theta1, theta2 = reravel(theta, hidden_size, input_size, num_labels)
    m, n = X.shape
    # feedforward
    z2 = np.dot(X, theta1.T)
    a2 = biasTerm(sigmoid(z2))
    z3 = np.dot(a2, theta2.T)
    htheta = sigmoid(z3)
    labelMatrix = np.zeros((m, num_labels))
    for i, row in enumerate(labelMatrix):
        row[y[i]] = 1
    # backprop
    d3 = htheta - labelMatrix
    d2 = sigmoidGradient(biasTerm(z2)) * np.dot(d3, theta2)
    bigDelta2 = np.dot(d3.T , a2)
    bigDelta1 = np.dot(d2[:, 1:].T , X)
    t1reg = theta1 * (lam / m)
    t1reg[:, 0] = 0
    grad1 = bigDelta1 / m  + t1reg
    t2reg = theta2 * (lam / m);
    t2reg[:, 0] = 0
    grad2 = bigDelta2 / m + t2reg
    return unravel(grad1, grad2)    

def nnCostFunction(theta, input_size, hidden_size, num_labels, X, y, lam):
    theta1, theta2 = reravel(theta, hidden_size, input_size, num_labels)
    m, n = X.shape
    z2 = np.dot(X, theta1.T)
    a2 = biasTerm(sigmoid(z2))
    z3 = np.dot(a2, theta2.T)
    htheta = sigmoid(z3)
    label_vector = np.zeros((m, num_labels))
    for i, row in enumerate(label_vector):
        row[y[i]] = 1

    J = np.sum((- label_vector *  np.log(htheta)) - ((1. - label_vector) * np.log(1. - htheta))) / m
    theta1[:, 0] = 0
    theta2[:, 0] = 0
    return J + (np.sum(theta1 ** 2) + np.sum(theta2 ** 2)) * (lam / (2 * m))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def predict(t1, t2, X):
    a2 = biasTerm(sigmoid(np.dot(X , t1.T)))
    a3 = sigmoid(np.dot(a2 , t2.T))
    return np.argmax(a3, axis=1)
# 
def runTest():
    num_labels = random.randint(5, 10)
    input_size = random.randint(100, 1000)
    hidden_size = random.randint(num_labels, input_size)
    m, n = (10000, input_size)
    X = biasTerm(np.random.rand(m, n))
    y = np.random.randint(1, num_labels, (m, 1))
    # params = runNN(X, y, 0.25, input_size, hidden_size, num_labels)
    showLearningCurve(X, y, input_size, hidden_size, num_labels)

def showValidationCurve(X, y, input_size, hidden_size, num_labels):
    m, n = X.shape
    Xtrain, Xval = X[:m*3/5, :], X[m*3/5:m*4/5, :]
    ytrain, yval = y[:m*3/5, :], y[m*3/5:m*4/5, :]
    lamVec = np.array([0.0001 * 2 ** v for v in range(12)])
    lamVec = lamVec.reshape((len(lamVec), 1))
    pTrain = np.zeros((len(lamVec), 1))
    pVal = np.zeros((len(lamVec), 1))
    for i, lam in enumerate(lamVec):
        params = runNN(Xtrain, ytrain, lam, input_size, hidden_size, num_labels)
        pTrain[i] = nnCostFunction (params, input_size, hidden_size, num_labels, Xtrain, ytrain, 0)
        pVal[i] = nnCostFunction (params, input_size, hidden_size, num_labels, Xval, yval, 0)
    plt.plot(lamVec, pTrain, 'g^', lamVec, pVal, 'b^')
    plt.show()


def prepareLearningCurve(X, y, Xval, yval, input_size, hidden_size, num_labels):
    m, n = X.shape
    samples = len(range(m/10, m, m / 20))
    error_train = np.zeros((samples, 1))
    error_val = np.zeros((samples, 1))
    print "Input size: {0}, hidden size: {1}, num_labels: {2}".format(input_size, hidden_size, num_labels)        
    for i, num in enumerate(range(m/10, m, m / 20)):
        print "Evaluating X({0}, {1})".format(*X[:num, :].shape)
        params = runNN(X[:num, :], y[: num, :], 0.02, input_size, hidden_size, num_labels)
        error_train[i] = nnCostFunction (params, input_size, hidden_size, num_labels, X[:num, :], y[:num, :], 0)
        error_val[i] = nnCostFunction (params, input_size, hidden_size, num_labels, Xval, yval, 0)
    return error_train, error_val
     
def estimateAccuracy(X, y, lam, input_size, hidden_size, num_labels):
    m, n = X.shape
    Xtrain, Xtest = X[:m*4/5, :], X[m*4/5:, :]
    ytrain, ytest = y[:m*4/5, :], y[m*4/5:, :]    
    params = runNN(Xtrain, ytrain, lam, input_size, hidden_size, num_labels)
    t1, t2 = reravel(params, hidden_size, input_size, num_labels)
    predictions = predict(t1, t2, Xtest) 
    # print "predictions: " + str(predictions)
    trained = predict(t1, t2, Xtrain)
    precision = np.sum(predictions.reshape((len(predictions), 1)) == ytest) / float(len(predictions))
    print "Neural Network completed with a prediction accuracy of {0} %".format(str(100 * precision))
    print "Prediction {0} % accuracy on training set".format(100 * np.sum(trained.reshape((len(trained), 1)) == ytrain) / float(len(ytrain)))
# X, y, lam, input_layer_size, hidden_layer_size, num_labels
##def computeNumericalGradient():
    
# def compareWithBenchmark(X, y, bench, input_size, hidden_size, num_labels):
#     m, n = X.shape

def showLearningCurve(X, y, input_size, hidden_size, num_labels):
    m, n = X.shape
    Xtrain, Xval = X[:m*2/3, :], X[m*2/3:, :]
    ytrain, yval = y[:m*2/3, :], y[m*2/3:, :]
    trainingError, validationError = prepareLearningCurve(Xtrain, ytrain, Xval, yval, input_size, hidden_size, num_labels)   
    samples, _ = trainingError.shape
    xaxis = np.arange(0, samples, 1)
    plt.plot(xaxis, trainingError, 'g^', xaxis, validationError, 'b^')
    plt.show()    

if __name__ == "__main__":
    _, trainPath, testPath, benchPath = sys.argv    
    data = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=1)
    m, n = data.shape
    X = biasTerm(data[:, 1:])
    y = data[:, 0].reshape((m, 1)).astype(int)
    print "Xshape: " + str(X.shape)
    print "yshape: " + str(y.shape)
    input_layer_size  = 784
    hidden_layer_size = 25
    num_labels = 10
    estimateAccuracy(X, y, 0.25, input_layer_size, hidden_layer_size, num_labels)
    # showLearningCurve(X, y, input_layer_size, hidden_layer_size, num_labels)
    # showValidationCurve(X, y, input_layer_size, hidden_layer_size, num_labels)

    # params = runNN(X, y, 0.25, input_layer_size, hidden_layer_size, num_labels)
    # t1, t2 = reravel(params, hidden_layer_size, input_layer_size, num_labels)
    # Xtest = np.loadtxt(open(testPath,"rb"),delimiter=",",skiprows=1)
    # p = predict(t1, t2, X)
    # ybench = np.loadtxt(open(benchPath,"rb"),delimiter=",",skiprows=1)
    # print "Percent Accurate" + str(np.sum(p == ybench) / float(m))
