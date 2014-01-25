import numpy as np
import scipy.optimize as optimize
import sys
import random
fmin = optimize.fmin_cg

def runNN(X, y, lam, input_layer_size, hidden_layer_size, num_labels):

    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1)
    vec = unravel(Theta1, Theta2)
    return fmin(nnCostFunction, vec, fprime=nnGrad, args = (input_layer_size, hidden_layer_size, num_labels, X, y, lam))

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
    label_vector = np.zeros((m, num_labels))
    for i, row in enumerate(label_vector):
        row[y[i]] = 1
    # backprop
    d3 = htheta - label_vector
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
    J = 0
    m, n = X.shape
    z2 = np.dot(X, theta1.T)
    a2 = biasTerm(sigmoid(z2))
    z3 = np.dot(a2, theta2.T)
    htheta = sigmoid(z3)
    label_vector = np.zeros((m, num_labels))
    for i, row in enumerate(label_vector):
        row[y[i]] = 1
    J = np.sum((- label_vector *  np.log(htheta)) - ((1-label_vector) * np.log(1 - htheta))) / m
    theta1[:, 0] = 0
    theta2[:, 0] = 0
    return (J/m) + (np.sum(theta1 ** 2) + np.sum(theta2 ** 2)) * (lam / (2 * m))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def predict(t1, t2, X):
    m, n = X.shape
    num_labels, _ = t2.shape
    X = biasTerm(data[:, 1:])
    a2 = biasTerm(sigmoid(X * t1.T))
    a3 = sigmoid(a2 * t2.T)
    return np.argmax(a3, axis=1)
# 
def runTest():
    num_labels = random.randint(5, 10)
    input_size = random.randint(100, 1000)
    hidden_size = random.randint(num_labels, input_size)
    m, n = shape = (10000, input_size)
    X = biasTerm(np.random.randint(10, 100, shape))
    y = np.random.randint(1, num_labels, shape)
    params = runNN(X, y, 0.25, input_size, hidden_size, num_labels)

def prepareLearningCurve(X, y, Xval, yval):
    m, n = X.shape
    for num in range(0, m, m / 1000):
        params = runNN(X[:num, :], y[: num, :], 0.25)

def computeNumericalGradient():
    

def showLearningCurve():


if __name__ == "__main__":
    _, trainPath, testPath, benchPath = sys.argv    
    data = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=1)
    m, n = data.shape
    X = biasTerm(data[:, 1:])
    y = data[:, 0]
    input_layer_size  = 784
    hidden_layer_size = 25
    num_labels = 10
    params = runNN(X, y, 0.25, input_layer_size, hidden_layer_size, num_labels)
    t1, t2 = reravel(params, hidden_layer_size, input_layer_size, num_labels)
    Xtest = np.loadtxt(open(testPath,"rb"),delimiter=",",skiprows=1)
    p = predict(t1, t2, X)
    ybench = np.loadtxt(open(benchPath,"rb"),delimiter=",",skiprows=1)
    print "Percent Accurate" + str(np.sum(p == ybench) / float(m))