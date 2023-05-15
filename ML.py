import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


path = '/media/yassen/01D5F192D609BFA0/Education/Third year 2023/Expert Systems/project/data.txt'

# get data as scv file from the given path
data = pd.read_csv(path, header=None, names=[
                   'age', 'ratio', 'gender', 'smoking', 'alcohol', 'result'])

# insert a column in the data at position 0 which 'll be the bias
data.insert(0, 'ones', 1)

# print("data = ")
# print(data.head(10))
# print(data.describe())

# get people who have a positive covid test
positive = data[data['result'].isin([1])]

# get people who have a negative covid test
negative = data[data['result'].isin([0])]

# print("positive = ")
# print(positive)
#
# print("negative = ")
# print(negative)


# plot the graph between the age and the expected ratio to have covid

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(positive['age'], positive['ratio'],
           s=50, c='b', marker='o', label='positive')

ax.scatter(negative['age'], negative['ratio'],
           s=50, c='r', marker='x', label='negative')

plt.show()


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, step = 1)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(nums, sigmoid(nums), 'r')

plt.show()


# get dimension of the data matrix
cols = data.shape[1]

# let X be the features matrix columns for 0 to cols-1 for all rows, contains the bias column
X = data.iloc[:, 0:cols-1]

# let Y be the real test result matrix
Y = data.iloc[:, cols-1:cols]

# print("X = \n")
# print(X)
# print("Y = \n")
# print(Y)

# turns X and Y into arrays
x = np.array(X.values)
y = np.array(Y.values)

# define theta matrix of dimension (features x 1)
theta = np.zeros(x.shape[1])

# print dimensions

# print("x.shape = \n")
# print(x.shape)
# print("y.shape = \n")
# print(y.shape)
# print('theta.shape = ')
# print(theta.shape)

# definition of the cost function


def cost(thetaVal, xVal, yVal):
    thetaVal = np.matrix(thetaVal)
    xVal = np.matrix(xVal)
    yVal = np.matrix(yVal)

    # calculate cost value
    firstTerm = np.multiply(-yVal, np.log(sigmoid(xVal*thetaVal.T)))
    secondTerm = np.multiply((1- yVal), np.log(1 - sigmoid(xVal*thetaVal.T)))
    return np.sum(firstTerm - secondTerm) / len(xVal)


costBeforeOptimization = cost(theta, x, y)

# # print cost value
# print('cost = ', round(thisCost, 2))

# definition of the gradient function


def gradient(thetaV, xV, yV):
    thetaVal = np.matrix(thetaV)
    xVal = np.matrix(xV)
    yVal = np.matrix(yV)

    parameters = int(thetaVal.ravel().shape[1])
    grade = np.zeros(parameters)

    error = sigmoid(xVal * thetaVal.T) - yVal

    for i in range(parameters):
        term = np.multiply(error, xVal[:, i])
        grade[i] = np.sum(term)/ len(xVal)

    return grade


# using fmin_tnc function from scipy library to optimize the cost function
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))

# result is a tuple contains theta values, how many times it tries to get the theta
# values to optimize the cost function

# print(result)


costAfterOptimization = cost(result[0], x, y)
# print cost values before optimization and after

print('cost = ', round(costBeforeOptimization, 2))
print('cost after optimization = ', round(costAfterOptimization, 2))

# definition of predict function which get the predicted diagnosing result of the given
# training data


def predict(theta_v, x_v):
    probability = sigmoid(x_v*theta_v.T)
    return [1 if val >= 0.5 else 0 for val in probability]


# theta values after optimizing the cost function
theta_min = np.matrix(result[0])

# get predicted values of the diagnosing test
prediction = predict(theta_min, x)

# print predicted values
# print(prediction)

# calculate the accuracy of the model
correct = [1 if (a + b) % 2 == 0 else 0 for (a, b) in zip(prediction, y)]
accuracy = sum(map(int, correct)) / len(prediction)
print('accuracy = {0}%'.format(round(accuracy*100, 2)))


# Precision = TruePositives / (TruePositives + FalsePositives)
# Recall = TruePositives / (TruePositives + FalseNegatives)
# F1_score =  2*((Precision * Recall) / (Precision + Recall))

truePositive = [1 if (a + b) == 2 else 0 for (a, b) in zip(prediction, y)]
falsePositive = [1 if (a + b) == 1 and b == 1 else 0 for (a, b) in zip(prediction, y)]
falseNegative = [1 if (a + b) == 1 and a == 1 else 0 for (a, b) in zip(prediction, y)]

Precision = sum(map(int, truePositive)) / sum(map(int, (truePositive + falsePositive)))
Recall = sum(map(int, truePositive)) / sum(map(int, (truePositive + falseNegative)))
F1_Score = 2 * ((Precision * Recall) / (Precision + Recall))

print('Precision = {0}'.format(round(Precision, 2)))
print('Recall = {0}'.format(round(Recall, 2)))
print('F1_Score = {0}'.format(round(F1_Score, 2)))


