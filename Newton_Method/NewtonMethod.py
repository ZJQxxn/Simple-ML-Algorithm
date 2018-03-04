'''
Description:
    Use Newton Method to find where  the first derivation
    of sigmoid funtion is 0.

Author:
    zjq_1112@qq.com

Date:
    2017.10.29
'''

import matplotlib.pyplot as plt
import numpy as np
import random

def sigmoid(x):
    '''
    Calculate sigmoidfunction given X
    :param x: variable value
    :return: function value
    '''
    return (1.0/(1+np.exp(-x)))

def first_derivation(x):
    '''
    Claculate first derivation of sigmoid function
    given x
    :param x:variable value 
    :return: first derivation value
    '''
    return sigmoid(x)*(1-sigmoid(x))

def newton_method():
    x=random.randint(-6,6)
    while abs(first_derivation(x))>0.01:
        x=x-(sigmoid(x)/first_derivation(x))
    return x

def drawLine():
    X = [x * 0.01 for x in range(-600, 600)]
    Y = [sigmoid(x) for x in X]

    # sigmoid function line
    plt.plot(X, Y, 'r-')
    # draw minima
    x = newton_method()
    plt.plot([x], [sigmoid(x)], 'b*')

    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('Newton Method for Sigmoid')
    plt.show()


if __name__ == '__main__':
    drawLine()