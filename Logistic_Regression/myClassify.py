'''
Description:
    Implement classification by using logistic regression.
    Most examples you can find online are using matrix and
    it is not that clear for me,so I implement by using array
    and reorganize the function so that it become more simple.
    All the data is save in 'data.txt'.These 100 data are mean-
    ingless,they are just some numbers.
Author:
    zjq_1112@qq.com
Date:
    2017.10.26
'''
import matplotlib.pyplot as plt
import numpy as np


def readData():
    '''
    Read data from file:'data.txt'
    :return: 2-tuple includes a 3-list 'X' and a list of 'class number'. 
    '''
    X_Parameter=[]
    class_no=[]
    with open('data.txt','r') as file:
        for line in file:
            line_split=line.split()
            firstPara=float(line_split[0])
            secondPara=float(line_split[1])
            cls_no=int(line_split[2])
            #X_Parameter[0] indicate x0 which is 0:
            X_Parameter.append([1.0,firstPara,secondPara])
            class_no.append(cls_no)
    return X_Parameter,class_no

def sigmoid(x):
    '''
    Calculatee the sigmoid function value given by x
    :param x: value 
    :return: funtion value
    '''
    return 1.0/(1.0+np.exp(-x))

def logistic_regression(trainingdata):
    '''
    Do logistic regression and get the best theta.
    :param trainingdata: training data set
    :return: the best theta
    '''
    data=trainingdata
    #initialize
    learning_rate=1e-4
    theta=np.array([1.0,1.0,1.0])

    epoch=1
    while epoch<=2000:
        theta=gradient_descent(data,theta,learning_rate)
        epoch+=1

    return theta

def gradient_descent(data,theta,learning_rate):
    '''
    Fitting theta by using gradient descent
    :param data: data set
    :param theta: current theta
    :param learning_rate: learning rate eta
    :return: new theta after a single iteration
    '''
    sampleData=data[0]
    classData=data[1]
    gradient=0.0
    sampleNum=len(sampleData)
    for index in range(sampleNum):
        X=np.array(sampleData[index])
        cls=classData[index]
        error=cls-sigmoid(theta.dot(X.T))
        gradient+=error*X

    new_theta=theta+learning_rate*gradient
    return new_theta

def classify(sampleData,theta):
    '''
    Do binary classify by using the model 
    :param sampleData: a 3-list represent parameter of a sample
    :param theta: parameter get from logistic regression
    :return: class number
    '''
    estimate=sigmoid(theta.dot(sampleData.T))
    if estimate >0.5:
        return 1
    else:
        return  0

def show(data):
    '''
    Do classification given by data.
    Draw all the points and the seperate line.
    :param data: data for training and testing
    :return: void
    '''
    sampleNum = len(data[0])
    sampleData=data[0]
    classData=data[1]

    class_1_X=[]
    class_1_Y=[]
    class_2_X=[]
    class_2_Y=[]
    for index in range(sampleNum):
        if classData[index]==0:
            class_1_X.append(sampleData[index][1])
            class_1_Y.append(sampleData[index][2])
        elif classData[index]==1:
            class_2_X.append(sampleData[index][1])
            class_2_Y.append(sampleData[index][2])

    #draw points apart
    plt.plot(class_1_X,class_1_Y,'r*',label='Class 1')
    plt.plot(class_2_X,class_2_Y,'bo',label='Class 2')
    plt.title('Classification by logistic')
    plt.xlabel('Parameter x1')
    plt.ylabel('Parameter x2')
    plt.legend()

    #get theta using logistic regression
    trainingdata=(data[0][:50],data[1][:50])
    theta=logistic_regression(trainingdata)
    print("Theta :\n",theta)

    # draw line
    x=[each*0.01 for each in range(-300,300)]
    y=[((-theta[0]-theta[1]*each)/theta[2])for each in x]
    plt.plot(x,y,'g-')

    #test on training data
    error=0
    for index in range(51,sampleNum):
        if classify(np.array(data[0][index]),theta)!=data[1][index]:
            error+=1
    print("Error percentage :",(error*100)/sampleNum)

    plt.show()

if __name__ == '__main__':
    show(readData())