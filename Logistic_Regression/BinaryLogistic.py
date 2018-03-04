'''
    Description: Implemente a binary class logistic regression.
    
    Author: zjq_1112@qq.com
    
    Date:2017.11.25
'''

import numpy as np
import math

class BinaryLogistic:
    '''
        Description: 
            Class to implement binary logistic regression.
        
        Attributes:
            self.data: Store all the sample data.
            self.featureNum: The number of sample features.
            self.learningRate: Learining rate used in gradient descent.
            self.traingData: Take 70% of data as training data.
            self.testingData: Take 30% of data as testing data.
            self.trainingNum: The number of training data.
            self.testingNum: The number of testing data.
    '''

    def _readData(self):
        '''
            Description:
                Read data from file:'data.txt' and store all 
                the data into self.data
        '''
        self.data=[]
        with open('data.txt', 'r') as file:
            for line in file:
                X = [1.0]
                Y = []
                line_split = line.split()
                firstPara = float(line_split[0])
                secondPara = float(line_split[1])
                cls_no = int(line_split[2])
                # X_Parameter[0] indicate x0 which is 0:
                X.extend([ firstPara, secondPara])
                Y.append(cls_no)
                self.data.append((np.matrix(X),np.matrix(Y)))

    def __init__(self):
        '''
        Description:   
            Initialize elarning model.
        '''
        self.featureNum=2
        self.theta=np.ones((1,self.featureNum+1))
        self.learningRate=1e-4
        self._readData()

        #70% as training data,30% as testing data.
        totalNum=len(self.data)
        trainNum=math.ceil(totalNum*0.7)
        self.trainingData=self.data[:trainNum]
        self.testingData=self.data[trainNum:]
        self.trainingNum=len(self.trainingData)
        self.testingNum = len(self.testingData)

    def _sigmoid(self,X):
        '''
            Description:
                Calculate sigmoid function value given X.
        :param X: X value in matrix.
        :return: Sigmoid function value.
        '''
        return 1/(1+np.exp(-(self.theta*X.T)))

    def _hypothesis(self,X):
        '''
            Description: Given X, determin its class label.
        :param X: Feature matrix
        :return: Class label of X
        '''
        if(self._sigmoid(X)>=0.5):
            return 1
        else:
            return 0

    def training(self):
        '''
            Description:
                Train the model on training data.
        :return: void
        '''
        epoch=0
        while (epoch<1000):
            #gradient acscent
            for index in range(self.trainingNum):
                currentSample=self.trainingData[index]
                error=currentSample[1]-self._sigmoid(currentSample[0])
                self.theta=self.theta+self.learningRate*error*currentSample[0]
            epoch+=1

        print("Theta :",self.theta)

    def testing(self):
        '''
            Description:Test model on testing data.
        :return: void
        '''
        error=0
        for index in range(self.testingNum):
            currentSample=self.testingData[index]
            if self._hypothesis(currentSample[0])!=currentSample[1]:
                error+=1

        print("%d error in %d data ,error percentage %f"%(error,self.testingNum,error/self.testingNum))



if __name__ == '__main__':
    model=BinaryLogistic()
    model.training()
    model.testing()
