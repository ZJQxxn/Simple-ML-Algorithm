'''
Description:
    Use perceptron to classify dataset into 2 classes.
    
    In theory, the perceptron performe well when dataset 
    is linear seperable.But this object only test perceptron's 
    performance on an ordinary dataset(which means it is
    nonlinear seperable).Select model by using 4-fold cross
    validation.
        
Author:
    zjqseu@gmail.com
'''

import numpy as np
import random


class Perceptron:
    '''
    Description:
        Class of perceptron.
    '''
    @classmethod
    def readData(cls):
        '''
        Description:
            Read data from file.
        :return: void
        '''
        cls.pos_sample = []
        cls.neg_sample = []
        with open("data.csv", 'r') as file:
            for line in file.readlines():
                attrs = line.split(',')
                if attrs[1] == 'B':
                    cls.pos_sample.append((1, np.matrix(attrs[2:], "float")))
                else:
                    cls.neg_sample.append((-1, np.matrix(attrs[2:], "float")))


    def __init__(self):
        pos_sample=[sample for sample in Perceptron.pos_sample]
        neg_sample=[sample for sample in Perceptron.neg_sample]
        #Randomly choose 80% samples as training data,
        #and the rest are testing data.
        self.subset=[[] for i in range(4)]
        for fold in range(4-1):
            for i in range(0, int(len(pos_sample) * 0.25)):
                num = random.randint(0, len(pos_sample) - 1)
                self.subset[fold].append(pos_sample[num])
                pos_sample.pop(num)
            for i in range(0, int(len(neg_sample) * 0.25)):
                num = random.randint(0, len(neg_sample) - 1)
                self.subset[fold].append(neg_sample[num])
                neg_sample.pop(num)
        self.subset[3].extend(pos_sample)
        self.subset[3].extend(neg_sample)
        #Initial parameters
        self.weight=np.matrix([random.random() for i in range(0,30)])
        self.bias=random.random()
        self.learning_rate=1e-4


    def estimator(self,X):
        '''
        Description:
            Given sample attributes, estimate its class lable.
        :param X: Sample attributes.
        :return: Class label.
        '''
        estimation=self.weight * X.T
        if estimation>=0:
            return 1
        else:
            return -1


    def modeling(self):
        '''
        Description:
            Train this perceptron using cross validation.
        :return: void
        '''
        misclassified = []
        error_rate=0
        #4-fold cross validation
        for i in range(4):
            test_data=self.subset[i]
            train_data=sum(self.subset[:i]+self.subset[i+1:],[])
            #Train
            while True:
                for sample in train_data:
                    if self.estimator(sample[1]) * sample[0] <= 0:
                        misclassified.append(sample)
                for sample in misclassified:
                    self.weight += self.learning_rate * sample[0] * sample[1]
                    self.bias += self.learning_rate * sample[0]
                if len(misclassified) < int(len(train_data) * 0.1):
                    break
                else:
                    misclassified.clear()
            #Test
            error = 0
            sample_num = len(test_data)
            for sample in test_data:
                if self.estimator(sample[1]) != sample[0]:
                    error += 1
            error_rate+=error/sample_num
        #Average error rate
        return error_rate/4


class Analyse:
    '''
    Description:
        Class to calculate some evaluation value.
    '''

    def __init__(self,data,model):
        self.data=data
        self.model=model
        self.confusion_matrix = [[0, 0], [0, 0]]


    def analyse(self):
        #Estimate positive samples
        for sample in self.data[0]:
            if model.estimator(sample[1])==1:
                self.confusion_matrix[0][0]+=1
            else:
                self.confusion_matrix[0][1]+=1
        #Estimate negative samples
        for sample in self.data[1]:
            if model.estimator(sample[1])==-1:
                self.confusion_matrix[1][1]+=1
            else:
                self.confusion_matrix[1][0]+=1
        print("Confusion matrix :\n",self.confusion_matrix)
        print("Precision: %f"%
              (self.confusion_matrix[0][0]/(self.confusion_matrix[0][0]+self.confusion_matrix[1][0])))
        print("Recall: %f"%
              (self.confusion_matrix[0][0]/(self.confusion_matrix[0][0]+self.confusion_matrix[1][1])))



if __name__ == '__main__':
    Perceptron.readData()
    models=[Perceptron() for i in range(0,10)]
    best_model=models[0]
    error_rate=1
    num=0
    #Train 20 models and pick out the best one.
    for model in models:
        num+=1
        print("===========================")
        print("Model %d"%num)
        cur_error_rate=model.modeling()
        print("\nAverage error rate is %f"%cur_error_rate)
        if cur_error_rate<error_rate:
            best_model=model
            error_rate=cur_error_rate
    print("\nBest error rate is %f"%error_rate)
    #Analyse the best model
    a=Analyse([Perceptron.pos_sample,Perceptron.neg_sample],best_model)
    a.analyse()