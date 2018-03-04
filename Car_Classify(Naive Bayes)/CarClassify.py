'''
    Description: 
        Evaluate car by using naive bayes.
        
        Consider the distribution of each feature as mutinomial,
        the average error on testing set is about 17%,while can 
        be around 9% at least.(90% training set, 10% testing set)
        
    Author:
        zjq_1112@qq.com
        
    Date:
        2017.11.26
        
    TODO:
        Complete annotation.
'''

import math
import random

class CarClassify:
    '''
        Description:
            Train model to evaluate car.
             
        Attributes:
            
    '''

    def _readFile(self):
        self.price_category=['vhigh','high','med','low']
        self.doors_category=['2','3','4','5more']
        self.persons_category=['2','4','more']
        self.lugboot_category=['small','med','big']
        self.safety_category=['low','med','high']
        self.label_category=['unacc','acc','good','vgood']
        data=[]
        with open('car.data') as file:
            for line in file:
                X=[]
                Y=[]
                #convert categorical value to integers
                line_split=line.split(',')
                X.append(self.price_category.index(line_split[0].strip()))
                X.append(self.price_category.index(line_split[1].strip()))
                X.append(self.doors_category.index(line_split[2].strip()))
                X.append(self.persons_category.index(line_split[3].strip()))
                X.append(self.lugboot_category.index(line_split[4].strip()))
                X.append(self.safety_category.index(line_split[5].strip()))
                Y.append(self.label_category.index(line_split[6].strip()))
                data.append((X,Y))
        return data

    def __init__(self):
        data = self._readFile()

        # 3-fold validation
        self.trainingData=[]
        self.trainNum = math.ceil(len(data)*0.9)
        for i in range(int(self.trainNum)):
            index = random.randint(0, len(data) - 1)
            self.trainingData.append(data.pop(index))
        self.testingData=data

        #initials
        self.classNum=4
        self.label_N = [0, 0, 0, 0]
        self.buying_N = self.maint_N = self.doors_N = [[0 for i in range(self.classNum)] for i in range(4)]
        self.persons_N = self.lugboot_N = self.safety_N = [[0 for i in range(self.classNum)] for i in range(3)]

    def _hypothesis(self,X):
        label_prob=[0,0,0,0]
        #calculate probability for each label
        for index in range(self.classNum):
            posterior=1
            pro=[]
            pro.append(self.buying_N[X[0]][index]/self.label_N[index])
            pro.append(self.maint_N[X[1]][index] / self.label_N[index])
            pro.append(self.doors_N[X[2]][index]/ self.label_N[index])
            pro.append(self.persons_N[X[3]][index] / self.label_N[index])
            pro.append(self.lugboot_N[X[4]][index] / self.label_N[index])
            pro.append(self.safety_N[X[5]][index] / self.label_N[index])
            for each_prob in pro:
                posterior*=each_prob
            label_prob[index]=posterior*(self.label_N[index]/len(self.trainingData))
        return label_prob.index(max(label_prob))

    def training(self):
        # traverse all samples,compute frequency
        for sample in self.trainingData:
            currentX = sample[0]
            currentY = sample[1][0]
            self.label_N[currentY] += 1
            self.buying_N[currentX[0]][currentY] += 1
            self.maint_N[currentX[1]][currentY] += 1
            self.doors_N[currentX[2]][currentY] += 1
            self.persons_N[currentX[3]][currentY] += 1
            self.lugboot_N[currentX[4]][currentY] += 1
            self.safety_N[currentX[5]][currentY] += 1

    def testing(self):
        error=0
        for each in self.testingData:
            currentX=each[0]
            currentY=each[1]
            if self._hypothesis(currentX)!=currentY[0]:
                error+=1
        return error/len(self.testingData)


if __name__ == '__main__':
    total=[]
    print("Randomly pick training set for 1000 times:")
    for i in range(1000):
        model=CarClassify()
        model.training()
        total.append(model.testing())
    print("Everage :",sum(total)/1000)
    print("Min :",min(total))
