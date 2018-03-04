'''
Description:
    Implement Gaussian Discrimnant Analysis on binary classes.
    
Author:
    zjq_1112@qq.com
    
Date:
    2017.11.1
'''
import numpy as np
import matplotlib.pyplot as plt


class GDA:
    '''
    Description:
        Class for training GDA model and estimate new sample.
        Assume y is Bernoulli Distribution, and each sample
        in different is Gausian Distribution.
    Attributes:
        self.trainingData:2-tuple training data.
        self.testingData:2-tuple testing data
        self.sampleNum:Number of total samples.
        self.phi:Parameter of Bernoulli distribution.
        self.theta_0:Mean of Gaussian Distribution of class 0.
        self.theta_1:Mean of Gaussian Distribution of class 1.
        self.sigma:Covariance for Gaussian Distributin.
    Method:
        self.training:Training GDA model on training data.
        self.testing:Testing model given testing data.
        self.draw:Draw points and boundry.
        self._gaussian:Calculate P(x|y) given x and y.
        self._estimate:Estimate the class of a data given attributes.
        self._drawPoint:Draw all the data points.
        self._drawBoundry:Draw th boundry of two classes.
    '''
    def __init__(self):
        '''
        Read data from file and init class.
        '''
        sampleX=[]
        sampleY=[]
        with open('data.txt','r') as file:
            for line in file.readlines():
                line_split=line.split()
                sampleX.append([float(line_split[0]),float(line_split[1])])
                sampleY.append(int(line_split[2]))
        self.trainingData=(np.array(sampleX),np.array(sampleY))
        self.testingData=(np.array(sampleX),np.array(sampleY))
        self.sampleNum=len(self.trainingData[0])


    def _drawPoint(self):
        '''
        Draw points on a diagram.
        :return: void
        '''
        #split points into two class
        first_class=[[],[]]
        second_class=[[],[]]
        for index in range(self.sampleNum):
            if self.trainingData[1][index]==0:
                first_class[0].append(self.trainingData[0][index][0])
                first_class[1].append(self.trainingData[0][index][1])
            elif self.trainingData[1][index]==1:
                second_class[0].append(self.trainingData[0][index][0])
                second_class[1].append(self.trainingData[0][index][1])
        #draw points
        plt.plot(first_class[0],first_class[1],'ro',label='First class')
        plt.plot(second_class[0],second_class[1],'b*',label='Second class')
        plt.xlabel('First attribute')
        plt.ylabel('second attribute')
        plt.title('GDA Classification')
        #plt.plot([self.theta_0[0]],[self.theta_0[1]],'g^')
        #plt.plot([self.theta_1[0]], [self.theta_1[1]], 'g^')
        plt.legend()


    def _drawBoundry(self):
        '''
        Draw clas boundry for these points.
        Boundry means P(x|y=0)==P(x|y=1), ie.
        the probability of this data belongs 
        to either class is equal.
        :return: void
        '''
        k=1/(self.theta_0[1]-self.theta_1[1])/(self.theta_0[0]-self.theta_1[0])
        midPoint=[(self.theta_0[0]+self.theta_1[0])/2,(self.theta_0[1]+self.theta_1[1])/2]
        b=midPoint[1]-midPoint[0]*k
        x=[each for each in range(-4,4)]
        y=[each*k+b for each in x]
        plt.plot(x,y,'g-')


    def _gaussian(self,x_attributes,y=1):
        '''
        Calculate P(x|y) given different y.
        :param y: value of y
        :return: probability of P(x|y)
        '''
        #initialize some value for multi-gaussian distribution
        theta=np.matrix([0.0,0.0])
        coef=1/(2*np.pi*np.sqrt(np.linalg.det(self.sigma)))
        if y==0:
            theta=self.theta_0
        elif y==1:
            theta=self.theta_1
        #calculate probability
        error=x_attributes-theta
        probability=coef*np.exp((-0.5)*(error.dot(np.linalg.inv(self.sigma)).dot(error.T)))
        return probability


    def _estimate(self,x_attributes):
        '''
        Classify a new sample given its attributes.
        :param x_attributes: An array of attributes
        :return: int    The number of its class.
        '''
        probal_ratio=self._gaussian(x_attributes,y=0)/self._gaussian(x_attributes,y=1)
        prob=1/(1+probal_ratio)
        if(prob>0.5):
            return 1
        else:
            return 0


    def draw(self):
        '''
        Draw all the samples and boundry in a diagram.
        :return: void
        '''
        self._drawPoint()
        self._drawBoundry()
        plt.show()


    def training(self):
        '''
        Training model on sample data.
        Get parameters of distributions.
        :return: void
        '''
        sampleX=self.trainingData[0]
        sampleY=self.trainingData[1]
        label1_count=0   #number of sample which are labeled class 1
        label0_count=0   #number of sample which are labeled class 1
        label1_xsum=np.array([0.0,0.0])    #sumption of x of sample which is labeled class 0
        label0_xsum=np.array([0.0,0.0])    #sumption of x of sample which is labeled class 1
        for index in range(self.sampleNum):
            if sampleY[index]==0:
                label0_count+=1
                label0_xsum+=sampleX[index]
            elif sampleY[index]==1:
                label1_count+=1
                label1_xsum+=sampleX[index]

        #calculate parameters for distributions
        self.phi=label1_count/self.sampleNum
        self.theta_0=label0_xsum/label0_count
        self.theta_1=label1_xsum/label1_count

        #calculate covariance
        temp=np.matrix([[0.0,0.0],[0.0,0.0]])
        theta=np.matrix([0.0,0.0])
        for index in range(self.sampleNum):
            currentX=np.matrix(sampleX[index])
            currentY=sampleY[index]
            if currentY==0:
                theta=self.theta_0
            elif currentY==1:
                theta = self.theta_1
            error=currentX-theta
            temp =temp+ error.T*error
        self.sigma=temp/self.sampleNum


    def testing(self,testingdata):
        '''
        Test this model on testing data.Print testing result.
        :param testingdata: 2-tuple contains attributes-array and class-number
        :return: void
        '''
        error=0
        testingX=testingdata[0]
        testingY=testingdata[1]
        testingNum=len(testingY)
        for index in range(testingNum):
            estimateY=self._estimate(testingX[index])
            if estimateY!=testingY[index]:
                error+=1
        print("%d error in %d data, error percentage is %f"
              %(error,testingNum,(error/testingNum)*100))



if __name__ == '__main__':
    model=GDA()
    model.training()
    print("Sigma :\n",model.sigma)
    print("Theta 0 :",model.theta_0)
    print("Theta 1 :",model.theta_1)
    print("Phi : ",model.phi)
    model.testing(model.testingData)
    model.draw()






