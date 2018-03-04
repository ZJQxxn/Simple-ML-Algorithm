'''
    Description: 
        Mixture of gaussian cluster.
        Because the parameters are initialized randomly,
        so the clustering result are various.
    
    Author:
        zjq_1112@qq.com
        
    Date:
        2017.12.17
    
    TODO:
        Modify '_randfile()' so that it can read file given filename.
'''

import matplotlib.pyplot as plt
import numpy as np
import random


class MoGCluster:
    '''
    Description: 
        Mixture of gaussian cluster.
    
    Attributes:
        self.data:Store all the samples.
        self.clusterNum: Number of clusters.
        self.featureNum: Number of features.
        self.epochs: Max iterate epochs.
        self.alpha: Probabilities for each gaussian distribution.
        self.means: Mean for each gaussian distribution.
        self.sigma: Covariance for each gaussian distribution.
        self.posterior: P(z|x)
        
    Main methods:
        _gaussian: Calculate gaussian probabilities given sample value
                ,mean and covariance.
        _likelihood: Calculate current likelihood.
        _cluster: Clustering a sample.
        training: Train model until convergence.        
    '''
    def  _readFile(self):
        '''
        Description:Read data from file.
        :return: void
        '''
        self.data = []
        with open('WaterMelon_for_Cluster.txt', 'r') as file:
            for line in file:
                line_split = line.split()
                self.data.append(np.array([float(line_split[1]), float(line_split[2])]))

    def __init__(self,clusterNum,featureNum,epoch=100):
        self._readFile()
        self.sampleNum = len(self.data)
        self.clusterNum=clusterNum
        self.featureNum=featureNum
        self.epochs=epoch
        # Initialize parameters
        self.alpha=[1/self.clusterNum for i in range(self.clusterNum)]
        index=[]
        while len(index)<3:
            temp=random.randint(1,self.sampleNum-1)
            if temp not in index:
                index.append(temp)
        self.means=[self.data[each] for each in index]
        self.sigma=[np.matrix([[1.0,0.0],[0.0,1.0]],dtype='float64') for i in range(self.clusterNum)]
        self.posterior=np.zeros((self.sampleNum,self.clusterNum),dtype='float64')

    def _gaussian(self,X,mean,sigma):
        '''
        Description: Calculate gaussian probability.
        :param X: X value.
        :param mean: Mean for gaussian diatribution.
        :param sigma: Covariance for gaussian distribution.
        :return: Gaussian probability. 
        '''
        temp=(np.sqrt(2*np.pi)**self.featureNum)*np.sqrt(np.linalg.det(np.matrix(sigma)))
        return np.exp(-0.5*np.matrix(X-mean)*np.matrix(sigma).I*np.matrix(X-mean).T)\
               *(1.0/temp)

    def _likelihood(self):
        result=0.0
        for sample_index in range(self.sampleNum):
            temp=0.0
            for cluster_index in range(self.clusterNum):
                temp+=np.log2(self.alpha[cluster_index]*
                       self._gaussian(self.data[sample_index],self.means[cluster_index],self.sigma[cluster_index]))
            result+=temp
        return result

    def _cluster(self,X):
        '''
        Given X, return label of this sample.
        :param X: X features array.
        :return: Sample label.
        '''
        prob=[]
        for cluster in range(self.clusterNum):
            prob.append(self._gaussian(X,self.means[cluster],self.sigma[cluster]))
        label=0
        for i in range(self.clusterNum):
            if prob[i]>prob[label]:
                label=i
        return label

    def training(self):
        '''
        Description: Trainning model by using EM algorithm.
        :return: void
        '''
        #EM algorithm
        pre_likeli=1
        epoch=0
        #loop until likelihood doesn;t change too much or reach at maximize epochs
        while (True in (np.abs((self._likelihood()-pre_likeli)/pre_likeli)>0.001)) \
                and epoch<self.epochs:
            pre_likeli=self._likelihood()
            # E-step
            #gaussian_sum=[0.0 for i in range(self.clusterNum)]
            for sample_index in range(self.sampleNum):
                for cluster_index in range(self.clusterNum):
                    self.posterior[sample_index,cluster_index]=\
                        self.alpha[cluster_index]*self._gaussian(self.data[sample_index],self.means[cluster_index],self.sigma[cluster_index])
                self.posterior[sample_index]=[self.posterior[sample_index,j]/np.sum(self.posterior[sample_index]) for j in range(self.clusterNum) ]

            # M-step
            posterior_sum=[0.0 for i in range(self.clusterNum)]
            mean_sum = [np.array([0.0 for i in range(self.featureNum)]) for i in range(self.clusterNum)]
            # update alpha and mean
            for cluster_index in range(self.clusterNum):
                for sample_index in range(self.sampleNum):
                    posterior_sum[cluster_index]+=self.posterior[sample_index,cluster_index]
                    mean_sum[cluster_index]+=self.data[sample_index]*self.posterior[sample_index,cluster_index]
                self.alpha[cluster_index]=posterior_sum[cluster_index]/self.sampleNum
                self.means[cluster_index]=np.nan_to_num(mean_sum[cluster_index]/posterior_sum[cluster_index])
            #update covariance.
            sigma_sum = [np.zeros((self.featureNum, self.featureNum)) for i in range(self.clusterNum)]
            for cluster_index in range(self.clusterNum):
                for sample_index in range(self.sampleNum):
                    error = self.data[sample_index] - self.means[cluster_index]
                    sigma_sum[cluster_index] += self.posterior[sample_index, cluster_index] * np.matrix(error).T * np.matrix(error)
                self.sigma[cluster_index]=np.nan_to_num(sigma_sum[cluster_index]/posterior_sum[cluster_index])
            #generalization
            #self.alpha=[np.nan_to_num(self.alpha[i]/sum(self.alpha)) for i in range(self.clusterNum)]
            epoch+=1
        print("Training finished in %s epochs" % epoch)


def showData(cluster):
        '''
        Description: Show data with label.
        :return: void
        '''
        labels=[[] for i in range(cluster.clusterNum)]
        for each in cluster.data:
            labels[cluster._cluster(each)].append(each)
        #plot data
        plt.title('Samples after mixture gaussian clustering')
        plt.plot([each[0] for each in labels[0]],[each[1] for each in labels[0]],'r*',label='First Cluster')
        plt.plot([each[0] for each in labels[1]],[each[1] for each in labels[1]],'g*',label='Second Cluster')
        plt.plot([each[0] for each in labels[2]],[each[1] for each in labels[2]],'b*',label='Third Cluster')
        #plot means
        plt.plot([each[0] for each in cluster.means],[each[1] for each in cluster.means],'yo',label='Means')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    cluster=MoGCluster(clusterNum=3,featureNum=2)
    cluster.training()
    print("Means:\n",cluster.means)
    print("Alpha:\n",cluster.alpha)
    print("sigma:\n",cluster.sigma)
    showData(cluster)