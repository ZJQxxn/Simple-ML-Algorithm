'''
    Description: 
            A 3-means cluster.
    
    Author:
            zjq_1112@qq.com
            
    Date:
            2017.12.14
'''

import matplotlib.pyplot as plt
import numpy as np
import random


class Cluster:
    '''
        Description:
            K-means cluster.
                
        Attributes:
            self.data:A list of 2-tuple contains feartures of each sample.
            self.sampleNum: Number of samples.
            self.clusterNum: Number of clusters.
            self.featureNum: Number of features.
            self.epochs: Limit epochs.
            self.clusters: Temporarily store samples by its cluster.
            self.means: Mean for each cluster.
            
        Main methods:
            training: Train model to find mean for each cluster.
            showData: Show data.
            
    '''

    def readFile(self):
        '''
        Description:
            Read data from file:'WaterMelon_for_Cluster.txt'.
        :return: void
        '''
        self.data=[]
        with open('WaterMelon_for_Cluster.txt','r') as file:
            for line in file:
                line_split=line.split()
                self.data.append(np.array([float(line_split[1]),float(line_split[2])]))


    def __init__(self):
        self.readFile()
        self.sampleNum=len(self.data)
        self.clusterNum = 3
        self.featureNum = 2
        self.epochs=200
        self.clusters=[[] for i in range(self.clusterNum)]#store samples of each cluster

        #randomly choose sample as means
        index=[]
        while len(index)<3:
            temp=random.randint(0,len(self.data))
            if temp not in index:
                index.append(temp)
        self.means=np.array([self.data[i] for i in index])


    def _clustering(self):
        '''
        Description: K-means clustering.
        :return: 
        '''
        #clear all clusters
        self.clusters = [[] for i in range(self.clusterNum)]

        #update label for each sample
        for index in range(self.sampleNum):
            sample=self.data[index]
            dist=[]
            #calculate distance between sample and each mean
            for cluster in range(self.clusterNum):
                dist.append(np.sqrt(np.sum((self.means[cluster]-sample)**2)))
            #set index of maximize distance as label of this sample
            maximize=0
            for i in range(1,self.clusterNum):
                if dist[i]>dist[maximize]:
                    maximize=i
            self.clusters[maximize].append(sample)

        #update each means
        for cluster_index in range(self.clusterNum):
            self.means[cluster_index]=np.array([
                np.mean(np.array([each[0] for each in self.clusters[cluster_index]])),
                np.mean(np.array([each[1] for each in self.clusters[cluster_index]]))
            ])


    def training(self):
        '''
        Description:Train this model to get means.
        :return: void
        '''
        for epoch in range(self.epochs):
            self._clustering()


    def showData(self):
        '''
        Description:Show data in figure.
        :return: void
        '''
        plt.plot([each[0] for each in self.clusters[0]],[each[1] for each in self.clusters[0]],'r*',label='First Cluster')
        plt.plot([each[0] for each in self.clusters[1]],[each[1] for each in self.clusters[1]],'b*',label='Second Cluster')
        plt.plot([each[0] for each in self.clusters[2]],[each[1] for each in self.clusters[2]],'g*',label='Third Cluster')
        plt.plot([each[0] for each in self.means],[each[1] for each in self.means],'y^',label='Cluster Mean')
        plt.title('3-means Clustering')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    cluster=Cluster()
    cluster.training()
    cluster.showData()