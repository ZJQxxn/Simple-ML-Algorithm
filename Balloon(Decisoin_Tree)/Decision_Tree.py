'''
Description:
    Decision tree.
    
Author:
    Bill Roland
    
Date:
    2018/3/4
'''
import numpy as np
import TreeNode

class DecisionTree:

    def __init__(self):
        #Read data from file
        self.data=[]
        with open("adult_stretch.data",'r') as file:
            for line in file.readlines():
                attr=line.split(',')
                self.data.append((attr[:4],attr[4].strip()))
        #Other initialization
        self.root=TreeNode.TreeNode()
        self.attrSet=[i for i in range(4)]

    def _shannoEnt(self,dataSet):
        '''
        Description:
            Calculate information entropy for a dataset. 
        :param dataSet: 
        :return: 
        '''
        lable_dict={}
        entropy=0.0
        for each in dataSet:
            if each[1] not in lable_dict.keys():
                lable_dict[each[1]]=1
            else:
                lable_dict[each[1]]+=1
        for lable in lable_dict.keys():
            prob=lable_dict[lable]/len(dataSet)
            entropy+=prob*np.log2(prob)
        return -entropy

    def _infor_gain(self,dataSet,attr):
        '''
        Description:
            Calculate information gain.
        :param dataSet: Data set
        :param attr: Atrribute used for splitting dataset.
        :return: Information gain on this dataset.
        '''
        befor_entropy=self._shannoEnt(dataSet)
        attr_dict={}
        for each in dataSet:
            if each[0][attr] not in attr_dict:
                attr_dict[each[0][attr]]=[each]
            else:
                attr_dict[each[0][attr]].append(each)
        after_entropy=sum([len(attr_dict[each])/len(dataSet)
                           *self._shannoEnt(attr_dict[each]) for each in attr_dict])
        return befor_entropy-after_entropy

    def _generateTree(self,cur_node,dataSet,attrSet):
        '''
        Description:
            Private function to generate decision tree recurssively.
        :param cur_node: Current tree node
        :param dataSet: Current data set
        :param attrSet: Current attribute set
        :return: Return leaf node when reach to it
        '''
        lable_dict={}
        all_same=True
        flag_sample=dataSet[0]
        for sample in dataSet:
            #Count sample number of each class
            if sample[1] not in lable_dict:
                lable_dict[sample[1]]=1
            else:
                lable_dict[sample[1]]+=1
            #Determine whether samples are same
            if all_same:
                for attr in attrSet:
                    if sample[0][attr]!=flag_sample[0][attr]:
                        all_same=False
        #No attribute left
        if len(attrSet)==0:
            lable=list(lable_dict.keys())[0]
            for each in lable_dict.keys():
                if lable_dict[each]>lable_dict[lable]:
                    lable=each
            cur_node.setLeaf(True)
            cur_node.setLable(lable)
            return cur_node
        #All data belongs to one class or all samples are the same
        if len(lable_dict.keys())==1 or all_same:
            cur_node.setLeaf(True)
            cur_node.setLable(list(lable_dict.keys())[0])
            return cur_node
        #Select attribute with the most information gain
        entropys=[self._infor_gain(dataSet,each) for each in attrSet]
        selected_attr=attrSet[entropys.index(max(entropys))]
        #Split dataset on this selected attribute and generate subtree for each value
        attr_dict={}
        for sample in dataSet:
            if sample[0][selected_attr] not in attr_dict:
                attr_dict[sample[0][selected_attr]]=[sample]
            else:
                attr_dict[sample[0][selected_attr]].append(sample)
        for value in attr_dict:
            tempSet=attrSet.copy()
            tempSet.remove(selected_attr)
            cur_node.setAttrNum(selected_attr)
            cur_node.addChild(TreeNode.TreeNode(),value)
            self._generateTree(cur_node.children[value], attr_dict[value], tempSet)

    def _estimate(self,sample):
        '''
        Description:
            Private function to estimate a sample usign decision tree.
        :param sample: 2-tuple sample
        :return: Sample lable
        '''
        cur_node=self.root
        while not cur_node.isLeaf():
            cur_node=cur_node.getChildren()[sample[cur_node.getAttrNum()]]
        return cur_node.getLable()

    def generateTree(self):
        '''
        Description:
            Generate a decision tree.
        :return: 
        '''
        self._generateTree(self.root,self.data,self.attrSet)

    def testing(self):
        '''
        Description:
            Test this model and show result.
        :return: void
        '''
        error=0
        for sample in self.data:
            if sample[1]!=self._estimate(sample[0]):
                error+=1
        print("%d error in %d testing samples, error rate is %.2f"
              %(error,len(self.data),error/len(self.data)))


if __name__ == '__main__':
    decTree=DecisionTree()
    decTree.generateTree()
    decTree.testing()
