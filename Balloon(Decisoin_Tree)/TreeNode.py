'''
Description:
    Node for decision tree.
    
Author:
    Bill Roland
    
Date:
    2018/3/4
'''

class TreeNode:

    def __init__(self,leaf=False,lable=None,attrNum=None):
        self.leaf=leaf
        self.classLable = lable
        self.attrNum = attrNum
        self.children = {}

    def getChildren(self):
        return self.children

    def addChild(self,child,attr_value=None):
        self.children[attr_value]=child

    def setLeaf(self,flag):
        self.leaf=flag

    def setLable(self,lable):
        if self.leaf==True:
            self.classLable=lable

    def setAttrNum(self,num):
        if  self.leaf==False:
            self.attrNum=num

    def isLeaf(self):
        return self.leaf

    def getAttrNum(self):
        return self.attrNum

    def getLable(self):
        return self.classLable



if __name__ == '__main__':
    pass
