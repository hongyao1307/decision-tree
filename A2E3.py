import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt

class treeNode:
    
    def __init__(self):
        self.leftchild=None
        self.rightchild=None
        self.value=None #featrue & result
        self.feature=None
        self.threshold=None

def get_value(y):
    yes=0
    for i in range(len(y)):
        if y[i]==1:
            yes+=1
    p_yes=yes/len(y)
    if p_yes>=0.5:
        value=1
    else:
        value=0
    return value

def split(x,y,threshold,feature):
    xleft=[]
    yleft=[]
    xright=[]
    yright=[]
    for i in range(len(x)):
        if x[i][feature]<=threshold:
            xleft.append(x[i])
            yleft.append(y[i])
        else:
            xright.append(x[i])
            yright.append(y[i])
            
    return xleft,yleft,xright,yright

def lossEn_select(x,y,var_index,feat_index):
    yes1=0
    yes2=0
    num1=0
    num2=0
    loss1=0
    loss2=0
    for i in range(len(x)):
        if x[i][feat_index]<=x[var_index][feat_index]:
            num1+=1
            if y[i]==1:
                yes1+=1
        else:
            num2+=1
            if y[i]==1:
                yes2+=1
        
    if yes1!=0 and (1-yes1/num1)!=0:
        loss1=num1*(-1*((yes1/num1)*math.log2(yes1/num1)+(1-yes1/num1)*math.log2(1-yes1/num1)))
    if yes2!=0 and (1-yes2/num2)!=0:
        loss2=num2*(-1*((yes2/num2)*math.log2(yes2/num2)+(1-yes2/num2)*math.log2(1-yes2/num2)))
    loss=(loss1+loss2)/(num1+num2)
    return loss

def lossCla_select(x,y,var_index,feat_index):
    yes1=0
    yes2=0
    num1=0
    num2=0
    loss1=0
    loss2=0
    p1=0
    p2=0
    for i in range(len(x)):
        if x[i][feat_index]<=x[var_index][feat_index]:
            num1+=1
            if y[i]==1:
                yes1+=1
        else:
            num2+=1
            if y[i]==1:
                yes2+=1
    if num1!=0:
        p1=yes1/num1
    if num2!=0:
        p2=yes2/num2
    loss1=num1*min(p1,1-p1)/(num1+num2)
    loss2=num2*min(p2,1-p2)/(num1+num2)
    loss=loss1+loss2
    return loss

def lossGini_select(x,y,var_index,feat_index):
    yes1=0
    yes2=0
    num1=0
    num2=0
    loss1=0
    loss2=0
    p1=0
    p2=0
    for i in range(len(x)):
        if x[i][feat_index]<=x[var_index][feat_index]:
            num1+=1
            if y[i]==1:
                yes1+=1
        else:
            num2+=1
            if y[i]==1:
                yes2+=1
    if num1!=0:
        p1=yes1/num1
    if num2!=0:
        p2=yes2/num2
    loss1=num1*p1*(1-p1)/(num1+num2)
    loss2=num2*p2*(1-p2)/(num1+num2)
    loss=loss1+loss2
    return loss

def lossEn_check(y):#var=the part set of variables of one featrue 
    loss=0
    pure=-1
    yes=0
    no=0
    for i in range(len(y)):
        if y[i]==1:
            yes+=1
        else:
            no+=1
    p_yes=yes/len(y)
    p_no=no/len(y)
    if p_yes!=0 and p_no!=0:
        loss=-1*p_yes*math.log2(p_yes)-p_no*math.log2(p_no)
    if p_yes==1:
        pure=1
    if p_no==1:
        pure=0
    return loss,pure

def lossCla_check(y):#var=the part set of variables of one featrue 
    loss=0
    pure=-1
    yes=0
    no=0
    for i in range(len(y)):
        if y[i]==1:
            yes+=1
        else:
            no+=1
    p_yes=yes/len(y)
    p_no=no/len(y)
    loss=min(p_yes,p_no)
    if p_yes==1:
        pure=1
    if p_no==1:
        pure=0
    return loss,pure

def lossGini_check(y):#var=the part set of variables of one featrue 
    loss=0
    pure=-1
    yes=0
    no=0
    for i in range(len(y)):
        if y[i]==1:
            yes+=1
        else:
            no+=1
    p_yes=yes/len(y)
    p_no=no/len(y)
    loss=p_yes*p_no
    if p_yes==1:
        pure=1
    if p_no==1:
        pure=0
    return loss,pure

def findMinEn(x,y,block_featrue):#check
    var_index=0
    feat_index=0
    threshold=0
    minEn=float("inf") 
    for j in range(len(x[0])):
        if j != block_featrue:
            for i in range(len(x)):
                calculateEn=lossEn_select(x,y,i,j)
                if calculateEn<=minEn:
                    minEn=calculateEn
                    var_index=i
                    feat_index=j
                    threshold=x[i][j]
    return minEn,var_index,feat_index,threshold

def findMinCla(x,y,block_feature):#check
    var_index=0
    feat_index=0
    threshold=0
    minCla=float("inf") 
    for j in range(len(x[0])):
        if j != block_feature:
            for i in range(len(x)):
                calculateCla=lossCla_select(x,y,i,j)
                if calculateCla<=minCla:
                    minCla=calculateCla
                    var_index=i
                    feat_index=j
                    threshold=x[i][j]
    return minCla,var_index,feat_index,threshold

def findMinGini(x,y,block_featrue):#check
    var_index=0
    feat_index=0
    threshold=0
    minGini=float("inf") 
    for j in range(len(x[0])):
        if j != block_featrue:
            for i in range(len(x)):
                calculateGini=lossGini_select(x,y,i,j)
                if calculateGini<=minGini:
                    minGini=calculateGini
                    var_index=i
                    feat_index=j
                    threshold=x[i][j]
    return minGini,var_index,feat_index,threshold

def buildEn(x,y,block_feature,depth,maxDepth):
    root=treeNode()
    if maxDepth==0:
        root.value=get_value(y)
    if depth>=maxDepth:
        root.value=get_value(y)
        return root
   
    entropy,pure=lossEn_check(y)

    if entropy==0:
        if pure==1:
            root.value=1
        if pure==0:
            root.value=0
        return root    
    
    root.value=get_value(y)
    
    minEn,var_index,feat_index,threshold=findMinEn(x,y,block_feature)

    if 0.000001<=entropy-minEn:
        root.feature=feat_index
        root.threshold=threshold
        block_feature=feat_index
        xleft,yleft,xright,yright=split(x,y,threshold,feat_index)
        depth+=1
        temp_left=buildEn(xleft,yleft,block_feature,depth,maxDepth)
        root.leftchild=temp_left
        temp_right=buildEn(xright,yright,block_feature,depth,maxDepth)
        root.rightchild=temp_right
    else:
        return root
               
    return root

def buildCla(x,y,block_feature,depth,maxDepth):
    root=treeNode()
    if maxDepth==0:
        root.value=get_value(y)
    if depth>=maxDepth:
        root.value=get_value(y)
        return root
   
    misclass,pure=lossCla_check(y)

    if misclass==0:
        if pure==1:
            root.value=1
        if pure==0:
            root.value=0
        return root    
    
    root.value=get_value(y)
    
    minCla,var_index,feat_index,threshold=findMinCla(x,y,block_feature)

    if 0.000001<=misclass-minCla:
        root.feature=feat_index
        root.threshold=threshold
        block_feature=feat_index
        xleft,yleft,xright,yright=split(x,y,threshold,feat_index)
        depth+=1
        temp_left=buildCla(xleft,yleft,block_feature,depth,maxDepth)
        root.leftchild=temp_left
        temp_right=buildCla(xright,yright,block_feature,depth,maxDepth)
        root.rightchild=temp_right
    else:

        return root
               
    return root

def buildGini(x,y,block_feature,depth,maxDepth):
    
    root=treeNode()
    if maxDepth==0:
        root.value=get_value(y)
    if depth>=maxDepth:
        root.value=get_value(y)
        return root
    Gini_loss,pure=lossGini_check(y)

    if Gini_loss==0:
        if pure==1:
            root.value=1
        if pure==0:
            root.value=0
        return root    
    
    root.value=get_value(y)
    minGini,var_index,feat_index,threshold=findMinGini(x,y,block_feature)

    if 0.000001<=Gini_loss-minGini:
        root.feature=feat_index
        root.threshold=threshold
        block_feature=feat_index
        xleft,yleft,xright,yright=split(x,y,threshold,feat_index)
        depth+=1
        temp_left=buildGini(xleft,yleft,block_feature,depth,maxDepth)
        root.leftchild=temp_left
        temp_right=buildGini(xright,yright,block_feature,depth,maxDepth)
        root.rightchild=temp_right
    else:

        return root
                        
    return root

def predict(root,xtest_i):
    if root.leftchild==None and root.leftchild==None:
        if root.value==1:
            return 1
        if root.value==0:
            return 0
    if xtest_i[root.feature]<=root.threshold:
        if root.leftchild!=None:
            return predict(root.leftchild,xtest_i)
    if xtest_i[root.feature]>root.threshold:
        if root.rightchild!=None:
            return predict(root.rightchild,xtest_i)

def cal_acc(predict_y,ytest):
    match=0
    for i in range(len(predict_y)):
        if predict_y[i]==ytest[i]:
            match+=1
    acc=match/len(ytest)
    return acc

def main():
    
    pathy=r"./a2_files/decision_tree data/data/y_train.csv"
    y=np.genfromtxt(pathy, delimiter=',')
    y = y[:, None]
    pathx=r"./a2_files/decision_tree data/data/X_train.csv"
    x=np.genfromtxt(pathx, delimiter=',')

    pathy=r"./a2_files/decision_tree data/data/y_test.csv"
    ytest=np.genfromtxt(pathy, delimiter=',')
    ytest = ytest[:, None]
    pathx=r"./a2_files/decision_tree data/data/X_test.csv"
    xtest=np.genfromtxt(pathx, delimiter=',')
    
    xtrain=copy.deepcopy(x)
    ytrain=copy.deepcopy(y)
    

    depth_arr=[0,1,2,3,4,5,6,7,8,9,10]
    depth=0
    
#entropy
    acc_testEn=[]
    acc_trainEn=[]
    for i in range(len(depth_arr)):
        root=buildEn(x,y,-1,depth,depth_arr[i])
        y_testEn=[]
        y_trainEn=[]
        for j in range(len(xtest)):
            xtest_i=xtest[j][:, None]
            y_testEn.append(predict(root,xtest_i))
        acc_testEn.append(cal_acc(y_testEn,ytest))
        for m in range(len(x)):
            xtrain_i=xtrain[m][:, None]
            y_trainEn.append(predict(root,xtrain_i))
        acc_trainEn.append(cal_acc(y_trainEn,ytrain))
    plt.title('decision tree accuracy by using Entropy with different depth')
    plt.ylabel('accuracy')
    plt.xlabel('depth')
    plt.plot(depth_arr,acc_testEn,label='test accuracy')
    plt.legend()
    plt.plot(depth_arr,acc_trainEn,label='train accuracy')
    plt.legend()
    plt.show()
    
#misclass
    acc_testCla=[]
    acc_trainCla=[]
    for i in range(len(depth_arr)):
        root=buildCla(x,y,-1,depth,depth_arr[i])
        y_testCla=[]
        y_trainCla=[]
        for j in range(len(xtest)):
            xtest_i=xtest[j][:, None]
            y_testCla.append(predict(root,xtest_i))
        acc_testCla.append(cal_acc(y_testCla,ytest))
        for m in range(len(x)):
            xtrain_i=xtrain[m][:, None]
            y_trainCla.append(predict(root,xtrain_i))
        acc_trainCla.append(cal_acc(y_trainCla,ytrain))
    plt.title('decision tree accuracy by using Misclassification error with different depth')
    plt.ylabel('accuracy')
    plt.xlabel('depth')
    plt.plot(depth_arr,acc_testCla,label='test accuracy')
    plt.legend()
    plt.plot(depth_arr,acc_trainCla,label='train accuracy')
    plt.legend()
    plt.show()
    
#Gini
    acc_testGini=[]
    acc_trainGini=[]
    for i in range(len(depth_arr)):
        root=buildGini(x,y,-1,depth,depth_arr[i])
        y_testGini=[]
        y_trainGini=[]
        for j in range(len(xtest)):
            xtest_i=xtest[j][:, None]
            y_testGini.append(predict(root,xtest_i))
        acc_testGini.append(cal_acc(y_testGini,ytest))
        for m in range(len(x)):
            xtrain_i=xtrain[m][:, None]
            y_trainGini.append(predict(root,xtrain_i))
        acc_trainGini.append(cal_acc(y_trainGini,ytrain))
    plt.title('decision tree accuracy by using Gini coefficient with different depth')
    plt.ylabel('accuracy')
    plt.xlabel('depth')
    plt.plot(depth_arr,acc_testGini,label='test accuracy')
    plt.legend()
    plt.plot(depth_arr,acc_trainGini,label='train accuracy')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
