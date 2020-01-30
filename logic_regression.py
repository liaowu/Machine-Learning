import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
path='data'+os.sep+'LogiReg_data.txt'
pdData=pd.read_csv(path,header=None,names=['Exam1','Exam2','Admitted'])
#pdData.head()
positive=pdData[pdData['Admitted']==1]
negative=pdData[pdData['Admitted']==0]
fig,ax=plt.subplots(figsize=(10,3))
ax.scatter(positive["Exam1"],positive["Exam2"],s=30,c='b',marker='o',label="Admitted")
ax.scatter(negative["Exam1"],negative["Exam2"],s=30,c='r',marker='x',label="Not Admitted")
ax.legend()
ax.set_xlabel("Exam 1 Score")
ax.set_ylabel("Exam 2 Score")

def sigmoid(z):
	return 1/(1+np.exp(-z))

nums=np.arange(-10,10,step=1) #return a vector containing 20 equally spaced value from -10 to 10
fig, ax=plt.subplots(figsize=(12,4))
ax.plot(nums,sigmoid(nums),'r')

def model(X,theta):  #预测函数
	return sigmoid(np.dot(X,theta.T))

pdData.insert(0,'Ones',1) # 新加一列名为'Ones',值全为1; in a try /except structure so as not to return an error if the block  

#Set X (training data) and y (target variable)
orig_data=pdData.as_matrix() #convert the pandas representation of the data to an array useful for further
cols=orig_data.shape[1]
X=orig_data[:,0:cols-1]    #training data  
y=orig_data[:,cols-1:cols]  #target

#convert to numpy arrays and initialize the parameter array theta

#X=np.matrix(X.values)

#y=np.matrix(data,iloc[:,3:4].values) #np.array(y.values)


theta=np.zeros([1,3]) #构造一个空的矩阵
#verify if data are correct: 
#X[:5]
#print(y)
#theta
#X.shape,y.shape,theta.shape

def cost(X,y,theta):
	left=np.multiply(-y,np.log(model(X,theta)))   #减号左边 
	right=np.multiply(1-y,np.log(1-model(X,theta)))  #减号右边
	return np.sum(left-right)/(len(X))

#cost(X,y,theta)

def gradient(X,y,theta):
	grad=np.zeros(theta.shape)  #zero占位
	error=(model(X,theta)-y).ravel()
	for j in range(len(theta.ravel())):  #for each parameter对每个参数求偏导
		term=np.multiply(error,X[:,j])
		grad[0,j]=np.sum(term)/len(X)
	return grad

STOP_ITER=0
STOP_COST=1
STOP_GRAD=2

def stopCriterion(type,value,threshold):
	#设定三种不同的停止策略
	if type==STOP_ITER:   return value>threshold
	elif type==STOP_COST: return abs(value[-1]-value[-2])<threshold
	elif type==STOP_GRAD: return np.linalg.norm(value)<threshold

import numpy.random
#洗牌
def shuffleData(data):
	np.random.shuffle(data) #洗牌，把数据打乱，使数据更随机
	cols=data.shape[1]
	X=data[:,0:cols-1]  #洗牌后重新指定X
	y=data[:,cols-1:]   #洗牌后重新指定y
	return X,y

import time
#梯度下降求解
def descent(data,theta,batchSize,stopType,thresh,alpha):
    init_time=time.time()
    i=0
    k=0
    X,y=shuffleData(data)
    grad=np.zeros(theta.shape)
    costs=[cost(X,y,theta)]  

    while True:
	    grad=gradient(X[k:k+batchSize],y[k:k+batchSize],theta)
	    k+=batchSize  
	    if k>=n:
		    k=0
		    X,y=shuffleData(data)
	    theta=theta-alpha*grad
	    costs.append(cost(X,y,theta))
	    i+=1

	    if stopType==STOP_ITER:    value=i
	    elif stopType==STOP_COST:    value=costs
	    elif stopType==STOP_GRAD:    value=grad
	    if stopCriterion(stopType,value,thresh):   break
    return theta,i-1,costs,grad,time.time()-init_time


def runExpe(data,theta,batchSize,stopType,thresh,alpha):
            #import pub: pub.set_trace()
	    theta,iter,costs,grad,dur=descent(data,theta,batchSize,stopType,thresh,alpha)
	    name="Original" if (data[:,1]>2).sum()>1 else "Scaled"
	    name+="data-learning.rate:{}-".format(alpha)
	    if batchSize==n: strDescType="Gradient"
	    elif batchSize==1: strDescType="Stochastle"
	    else: strDescType="Mini-batch({})".format(batchSize)
	    name+=strDescType+"descent-Stop:"
	    if stopType==STOP_ITER: strStop="{} iterations".format(thresh)
	    elif stopType==STOP_COST:strStop="costs change {}".format(thresh)
	    name+=strStop
	    print("***{}\nTheta:{}-Iter:{}-Last cost:{:03.2f}-Duration:{:03.2f}s".format(
	    name,theta,iter,costs[-1],dur))
	    fig,ax=plt.subplots(figsize=(12,4))
	    ax.plot(np.arange(len(costs)),costs,'r')
	    ax.set_xlabel('Iterations')
	    ax.set_ylabel('Cost')
	    ax.set_title(name.upper()+'--Error vs.Iteratin')
	    return theta

n=100

runExpe(orig_data,theta,n,STOP_ITER,thresh=5000,alpha=0.000001)
plt.show()
