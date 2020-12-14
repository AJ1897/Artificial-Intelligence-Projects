import csv
import sys
import numpy as np
import random
from copy import deepcopy
from numpy import genfromtxt
from scipy import linalg
import math
import time
np.seterr(all=None, divide=None, over='ignore', under=None, invalid='ignore')

class EM:
    def __init__(self,points,noc):
        self.noc=noc
        self.points=points
        self.n=len(points)
        self.dimensions=len(points[0])
        self.cont=np.ones(noc)*(1/noc)
        self.means=np.asarray(random.sample(list(points),k=noc))
        self.cov=np.ones((noc,self.dimensions,self.dimensions))*np.identity(self.dimensions)
        self.k=(noc-1)+(noc*self.dimensions)+0.5*(noc*self.dimensions*(self.dimensions-1))

    def E_step(self):
        e=[]
        for j in range(self.noc):
            dist=self.points-self.means[j]
            det=np.linalg.det(self.cov[j])
            if det==0:return 0,0
            num=np.exp((-0.5)*np.sum(np.dot(dist,np.linalg.inv(self.cov[j]))*dist,axis=1))*self.cont[j]
            denom=math.sqrt(math.pow(2*math.pi,self.dimensions)*abs(det))
            e.append(num/denom)

        e=np.asarray(e)
        sum=np.sum(e,axis=0)

        if sum.all()>0:
            ll=np.sum(np.log(sum))
            return e/sum,ll
        else:
            return 0,0

    def M_step(self,e):
        sum_m=np.sum(e,axis=1)
        for j in range(self.noc):
            self.means[j]=np.sum(self.points*e[j].reshape(self.n,1),axis=0)/sum_m[j]
            dist=self.points-self.means[j]
            self.cov[j]=np.dot(dist.T,dist*e[j].reshape(self.n,1))/sum_m[j]
            self.cont[j]=sum_m[j]/self.n

    def BIC(self,ll):
        return self.k*math.log(self.n)-2*ll

    def assign_clusters(self,e):
        temp=np.zeros((self.n,self.dimensions+1))
        temp[:, :-1] = np.asarray(self.points)
        for i in range(self.n):
            temp[i,-1]=np.argmax(e.T[i])+1
        return temp


 


