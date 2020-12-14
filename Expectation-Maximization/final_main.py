import csv
import sys
import numpy as np
import random
from copy import deepcopy
from numpy import genfromtxt
from scipy import linalg
from em_final import EM
import math
import time
import csv

# FUNCTION TO GENERATE RANDOM DATA
def generate_random_pts(noc):
    dimension=2
    m=[]
    c=[]
    clust=np.zeros(((25-noc)*20*noc,dimension+1))
    for i in range(noc):
        if(i>0):
            mean=mean+np.random.randint(1,5)
            pts=np.vstack((pts,np.random.multivariate_normal(mean,covariance,(25-noc)*20)))
            clust[:(25-noc)*20*(i+1),:-1]=pts
            clust[(25-noc)*20*i:(25-noc)*20*(i+1),-1]=i+1
        else:
            mean=np.random.rand(dimension)*np.random.randint(100,size=dimension)
            rm=mean
            covariance=np.diag(np.random.rand(dimension)*np.random.randint(10,size=dimension))
            pts=np.random.multivariate_normal(mean,covariance,(25-noc)*20)
            clust[:(25-noc)*20,:-1]=pts
            clust[:(25-noc)*20,-1]=i+1
        m.append(mean)
        c.append(covariance)
        #GENERATING TEST DATA FILE
        # file_name="test4.csv"
        # with open(file_name, 'w') as csvfile: 
        # # creating a csv writer object 
        #     csvwriter = csv.writer(csvfile) 
        #     csvwriter.writerow(["x","y","Cluster"])  
        #     # writing the data rows 
        #     csvwriter.writerows(clust)
        #     csvwriter.writerow(["Means"])
        #     csvwriter.writerows(m)
        #     csvwriter.writerow(["Covariance"])
        #     csvwriter.writerows(c)
    print("Original Data Means:\n",np.asarray(m),"\n")
    print("Original Data Covariance:\n",np.asarray(c),"\n")
    return pts


def main(argv):
    # STORING GIVEN POINTS
    points = genfromtxt(argv[0], delimiter=" ")
    
    # USING RANDOMLY GENERATED POINTS
    #points=generate_random_pts(15)

    # INITIALIZING PARAMETERS
    best_means=[]
    best_cov=[]
    best_ll=-(math.inf)
    best_bic=math.inf
    n=int(argv[1])
    prog_start_time=time.time()

    # CHECKING WHICH MODES TO FOLLOW:
        # n=0 for finding best-fit number of clusters
        # n!=0 for finding best-fit clusters

    if n==0:
        threshold=1 # Thershold increased so as to accomodate more time for each cluster
        runtime=np.arange(2,21) # Runtime proportionally distributed according to the number of clusters.
        runtime=10*runtime/runtime.sum() # Total runtime accounts to 10 seconds.
        i=2 
        r=21
    else:
        threshold=0.01 # Thershold reduced so as to reach more better cluster estimates
        rt=10 # Runtime of 10 seconds
        i=n
        r=n+1
    while i <r:
        # print(i)
        # INITIALIZING LOCAL CLUSTER PARAMETERS
        c_ll=-math.inf 
        c_m=[]
        c_c=[]
        start_time=time.time() # Start-time for each cluster iteration
        if n==0: rt=runtime[i-2] 

        # LOOPING FOR BEST CUSTER ESTIMATES
        while time.time()-start_time <rt:
            pll=-(math.inf)
            ll=0
            test=EM(points,i) 
            while abs(ll-pll)>threshold:
                pll=ll
                e,ll=test.E_step() # Calculating expectations and log-likelihood
                # print(ll)
                if ll==0: # Avoiding errors
                    ll=c_ll
                    break
                print(ll)
                test.M_step(e) # Maximizing the cluster estimates
                # print(ll)
            if ll>c_ll: # Prioritizing best cluster estimates
                clusters=test.assign_clusters(e) #ASSIGNING CLUSTERS
                c_ll=ll
                c_m=deepcopy(test.means)
                c_c=deepcopy(test.cov)
            print("\n")
        # CALCULATING BIC INDEX
        bic=test.BIC(c_ll) 
        # print(bic)

        # PRIORITIZING CLUSTER WITH BEST BIC INDEX
        if bic<best_bic:
            best_clusters=clusters
            best_bic=bic
            best_ll=c_ll
            best_means=c_m
            best_cov=c_c
            best_cluster=i
        i+=1

    #################################################################
    ############### P R I N T I N G    R E S U L T S ################
    #################################################################
    print("\nBest Likelihood: ",best_ll)
    print("\nBest BIC: ",best_bic)
    print("\nBest Cluster: ",best_cluster)
    print("\nBest Means:\n",best_means)
    print("\nBest Covariance:\n",best_cov)
    print("\nEnd-Time: %0.2f"%(time.time()-prog_start_time))

    
    #CHECKING CLUSTER ASSIGNMENT
    # file_name="result11.csv"
    # with open(file_name, 'w') as csvfile: 
    #     # creating a csv writer object 
    #     csvwriter = csv.writer(csvfile) 
    #     csvwriter.writerow(["x","y"])  
    #     # writing the data rows 
    #     csvwriter.writerows(best_clusters)
    #     csvwriter.writerow(["Means"])
    #     csvwriter.writerows(best_means)
    #     csvwriter.writerow(["Covariance"])
    #     csvwriter.writerows(best_cov)

if __name__ == "__main__":
    main(sys.argv[1:])