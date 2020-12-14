from copy import copy, deepcopy
import time
import random
import numpy as np
import math
import sys
import csv
from q_learning import rl
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages


def get_grid(arg):
        grid_file = open(arg)
        grid_list=grid_file.readlines()
        grid=[]
        for x in grid_list:
                x=x.replace('\r','')
                x=x.replace('\n','')
                grid.append(np.asarray(x.split(",")))
        return np.asarray(grid).astype(int)

def main(argv):
        grid=get_grid(argv[0])
        reward=float(argv[1])
        probability=float(argv[2])
        alpha=0.9
        gamma=0.99
        epsilon=0
        epsilon=1
        decline=0.9
        e_decay=0.99
        Game=np.zeros(4).astype(rl)
        Game=rl(grid,reward,probability,alpha,gamma,epsilon,decline,e_decay)
        print("\nGamma = %s Alpha = %s E_Decay = %s Epsilon = %s Decline = %s Probability = %s\n"%(gamma,alpha,e_decay,epsilon,decline,probability))
        print("Expected Reward at Start is: %s\n"%(Game.play()))
        print(Game.draw_grid(),"\n")
if __name__ == "__main__":
	main(sys.argv[1:])