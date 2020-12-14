from copy import copy, deepcopy
import time
import random
import numpy as np
import math
import sys
import matplotlib.pyplot as plt 


# Defining Class STATE for each state in the grid
class state:
        def __init__(self,q=[0,0,0,0]):
                self.q =deepcopy(q)
                self.visited=[1,1,1,1]

# Defining Class RL for Q-learning
class rl:
        def __init__(self,grid,reward,probability,alpha,gamma,epsilon,decline,e_decay):
                self.grid=grid #Initialising Grid
                self.reward=reward #Initialising Move Cost
                self.probability=probability #Initialising probability of actually following an action
                self.game=np.zeros(grid.shape).astype(state) #Initialising variable to store the Q values of each state
                self.alpha=alpha #Initialising Alpha
                self.gamma=gamma #Initialising Gamma
                self.epsilon=epsilon #Initialising Epsilon i.e. Probability of taking a random action
                self.decline=decline #Initialising Alpha decline rate
                self.e_decay=e_decay #Initialising Epsilon Decay rate
                
                # Initilizing the game by storing Q values for each state
                for i in range(len(grid)):
                        for j in range(len(grid[0])):
                                if grid[i,j]==0:
                                        self.game[i,j]=state()
                                else:
                                        self.game[i,j]=state(grid[i,j])
        
        # Defining fuction to make a move
        def move(self,probability,e_decay):
                i= random.random()
                if i<=probability*e_decay:
                        return 1
                else:
                        return 0

        # Function to draw the grid
        def draw_grid(self):
                gr=np.zeros(self.grid.shape).astype(str)
                for i in range(len(self.grid)):
                        for j in range(len(self.grid[0])):
                                if np.asarray(self.game[i,j].q).any()!=0:
                                        direct=np.argmax(self.game[i,j].q)
                                        if self.grid[i,j]!=0:
                                                gr[i,j]=self.grid[i,j]
                                        else:
                                                if direct==0: gr[i,j]="Up"
                                                if direct==1: gr[i,j]="Down"
                                                if direct==2: gr[i,j]="Left"
                                                if direct==3: gr[i,j]="Right"
                                else: gr[i,j]="N/A"                
                return np.asarray(gr)
        

        # Defining function to play for Q-Learning
        def play(self): 
                # Initialising parameters to be used for the function
                t=1 # Counter for storing q values and controlling decay intervals
                prev_state=-math.inf 
                c=0 # Counter to check for convergence
                e_decay=1
                w=e_decay
                start_time=time.time()
                elapsed_time=start_time
                while(c<10000 and elapsed_time - start_time < 20):
                        # Checking for convergence of the program
                        if abs(np.max(self.game[-1,0].q)-prev_state)< 0.0001: c+=1
                        
                        prev_state=np.max(self.game[-1,0].q)
                        current_state=self.game[-1,0]
                        i,j=[-1,0]
                        elapsed_time=time.time()-start_time
                        
                        # Looping over till we reach a terminal state 
                        while(self.grid[i,j]==0):
                                prev_i,prev_j=i,j
                                
                                # Epsilon Decay
                                if t%1000==0:
                                        e_decay=w*self.e_decay
                                        w=e_decay

                                if self.move(self.epsilon,e_decay):
                                        action=random.choice([0,1,2,3])
                                else:
                                        action=np.argmax(current_state.q)

                                # Checking with probability for following a taken action
                                if self.move(self.probability,1): 
                                        m=action
                                else:
                                        if action>1:
                                                m=random.choice([0,1])
                                        else:
                                                m=random.choice([2,3])

                                current_state.visited[action]+=1

                                # Checking for boundary conditions
                                if m==0: i=i-1
                                if m==1: i=i+1
                                if m==2: j=j-1
                                if m==3: j=j+1
                                if i<-len(self.grid): i=-len(self.grid)
                                if i>-1: i=-1
                                if j<0: j=0
                                if j==len(self.grid[0]): j=len(self.grid[0])-1

                                next_state=self.game[i,j]

                                # Q-Learning Formula 
                                current_state.q[action]+=self.alpha * (self.reward + self.gamma * (np.max(next_state.q)) - current_state.q[action])
                                self.game[prev_i,prev_j]=current_state

                                current_state=next_state

                        t+=1
                        if t%100==0:
                                self.alpha*=self.decline
                return np.max(self.game[-1,0].q)


                        
