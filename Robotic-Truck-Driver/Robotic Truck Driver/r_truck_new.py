import enum
from copy import copy, deepcopy
import time
import random
import numpy as np
import math
import sys
# import matplotlib.pyplot as plt


class State:
    def __init__(self,wait_time,number_of_packages):
        self.wait_time=wait_time
        self.number_of_packages=number_of_packages
        self.q=[0,0]

class Package:
    def __init__(self, time, destination):
        self.time = time
        self.destination = destination


class Warehouse:
    def __init__(self, length):
        self.length = length
        self.pkg_prod_prob = 0.15
        self.packages = []
        self.reward = 0

    def Production(self, time):
        p_chance = random.random()

        if p_chance < self.pkg_prod_prob:
            new_package = Package(
                time, np.random.randint(1, self.length + 1))
            if self.pkg_prod_prob >= 0.25:
                pass
            else:
                self.pkg_prod_prob += 0.02
        else:
            if self.pkg_prod_prob <= 0.05:
                pass
            else:
                self.pkg_prod_prob -= 0.02
            new_package=None
        return new_package

    def UpdateWaitReward(self, time, packages):
        self.reward = self.GetWaitReward(time, packages)

    # argument 'can_deliver' is for the q function calculator
    # If N number of pkgs are delivered, then their penalty
    # Should not be added to the combined penalty.
    def GetWaitReward(self, time, packages):
        reward_temp = 0
        for each_pkg in packages:
            reward_temp -= time - each_pkg.time
        return reward_temp


class Truck:
    def __init__(self, capacity, length, penalty_deli):
        self.capacity = capacity
        self.length = length
        self.penalty_deli = penalty_deli
        self.packages = []
        self.reward = 0
        self.deli_dep_time = 0
        self.farthest_house = 0
        self.available = True

    def Deliver(self, time):
        self.reward = self.GetReward(time)
        self.available = False
        self.deli_dep_time = time
        self.farthest_house=0
        for each_pkg in self.packages:
            if self.farthest_house < each_pkg.destination:
                self.farthest_house = each_pkg.destination
        # print("Farthest house:", self.farthest_house)
        self.packages = []

    def GetReward(self, time):
        reward_temp = 0
        # Penalty for starting the truck
        reward_temp -= self.penalty_deli

        # Reward for delivering packages
        # If zero, then wont add anything.
        reward_temp += 30 * self.length * len(self.packages)

        # Penalty for waiting till delivering packages
        for each_pkg in self.packages:
            reward_temp -= np.sum(np.arange(1,
                                            each_pkg.destination) + time - each_pkg.time)

        return reward_temp

class QAgent:
    def __init__(self, capacity, road_length, penalty_d, time_max):
        self.Warehouse = Warehouse(road_length)
        self.Truck = Truck(capacity, road_length, penalty_d)
        self.time_max = time_max
        self.road_length = road_length
        self.penalty_d = penalty_d
        self.epsilon=0.8
        self.e_decay=0.9
        self.time = 0
        self.reward = 0
        self.transition_reward=0

        # Q parameters
        self.alpha = 0.1
        self.gamma = 0.99

    def Train(self):
        current_state=State(0,0)
        action=0
        td=0
        while(self.time < self.time_max or td == 1 or self.time_max == -1 ):  
            new_package=self.Warehouse.Production(self.time)
            if new_package:
                if (len(self.Truck.packages)< self.Truck.capacity) and self.Truck.available:
                    self.Truck.packages.append(new_package)
                else: 
                    self.Warehouse.packages.append(new_package)
            
            combined_packages=deepcopy(self.Warehouse.packages)+ (deepcopy(self.Truck.packages))
            # print("Time_Step = %s"%self.time,"Reward = %s"%self.reward,"number_of_packages = %s"%len(combined_packages),"Action = %s"%action)
            wait_time = self.GetWaitTime(self.time,combined_packages)
            
            prev_state=current_state
            prev_action=action
            
            self.Warehouse.UpdateWaitReward(self.time,combined_packages)
            self.reward+=self.Warehouse.reward
            if action==0:
                self.transition_reward=self.Warehouse.reward
            print("Time_Step = %s"%self.time,"Reward = %s"%self.reward,"number_of_packages = %s"%len(combined_packages),"Action = %s"%action,"Transition_reward = %s\n"%self.transition_reward)

            if self.Truck.available:
                
                current_state=State(wait_time,len(self.Truck.packages))
                prev_state.q[prev_action] += self.alpha * (self.transition_reward + self.gamma * (max(current_state.q)) - prev_state.q[prev_action])
                print("State: %s"%([prev_state.number_of_packages,prev_state.wait_time]),"State Q-values = %s"%prev_state.q)
                
                if self.move(self.epsilon,self.e_decay):
                    action=random.choice([0,1])
                else:
                    action=np.argmax(current_state.q)

                if action == 0: #Wait
                    self.time+=1
                
                if action == 1: #Deliver
                    self.Truck.Deliver(self.time)
                    td=1
                    self.transition_reward=0
                    print(self.Truck.reward,self.Truck.farthest_house)


            if action == 1:
                if self.time == self.Truck.deli_dep_time + 2 * self.Truck.farthest_house:
                    print("Entered Here")
                    self.Truck.available=True
                    self.Truck.packages=deepcopy(self.Warehouse.packages[:self.Truck.capacity])
                    self.Warehouse.packages=deepcopy(self.Warehouse.packages[self.Truck.capacity:])
                    print(self.Truck.reward)
                    self.reward+=self.Truck.reward
                    self.transition_reward+=self.Truck.reward
                    td=0
                self.transition_reward+=self.Warehouse.reward
                self.time+=1
        print("Time_Step = %s"%self.time,"Reward = %s"%self.reward,"number_of_packages = %s"%len(combined_packages),"Action = %s"%action)
       
    def move(self,probability,e_decay):
        i= random.random()
        if i<=probability*e_decay:
                return 1
        else:
                return 0

    @staticmethod
    def GetWaitTime(curr_time, packages):
        wait_time = 0
        earliest_pkg = sys.maxsize
        for each_pkg in packages:
            earliest_pkg = earliest_pkg if earliest_pkg < each_pkg.time else each_pkg.time
        if len(packages) > 0:
            wait_time = curr_time - earliest_pkg
        return wait_time
