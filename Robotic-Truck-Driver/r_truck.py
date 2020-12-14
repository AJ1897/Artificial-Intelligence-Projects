import enum
from copy import copy, deepcopy
import time
import random
import numpy as np
import math
import sys
import csv
# import matplotlib.pyplot as plt

#########################################################
####### E N V I R O N M E N T A L   C L A S S E S #######
#########################################################


class State:
    def __init__(self, wait_time, nos_pkgs, action):
        self.wait_time = wait_time
        self.nos_pkgs = nos_pkgs
        self.action = action


class ACTION(enum.Enum):
    WAIT = 0
    DELIVER = 1


class Package:
    def __init__(self, time, destination):
        self.time = time
        self.destination = destination


#########################################################
#################   W A R E H O U S E   #################
#########################################################

class Warehouse:
    def __init__(self, length):
        self.length = length
        self.pkg_prod_prob = 0.15
        self.packages = []
        self.reward = 0

    def Production(self, time):
        p_chance = random.random()
        self.UpdateWaitReward(time)

        if p_chance < self.pkg_prod_prob:
            new_package = Package(
                time, np.random.randint(1, self.length + 1))
            self.packages.append(new_package)
            if self.pkg_prod_prob >= 0.25:
                pass
            else:
                self.pkg_prod_prob += 0.02
        else:
            if self.pkg_prod_prob <= 0.05:
                pass
            else:
                self.pkg_prod_prob -= 0.02
        return self.packages

    def UpdateWaitReward(self, time):
        self.reward = self.GetWaitReward(time)

    # argument 'can_deliver' is for the q function calculator
    # If N number of pkgs are delivered, then their penalty
    # Should not be added to the combined penalty.
    def GetWaitReward(self, time, can_deliver=0):
        reward_temp = self.reward
        for each_pkg in self.packages[can_deliver:]:
            reward_temp -= time - each_pkg.time
        return reward_temp

    def RemoveDelivered(self, nos_delivered):
        del self.packages[:nos_delivered]
        # print(nos_delivered, " packages deleted")


#########################################################
#######################   TRUCK   #######################
#########################################################

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
        for each_pkg in self.packages:
            if self.farthest_house < each_pkg.destination:
                self.farthest_house = each_pkg.destination
        # print("Farthest house:", self.farthest_house)
        self.packages = []

    def GetReward(self, time):
        reward_temp = self.reward
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

    def LoadPkg(self, warehouse_packages):
        if len(self.packages) < self.capacity:
            self.packages.extend(
                warehouse_packages[len(self.packages):self.capacity])

    def IsAvailable(self, time):
        if self.available == True:
            return self.available
        # Minus one, because last stop will be delivered in one time tick
        elif time > self.deli_dep_time + 2 * self.farthest_house - 1:
            self.available = True
            self.farthest_house = 0
            self.deli_dep_time = 0
            return True
        else:
            return False


#########################################################
###################   Q   A G E N T   ###################
#########################################################

class QAgent:
    def __init__(self, capacity, road_length, penalty_d, time_max):
        self.Warehouse = Warehouse(road_length)
        self.Truck = Truck(capacity, road_length, penalty_d)
        self.time_max = time_max
        self.road_length = road_length
        self.penalty_d = penalty_d
        self.test_runs = 10000
        self.iterations = 10
        self.final_rewards = []

        self.time = 0
        self.reward = 0
        self.state_table=[]
        self.q_table=[]
        self.visit_table=[]
        # Q parameters
        self.q_dict = {}
        self.alpha = 0.7
        # self.alpha_decay = 1
        self.alpha_decay = 0.05 ** (10 /
                                    self.time_max) if self.time_max > 0 else 0.996
        self.gamma = 0.99
        self.futr_state = State(0, 0, ACTION.WAIT)
        self.last_state = State(0, 0, ACTION.WAIT)
        self.transition_reward = 0
        self.epsilon = 1
        self.decay = 0.05 ** (10 /
                              self.time_max) if self.time_max > 0 else 0.996

    def Train(self):
        inf_reward = 0
        time_step = 50000
        elps_time = time_step

        while(self.time < self.time_max or self.time_max == -1):
            packages = deepcopy(self.Warehouse.Production(self.time))

            if self.Truck.IsAvailable(self.time):
                self.Truck.LoadPkg(self.Warehouse.packages)

                self.last_state = deepcopy(self.futr_state)
                action = self.GetActionFromQ(packages)
                self.futr_state.wait_time = self.GetWaitTime(
                    self.time, packages)
                self.futr_state.nos_pkgs = len(packages)
                self.futr_state.action = action

                self.UpdateQ(packages)
                if action == ACTION.DELIVER:
                    self.Warehouse.RemoveDelivered(
                        len(self.Truck.packages))
                    self.Truck.Deliver(self.time)

            self.transition_reward = self.reward
            self.reward = self.Warehouse.reward + self.Truck.reward
            self.transition_reward = self.reward - self.transition_reward

            self.time += 1
            if self.time > elps_time and self.time_max < 0:
                inf_reward = self.reward - inf_reward
                elps_time += time_step
                print("Reward from last", time_step, "ticks:", inf_reward)

        # print(np.asarray(self.q_table).astype(str))
        # print(q_array[:,0][:,0])
        # print(self.state_table)
        self.display_Q_Table()
        print("################################################################")
        print("The Reward of training model for ",
              self.time, " ticks is:", self.reward)
        print("################################################################")



    def display_Q_Table(self):
        q_array=np.asarray(self.q_table)
        x_max=int(max(q_array[:,0][:,0]))
        y_max=int(max(q_array[:,0][:,1]))
        Q_Table=np.full((x_max+1,y_max+1),"?")
        Visit_Q_Table=np.full((x_max+1,y_max+1),"0")
        for i in range(x_max+1):
            for j in range(y_max+1):
                if [i,j] in self.state_table:
                    index=self.state_table.index([i,j])
                    action=np.argmax(self.q_table[index][1])
                    Visit_Q_Table[-(i+1), j]=self.visit_table[index]
                    if action:    
                        Q_Table[-(i+1), j]="Deliver"
                    else: 
                        Q_Table[-(i+1), j]="Wait"
        print("################################################################")
        print("Policy Table is:")
        print("\nX_Axis: Number of Packages")
        print("Y_Axis: Earliest Waiting Time\n")
        print(np.asarray(Q_Table))
        # print("\n################################################################")
        # print("State Visit Table is:")
        # print(np.asarray(Visit_Q_Table),"\n"

    def Test(self):
        itern = 0
        while(itern < self.iterations):
            loc_warehouse = Warehouse(self.road_length)
            loc_truck = Truck(self.Truck.capacity,
                              self.road_length, self.penalty_d)
            reward = 0
            time_temp = 0
            while(time_temp < self.test_runs):
                packages = deepcopy(loc_warehouse.Production(time_temp))

                if loc_truck.IsAvailable(time_temp):
                    loc_truck.LoadPkg(loc_warehouse.packages)
                    action = self.GetGreedyAction(packages)
                    if action == ACTION.DELIVER:
                        loc_warehouse.RemoveDelivered(
                            len(loc_truck.packages))
                        loc_truck.Deliver(time_temp)

                reward = loc_warehouse.reward + loc_truck.reward
                time_temp += 1
            itern += 1
            self.final_rewards.append(reward)
            print("Reward for iteration ", itern, ": ", reward)

        avg = np.mean(self.final_rewards)
        print("################################################################")
        print("Average Rewards for ", self.iterations,
              " iterations of ", self.test_runs, " ticks: ", avg)
        print("################################################################")


###################   Q   T A B L E   ###################

    def UpdateQ(self, packages):
        self.q_dict[(self.last_state.wait_time, self.last_state.nos_pkgs,
                     self.last_state.action)] = self.GetQValue(packages)
        
        state=[int(self.last_state.wait_time),int(self.last_state.nos_pkgs)]
        wait_q=self.q_dict.get((self.last_state.wait_time, self.last_state.nos_pkgs,ACTION.WAIT), 0)
        deliver_q=self.q_dict.get((self.last_state.wait_time, self.last_state.nos_pkgs,ACTION.DELIVER), 0)

        if state in self.state_table:
            index=self.state_table.index(state)
            self.q_table[index]=[state,[wait_q,deliver_q]]
            self.visit_table[index]+=1
        else:
            self.state_table.append(state)
            self.visit_table.append(1)
            self.q_table.append([state,[wait_q,deliver_q]])

    def GetQValue(self, packages):
        curr_q = self.q_dict.get((self.last_state.wait_time, self.last_state.nos_pkgs,
                                  self.last_state.action), 0)

        # self.futr_state.action may have a random action, so chosing one with max Q
        next_q_deli = self.q_dict.get(
            (self.futr_state.wait_time, self.futr_state.nos_pkgs,
             ACTION.DELIVER), 0)
        next_q_wait = self.q_dict.get(
            (self.futr_state.wait_time, self.futr_state.nos_pkgs,
             ACTION.WAIT), 0)

        reward = self.transition_reward

        self.alpha *= self.alpha_decay
        curr_q += self.alpha * (reward + self.gamma *
                                (max(next_q_deli, next_q_wait)) - curr_q)

        return curr_q

    def GetActionFromQ(self, packages):
        action = self.GetGreedyAction(packages)
        return action if self.RandomPossiblity() else random.choice([ACTION.DELIVER, ACTION.WAIT])

    def GetGreedyAction(self, packages):
        wait_time = self.GetWaitTime(self.time, packages)
        q_deli = self.q_dict.get((wait_time, len(packages), ACTION.DELIVER), 0)
        q_wait = self.q_dict.get((wait_time, len(packages), ACTION.WAIT), 0)

        return ACTION.DELIVER if q_deli >= q_wait else ACTION.WAIT

    @staticmethod
    def GetWaitTime(curr_time, packages):
        wait_time = 0
        earliest_pkg = sys.maxsize

        for each_pkg in packages:
            earliest_pkg = earliest_pkg if earliest_pkg < each_pkg.time else each_pkg.time
        if len(packages) > 0:
            wait_time = curr_time - earliest_pkg
        return wait_time

    def RandomPossiblity(self):
        i = random.random()
        self.epsilon *= self.decay
        if i >= self.epsilon:
            return True
        else:
            return False