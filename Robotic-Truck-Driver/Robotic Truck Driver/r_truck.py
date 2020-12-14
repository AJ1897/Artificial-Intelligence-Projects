import enum
from copy import copy, deepcopy
import time
import random
import numpy as np
import math
import sys
# import matplotlib.pyplot as plt


class ACTION(enum.Enum):
    WAIT = 0
    DELIVER = 1


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


class QAgent:
    def __init__(self, capacity, road_length, penalty_d, time_max):
        self.Warehouse = Warehouse(road_length)
        self.Truck = Truck(capacity, road_length, penalty_d)
        self.time_max = time_max
        self.road_length = road_length
        self.penalty_d = penalty_d

        self.time = 0
        self.reward = 0

        # Q parameters
        self.q_dict = {}
        self.alpha = 0.1
        self.gamma = 0.99

    def Train(self):
        print("\n\n")
        while(self.time < self.time_max or self.time_max == -1):
            packages = deepcopy(self.Warehouse.Production(self.time))
            self.UpdateQ(packages)

            if self.Truck.IsAvailable(self.time):
                self.Truck.LoadPkg(self.Warehouse.packages)

            # This If condition is temporary for testing purpose.
                if self.GetActionFromQ(packages) == ACTION.DELIVER:
                    self.Warehouse.RemoveDelivered(
                        len(self.Truck.packages))
                    self.Truck.Deliver(self.time)
                    print("Off to delivery")

            self.reward = self.Warehouse.reward + self.Truck.reward
            self.time += 1
            print(self.time, self.reward, len(
                self.Warehouse.packages), len(self.Truck.packages), self.Truck.IsAvailable(self.time))
        print("lsls")

    def UpdateQ(self, packages):
        wait_time = self.GetWaitTime(self.time, packages)
        earliest_pkg = sys.maxsize

        for each_pkg in packages:
            earliest_pkg = earliest_pkg if earliest_pkg < each_pkg.time else each_pkg.time
        if len(packages) > 0:
            wait_time = self.time - earliest_pkg

        # I think second parameter should be no. of packages ON TRUCK.
        # Once we hit the capacity, then any additional packages in the
        # warehouse should not have any effect on state, as we anyway cannot deliver

        self.q_dict[(wait_time, len(packages), ACTION.DELIVER)
                    ] = self.GetQValue(wait_time, packages, ACTION.DELIVER)

        self.q_dict[(wait_time, len(packages), ACTION.WAIT)
                    ] = self.GetQValue(wait_time, packages, ACTION.WAIT)

    def GetQValue(self, wait_time, packages, action):
        curr_q = self.q_dict.get((wait_time, len(packages), action), 0)
        next_q_wait = self.q_dict.get(
            (wait_time+1, len(packages), ACTION.WAIT), 0)
        next_q_deli = self.q_dict.get(
            (wait_time + 1, len(packages), ACTION.DELIVER), 0)

        reward = self.getPossibleReward(action)

        curr_q += self.alpha * (reward + self.gamma *
                                (max(next_q_deli, next_q_wait)) - curr_q)

        return curr_q

    def getPossibleReward(self, action):
        wh_reward = 0
        deli_reward = 0
        if action == ACTION.DELIVER:
            wh_reward = self.Warehouse.GetWaitReward(
                self.time+1, len(self.Truck.packages))
            deli_reward = self.Truck.GetReward(self.time+1)
        else:
            wh_reward = self.Warehouse.GetWaitReward(self.time+1)

        return self.reward + wh_reward + deli_reward

    def GetActionFromQ(self, packages):
        wait_time = self.GetWaitTime(self.time, packages)
        q_deli = self.q_dict.get((wait_time, len(packages), ACTION.DELIVER), 0)
        q_wait = self.q_dict.get((wait_time, len(packages), ACTION.WAIT), 0)
        if q_deli >= q_wait:
            return ACTION.DELIVER
        else:
            return ACTION.WAIT

    @staticmethod
    def GetWaitTime(curr_time, packages):
        wait_time = 0
        earliest_pkg = sys.maxsize

        for each_pkg in packages:
            earliest_pkg = earliest_pkg if earliest_pkg < each_pkg.time else each_pkg.time
        if len(packages) > 0:
            wait_time = curr_time - earliest_pkg
        return wait_time
