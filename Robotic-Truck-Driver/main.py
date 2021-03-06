from copy import copy, deepcopy
import time
import random
import numpy as np
import math
import sys
# import matplotlib.pyplot as plt
from r_truck import Warehouse, Truck, QAgent


def main(argv):
    capacity, length, penalty_d, clocks = argv
    print("The Input Parameters are:")
    print("Truck Capacity:",capacity,"Road length:",length,"Penalty to deliver:",penalty_d,"Clock Ticks:",clocks)
    Agent = QAgent(int(capacity), int(length), abs(int(penalty_d)), int(clocks))
    Agent.Train()
    Agent.Test()


if __name__ == "__main__":
    main(sys.argv[1:])
