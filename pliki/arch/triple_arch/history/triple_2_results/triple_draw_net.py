import xml.etree.ElementTree as ET
import os
import sys
import optparse
import random
from sumolib import checkBinary
import traci
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import neat
import multiprocessing
import pickle
from datetime import datetime
import string
import visualize



def test():
    # NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "triple_config_3.cfg")
    filename_prefix = str(seed) + "_" + str(idx) + "_pop_size_"+str(pop_size)+"_"+data_value+"step_"+str(learningStepCount)+"gen_"+str(gen_count)
    with open("results\\winner" + filename_prefix, 'rb') as f:
        winner = pickle.load(f)
        net = neat.nn.RecurrentNetwork.create(winner, config)                                       
        visualize.draw_net(config, winner, view=False,filename="testing\\net"+filename_prefix+".gv")
                            

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    test()


