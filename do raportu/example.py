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
import time
import uuid
import traceback

def eval_genome(genome, config):
    return random.randint(1,5)

def create():
    config_file = 'example.cfg'
    # NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create NEAT population
    pop = neat.Population(config)
    cpus = multiprocessing.cpu_count()
    pe = neat.ParallelEvaluator(cpus, eval_genome)
    winner = pop.run(pe.evaluate, 2)
    visualize.draw_net(config, winner, view=False, filename="drawing_example")
       
if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    create()
