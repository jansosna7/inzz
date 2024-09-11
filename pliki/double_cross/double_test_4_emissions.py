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

learningStepCount = 1
street_net_name = "double"
epsilon = 0.01
harsh_time = 2
harsh_result = 1.5

def get_fitness(file_path,value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if lines and lines[-1].strip() != "</summary>":
        # Append "</summary>" to the file if it's not the last line
        with open(file_path, 'a') as file:
            file.write("</summary>\n")
    
    tree = ET.parse(file_path)
    root = tree.getroot()

    last_step = root.findall('step')[-1]

    return float(last_step.get(value))

def extract_emissions(file_path, value_type):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    values = []

    for trip in root.findall('tripinfo'):
        emissions = trip.find('emissions')
        if emissions is not None:
            value = emissions.get(value_type)
            if value is not None:
                values.append(float(value))
    
    return values

def extract_emissions(file_path, value_type):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    values = []

    for trip in root.findall('tripinfo'):
        emissions = trip.find('emissions')
        if emissions is not None:
            value = emissions.get(value_type)
            if value is not None:
                values.append(float(value))
    
    return values

def judge(currentPhase_duration, currentPhase, prediction):
    if(currentPhase_duration < 10):
        return False
    if(currentPhase_duration > 300):
        return True
    if(currentPhase == 0 or currentPhase == 2):
        if(prediction > (0.5+epsilon) and currentPhase == 0):
            # switch
            return True
        elif(prediction < (0.5-epsilon) and currentPhase == 2):
            # switch
            return True
    # do nothing
    return False

def run_sim(genome, config, num):    
    data_value = config.data_value

    result_f1 = config.result_f1
    result_f2 = config.result_f2
    result_f2 = config.result_f2

    t1 = 260 * harsh_time
    t2 = 430 * harsh_time
    t3 = 620 * harsh_time
    t4 = 1180 * harsh_time

    result_f1 = -1*(155*harsh_result)
    result_f2 = -1*(163*harsh_result)
    result_f3 = -1*(176*harsh_result)
    
    name = 'z' + config.id + num
    summary_file = os.path.join("emissions_test", f"{name}emissions{street_net_name}.xml")
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="'''+street_net_name+'''_neat.net.xml"/>
            <route-files value="'''+street_net_name+num+'''.rou.xml"/>
        </input>
        <output>
            <tripinfo-output value="'''+summary_file+'''"/>
        </output>
        <emissions>
            <device.emissions.probability value="1.0"/>
        </emissions>
    </configuration>'''
    filename = name + 'single_cross.sumocfg'
    with open(filename, 'w') as file:
        file.write(xml_content)

    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary,  "--no-warnings", "-c", filename,
                 "--time-to-teleport", "-1"])

    """execute the TraCI control loop"""
    step = 0
    junctions = [["x" , "C8", "D8", "x" , "x" , "G8", "H8", "x"],
                 ["B7", "C7", "D7", "E7", "F7", "G7", "H7", "I7"],
                 ["x" , "C6", "D6", "x" , "x" , "G6", "H6", "x"],
                 ["x" , "C5", "D5", "x" , "x" , "G5", "H5", "x"],
                 ["B4", "C4", "D4", "E4", "F4", "G4", "H4", "I4"],
                 ["x" , "C3", "D3", "x" , "x" , "G3", "H3", "x"]]
 
    rows = len(junctions)
    columns = len(junctions[0])

    last_changes = np.zeros((rows, columns))

    phases = np.zeros((rows, columns))

    phase_new = 0

    phases_adjusted = np.zeros((rows, columns))

    models = [[neat.nn.RecurrentNetwork.create(genome, config),neat.nn.RecurrentNetwork.create(genome, config)],
              [neat.nn.RecurrentNetwork.create(genome, config),neat.nn.RecurrentNetwork.create(genome, config)]]

    step_threshold = config.step_threshold

    #skip until first car approches
    for i in range(41):
        traci.simulationStep()
        
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        for i in range(0, rows):
            for j in range(0, columns):
                if(junctions[i][j] != "x"):
                    phase_new = traci.trafficlight.getPhase(junctions[i][j])
                    if phase_new == 0:
                        phases_adjusted[i][j] = 1
                    elif phase_new == 2:
                        phases_adjusted[i][j] = -1
                    else:
                        phases_adjusted[i][j] = 0
                    if(i == 0 or i == 2 or j == 0 or j == 2):
                        if(phase_new != phases[i][j]):
                            last_changes[i][j] = step
                    phases[i][j] = phase_new

        for net_i in range(2):
            for net_j in range(2):
                # for every net
                xy = []
                raw_data = []
                for junction_relative_i in range(1, 2):
                    junction_absolute_i = junction_relative_i + net_i*3
                    for junction_relative_j in range(1, 3):
                        junction_absolute_j = junction_relative_j + net_j*4
                        xy.append([[junction_absolute_i, junction_absolute_j], [junction_absolute_i, junction_absolute_j-1]])
                        xy.append([[junction_absolute_i, junction_absolute_j], [junction_absolute_i+1, junction_absolute_j]])
                        xy.append([[junction_absolute_i, junction_absolute_j], [junction_absolute_i, junction_absolute_j+1]])
                        xy.append([[junction_absolute_i, junction_absolute_j], [junction_absolute_i-1, junction_absolute_j]])
                        raw_data.append(phases_adjusted[junction_absolute_i][junction_absolute_j])
                        raw_data.append(step - last_changes[junction_absolute_i][junction_absolute_j])

                for coords in xy:
                    lane = junctions[coords[0][0]][coords[0][1]] + \
                        junctions[coords[1][0]][coords[1][1]] + "_0"

                    if(data_value == "VehicleNumber"):
                        raw_data.append(
                            traci.lane.getLastStepVehicleNumber(lane))
                    elif(data_value == "HaltingNumber"):
                        raw_data.append(
                            traci.lane.getLastStepHaltingNumber(lane))
                    elif(data_value == "MeanSpeed"):
                        raw_data.append(traci.lane.getLastStepMeanSpeed(lane))
                    elif(data_value == "VehicleNumberAndHaltingNumber"):
                        raw_data.append(
                            traci.lane.getLastStepVehicleNumber(lane))
                        raw_data.append(
                            traci.lane.getLastStepHaltingNumber(lane))
                    elif(data_value == "HaltingNumberAndMeanSpeed"):
                        raw_data.append(
                            traci.lane.getLastStepHaltingNumber(lane))
                        raw_data.append(traci.lane.getLastStepMeanSpeed(lane))
                    elif(data_value == "MeanSpeedAndVehicleNumber"):
                        raw_data.append(traci.lane.getLastStepMeanSpeed(lane))
                        raw_data.append(
                            traci.lane.getLastStepVehicleNumber(lane))
                    else:
                        raw_data.append(
                            traci.lane.getLastStepVehicleNumber(lane))
                        raw_data.append(
                            traci.lane.getLastStepHaltingNumber(lane))
                        raw_data.append(traci.lane.getLastStepMeanSpeed(lane))

                input_data = np.array(raw_data)
                prediction = models[net_i][net_j].activate(input_data)
                id_pred = 0
                for ix in range(1, 2):
                    junction_absolute_i = ix + net_i*3
                    for jx in range(1, 3):
                        junction_absolute_j = jx + net_j*4
                        if(judge(step - last_changes[junction_absolute_i][junction_absolute_j], phases[junction_absolute_i][junction_absolute_j], prediction[id_pred])):
                            traci.trafficlight.setPhase(
                                junctions[junction_absolute_i][junction_absolute_j], phases[junction_absolute_i][junction_absolute_j]+1)
                            last_changes[junction_absolute_i][junction_absolute_j] = step
                        id_pred = id_pred + 1

        for i in range(learningStepCount):
            traci.simulationStep()
        step += learningStepCount

    traci.close()
    fuels = extract_emissions(summary_file, "fuel_abs")
    meanF = sum(fuels) / len(fuels)
    if os.path.exists(filename):
        os.remove(filename)
    return meanF



def test():
    t1 = 259 * harsh_time
    t2 = 425 * harsh_time
    t3 = 613 * harsh_time
    t4 = 1182 * harsh_time

    result_f1 = -1*(155*harsh_result)
    result_f2 = -1*(163*harsh_result)
    result_f3 = -1*(176*harsh_result)
    data_values = ["VehicleNumber","VehicleNumberAndHaltingNumber","All"]
    pop_size = 200
    gen_count = 300
    for data_value, i in [[data_values[2],402],[data_values[1],401]]:
        if(data_value == "VehicleNumber" or data_value == "HaltingNumber" or data_value == "MeanSpeed"):
            config_file = 'double_config_1.cfg'
        elif(data_value == "VehicleNumberAndHaltingNumber" or data_value == "HaltingNumberAndMeanSpeed" or data_value == "MeanSpeedAndVehicleNumber"):
            config_file = 'double_config_2.cfg'
        elif(data_value == "All"):
            config_file = 'double_config_3.cfg'
        # NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        
        config.data_value = data_value
        config.pop_size = pop_size
        config.t1 = t1
        config.t2 = t2
        config.t3 = t3
        config.t4 = t4
        config.result_f1 = result_f1
        config.result_f2 = result_f2
        config.result_f3 = result_f3
        config.step_threshold = config.t3
        
        filename_prefix = str(i) + "_pop_"+str(pop_size)+"_data_"+data_value+"_gen_"+str(gen_count)
        config.id = filename_prefix
        with open("results\\winner" + filename_prefix, 'rb') as f:
            winner = pickle.load(f)
            results = []
            for num in ["_1","_2","_3","_4"]:
                results.append(run_sim(winner, config, num))
            print(results)
                
            with open("emissions_test\\results"+filename_prefix + ".txt", 'w') as file:
                file.write(str(results))
                                
                
                            


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.makedirs("results", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)
    os.makedirs("testing_just_one", exist_ok=True)
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    test()


