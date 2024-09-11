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

learningStepCount = 2
street_net_name = "double"
rou = "_3"
distance = "_500"

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

def judge(currentPhase_duration, currentPhase, prediction):
    if(currentPhase == 0 or currentPhase == 2):
        if(currentPhase_duration < 10):
            #do nothing
            return False
        elif (currentPhase_duration > 300):
            #switch
            return True
        else:
            if(prediction > 0.7 and currentPhase == 0):
                #switch
                return True
            elif(prediction < 0.3 and currentPhase == 2):
                #switch
                return True
    return False

def run_sim(model,lenght,num,data_value,testStepCount):
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S")
    characters = string.ascii_letters
    random_string = ''.join(random.choices(characters, k=8))
    name = 'z' + street_net_name + random_string + time_string
    summary_file = "summaries\\"+name+"summary_"+street_net_name+".xml"
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="'''+street_net_name+lenght+'''_neat.net.xml"/>
            <route-files value="'''+street_net_name+num+'''.rou.xml"/>
        </input>
        <output>
            <summary-output value="'''+summary_file+'''"/>
        </output>
    </configuration>'''
    filename = name + street_net_name + '.sumocfg'
    with open(filename, 'w') as file:
        file.write(xml_content)
    

    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary, "-c", filename,
                 "--time-to-teleport","-1"])
    
    """execute the TraCI control loop"""
    step = 0
    junctions = [["x" ,"C4","D4","x" ],
                 ["B3","C3","D3","E3"],
                 ["x","C2","D2","x"]]

    last_changes = [[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]]
    
    phases = [[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]]
    
    phase_new = 0
    
    phases_adjusted = [[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]]
    
    traci.trafficlight.setPhase(junctions[1][1], 0)
    while traci.simulation.getMinExpectedNumber() > 0:
        for i in range(0,3):
            for j in range(0,4):
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

        xy = []
        raw_data = []
        for i in range(1,2):
            for j in range(1,3):
                xy.append([[i,j],[i,j-1]])
                xy.append([[i,j],[i+1,j]])
                xy.append([[i,j],[i,j+1]])
                xy.append([[i,j],[i-1,j]])
                raw_data.append(phases_adjusted[i][j])
                raw_data.append(step - last_changes[i][j])
            
        for coords in xy:
            lane = junctions[coords[0][0]][coords[0][1]] + junctions[coords[1][0]][coords[1][1]] + "_0"
            
            if(data_value == "VehicleNumber"):
                raw_data.append(traci.lane.getLastStepVehicleNumber(lane))
            elif(data_value == "HaltingNumber"):
                raw_data.append(traci.lane.getLastStepHaltingNumber(lane))
            elif(data_value == "MeanSpeed"):
                raw_data.append(traci.lane.getLastStepMeanSpeed(lane))
            elif(data_value == "VehicleNumberAndHaltingNumber"):
                raw_data.append(traci.lane.getLastStepVehicleNumber(lane))
                raw_data.append(traci.lane.getLastStepHaltingNumber(lane))
            elif(data_value == "HaltingNumberAndMeanSpeed"):
                raw_data.append(traci.lane.getLastStepHaltingNumber(lane))
                raw_data.append(traci.lane.getLastStepMeanSpeed(lane))
            elif(data_value == "MeanSpeedAndVehicleNumber"):
                raw_data.append(traci.lane.getLastStepMeanSpeed(lane))
                raw_data.append(traci.lane.getLastStepVehicleNumber(lane))
            else:
                raw_data.append(traci.lane.getLastStepVehicleNumber(lane))
                raw_data.append(traci.lane.getLastStepHaltingNumber(lane))
                raw_data.append(traci.lane.getLastStepMeanSpeed(lane))

        input_data = np.array(raw_data)
        prediction = model.activate(input_data)

        id_pred = 0
        for i in range(1,2):
            for j in range(1,3):        
                if(judge(step - last_changes[i][j], phases[i][j], prediction[id_pred])):
                    traci.trafficlight.setPhase(junctions[i][j],phases[i][j]+1)
                    last_changes[i][j] = step
                id_pred = id_pred+ 1

        for i in range(testStepCount):
            traci.simulationStep()
        step += testStepCount
        
    traci.close()
    meanT = get_fitness(summary_file,"meanTravelTime")

    if os.path.exists(filename):
        os.remove(filename)
        os.remove(summary_file)
    return -meanT


def test():
    data_values = ["VehicleNumber", "HaltingNumber", "MeanSpeed","VehicleNumberAndHaltingNumber","HaltingNumberAndMeanSpeed","MeanSpeedAndVehicleNumber","All"]
    #pop_sizes = [100, 200, 400]
    pop_sizes = [8]
    for gen_count in [1,1,1]:#[40,40,120,300]:
        for i in range(21):
            data_value = data_values[i%7]
            pop_size = pop_sizes[int(i/7)]
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

            filename_prefix = "_pop_size_"+str(pop_size)+"_"+data_value+"step_"+str(learningStepCount)+"gen_"+str(gen_count)
            with open("results\\winner" + filename_prefix, 'rb') as f:
                winner = pickle.load(f)
                net = neat.nn.RecurrentNetwork.create(winner, config)
                for testStepCount in [1,2,3,4,5,6,7,8]:
                    print("lerning step: ", str(learningStepCount))
                    print("test step: ", str(testStepCount))
                    results = []
                    for num in ["_1","_2","_3","_4"]:
                        results.append(run_sim(net, distance,num,data_value,testStepCount))
                    print(results)
                        
                    with open("testing\\results"+filename_prefix + "_testing_step_"+str(testStepCount) + ".txt", 'w') as file:
                        file.write(str(results))
                                        
                visualize.draw_net(config, winner, view=False,
                                   filename="testing\\net"+filename_prefix+".gv")
                            


if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    test()


