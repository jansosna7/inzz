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

learningStepCount = 1
street_net_name = "triple"

epsilon = 0.2

harsh_time = 2
harsh_result = 1.25

t1 = 373 * harsh_time
t2 = 505 * harsh_time
t3 = 979 * harsh_time

result_f1_1 = -1*(169*harsh_result)
result_f2_1 = -1*(175*harsh_result)


def get_fitness(file_path, value):
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
    if(currentPhase_duration < 10 or currentPhase_duration > 300):
        return False
    if(currentPhase == 0 or currentPhase == 2):
        if(prediction > (0.5+epsilon) and currentPhase == 0):
            # switch
            return True
        elif(prediction < (0.5-epsilon) and currentPhase == 2):
            # switch
            return True
    # do nothing
    return False

def run_sim(genome, config, num, data_value, test_step):
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S.%f")
    characters = string.ascii_letters
    random_string = ''.join(random.choices(characters, k=50))
    name = 'z' + street_net_name + random_string + time_string
    summary_file = "summaries//"+name+"summary_"+street_net_name+".xml"
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="'''+street_net_name+'''_neat.net.xml"/>
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
                 "--time-to-teleport", "-1"])

    """execute the TraCI control loop"""
    step = 0
    junctions = [["x" , "C6", "D6", "E6", "x" ],
                 ["B5", "C5", "D5", "E5", "F5"],
                 ["x" , "C4", "D4", "E4", "x" ],
                 ["x" , "C3", "D3", "E3", "x" ],
                 ["B2", "C2", "D2", "E2", "F2"],
                 ["x" , "C1", "D1", "E1", "x" ]]

    tls =       [["x" , "x" , "x" , "x" ,"x"  ],
                 ["x" , "C5", "D5", "E5","x"  ],
                 ["x" , "x" , "x" , "x" ,"x"  ],
                 ["x" , "x" , "x" , "x" ,"x"  ],
                 ["x" , "C2", "D2", "E2","x"  ],
                 ["x" , "x" , "x" , "x" ,"x"  ]]

 
    rows = len(junctions)
    columns = len(junctions[0])

    last_changes = np.zeros((rows, columns))

    phases = np.zeros((rows, columns))

    phase_new = 0

    phases_adjusted = np.zeros((rows, columns))

    models = [neat.nn.RecurrentNetwork.create(genome, config),neat.nn.RecurrentNetwork.create(genome, config)]

    step_threshold = 1000
    if(num == "_1"):
        step_threshold = t1
    if(num == "_2"):
        step_threshold = t2
    if(num == "_3"):
        step_threshold = t3

    while traci.simulation.getMinExpectedNumber() > 0:
        if(step > step_threshold):
            traci.close()
            meanT = get_fitness(summary_file, "meanTravelTime")+200
            if os.path.exists(filename):
                os.remove(filename)
                os.remove(summary_file)
            return -meanT

        for i in range(0, rows):
            for j in range(0, columns):
                if(tls[i][j] != "x"):
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
            for net_j in range(1):
                # for every net
                xy = []
                raw_data = []
                for junction_relative_i in range(1, 2):
                    junction_absolute_i = junction_relative_i + net_i*3
                    for junction_relative_j in range(1, 4):
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
                #prediction = models[net_i][net_j].activate(input_data)
                prediction = models[net_i].activate(input_data)
                id_pred = 0
                for ix in range(1, 2):
                    junction_absolute_i = ix + net_i*3
                    for jx in range(1, 4):
                        junction_absolute_j = jx + net_j*4
                        if(judge(step - last_changes[junction_absolute_i][junction_absolute_j], phases[junction_absolute_i][junction_absolute_j], prediction[id_pred])):
                            traci.trafficlight.setPhase(
                                junctions[junction_absolute_i][junction_absolute_j], phases[junction_absolute_i][junction_absolute_j]+1)
                            last_changes[junction_absolute_i][junction_absolute_j] = step
                        id_pred = id_pred + 1

        for i in range(test_step):
            traci.simulationStep()
        step += test_step

    traci.close()
    meanT = get_fitness(summary_file, "meanTravelTime")

    if os.path.exists(filename):
        os.remove(filename)
        os.remove(summary_file)
    return -meanT


def test():
    data_values = ["VehicleNumber", "HaltingNumber", "MeanSpeed","VehicleNumberAndHaltingNumber","HaltingNumberAndMeanSpeed","MeanSpeedAndVehicleNumber","All"]
    for seed in [10048]:
        for ra in [0]:
            for gen_count in [300]:
                for i in range(1):
                    #data_value = data_values[i%7]
                    data_value = "All"
                    #pop_size = pop_sizes[int(i/7)]
                    pop_size = 300
                    if(data_value == "VehicleNumber" or data_value == "HaltingNumber" or data_value == "MeanSpeed"):
                        config_file = 'triple_1.cfg'
                    elif(data_value == "VehicleNumberAndHaltingNumber" or data_value == "HaltingNumberAndMeanSpeed" or data_value == "MeanSpeedAndVehicleNumber"):
                        config_file = 'triple_config_2.cfg'
                    elif(data_value == "All"):
                        config_file = 'triple_config_3.cfg'
                    # NEAT configuration
                    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                         config_file)

                    filename_prefix = str(seed) + "_" + str(ra) + "_"+data_value+"step_"+str(learningStepCount)+"gen_"+str(gen_count)
                    with open(os.path.join("part1", "winner" + filename_prefix), 'rb') as f:
                        winner = pickle.load(f)
                        net = neat.nn.RecurrentNetwork.create(winner, config)
                        for testStepCount in [1,2,3,4]:
                            print("lerning step: ", str(learningStepCount))
                            print("test step: ", str(testStepCount))
                            results = []
                            for num in ["_1","_2","_3"]:
                                results.append(run_sim(winner,config,num,data_value,testStepCount))
                            print(results)
                                
                            with open(os.path.join("testing", "results" + filename_prefix+ "_testing_step_"+str(testStepCount) + ".txt"), 'w') as file:
                                file.write(str(results))
                                                
                        visualize.draw_net(config, winner, view=False,
                                           filename=str(os.path.join("testing","net"+filename_prefix+".gv")))
                                    


if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    test()


