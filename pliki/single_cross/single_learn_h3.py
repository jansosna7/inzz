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

street_net_name = "single"
epsilon = 0.01
learningStepCount = 1

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
    model = neat.nn.FeedForwardNetwork.create(genome, config)

    data_value = config.data_value
    stepThreshold = config.stepThreshold
    
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S")
    random_string = str(uuid.uuid4())
    name = 'z' + street_net_name + random_string + time_string
    summary_file = os.path.join(
        "summaries", f"{name}summary_{street_net_name}.xml")
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
    filename = name + 'single_cross.sumocfg'
    with open(filename, 'w') as file:
        file.write(xml_content)

    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary,  "--no-warnings", "-c", filename,
                 "--time-to-teleport", "-1"])
    
    """execute the TraCI control loop"""
    junctions = [["x", "C3", "x"],
                 ["B2", "C2", "D2"],
                 ["x", "C1", "x"]]
    last_change = 0
    phase = 0
    phase_new = 0
    phase_adjusted = 0

    junction_i = 1
    junction_j = 1


    """execute the TraCI control loop"""
    
    
    #skip until first car approches
    for i in range(41):
        traci.simulationStep()
        
    step = 0

    traci.trafficlight.setPhase(junctions[junction_i][junction_j], 0)

    while traci.simulation.getMinExpectedNumber() > 0:
        if(step > stepThreshold):
            traci.close()
            meanT = get_fitness(summary_file, "meanTravelTime")+500
            if os.path.exists(filename):
                os.remove(filename)
                os.remove(summary_file)
            return -meanT

        
        # get tls data from sim
        phase_new = traci.trafficlight.getPhase(junctions[junction_i][junction_j])
        if phase_new == 0:
            phase_adjusted = 1
        elif phase_new == 2:
            phase_adjusted = -1
        else:
            phase_adjusted = 0
        
        if(phase_new != phase):
            last_change = step
        phase = phase_new
        

        xy = []
        xy.append([junction_i, junction_j-1])
        xy.append([junction_i+1, junction_j])
        xy.append([junction_i, junction_j+1])
        xy.append([junction_i-1, junction_j])
        raw_data = []
        raw_data.append(phase_adjusted)
        raw_data.append(step - last_change)
        # get lane usage data from sim
        for pair in xy:
            lane = junctions[pair[0]][pair[1]] + junctions[junction_i][junction_j] + "_0"
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

        if(judge(step - last_change, phase, prediction[0])):
            traci.trafficlight.setPhase(junctions[junction_i][junction_j], phase+1)
            last_change = step

        for i in range(learningStepCount):
            traci.simulationStep()
        step += learningStepCount

    traci.close()
    meanT = get_fitness(summary_file, "meanTravelTime")
    if os.path.exists(filename):
        os.remove(filename)
        os.remove(summary_file)
    return -meanT

def repeat_run_sim(genome, config, rou):
    # SUMO or multoprocessing can cause crashes, so try several times
    result = -500
    try:
        result = run_sim(genome, config, rou)
    except Exception as e:
        print(e)
        with open("log.txt", "a") as file:
            file.write("1 " + str(e) + "\n")
        time.sleep(0.6)
        try:
            result = run_sim(genome, config, rou)
        except Exception as e2:
            print(e2)
            with open("log.txt", "a") as file:
                file.write("2 " + str(e2) + "\n")
            time.sleep(1.2)
            try:
                result = run_sim(genome, config, rou)
            except Exception as e3:
                print(e3)
                with open("log.txt", "a") as file:
                    file.write("3 " + str(e3) + "\n")
                time.sleep(2.4)
                try:
                    result = run_sim(genome, config, rou)
                except Exception as e4:
                    print(e4)
                    with open("log.txt", "a") as file:
                        file.write("4 " + str(e4) + "\n")
                    time.sleep(1)
    return result

def eval_genome(genome, config):
    fitness = repeat_run_sim(genome, config, "_5")
    return fitness

def learn(prefix, data_value, stepThreshold):
    random.seed(42 + prefix)
    # NEAT configuration
    
    if(data_value == "VehicleNumber" or data_value == "HaltingNumber" or data_value == "MeanSpeed"):
        config_file_path = "single_config_1.cfg"
    elif(data_value == "VehicleNumberAndHaltingNumber" or data_value == "HaltingNumberAndMeanSpeed" or data_value == "MeanSpeedAndVehicleNumber"):
        config_file_path = "single_config_2.cfg"
    elif(data_value == "All"):
        config_file_path = "single_config_3.cfg"

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file_path)

    config.data_value = data_value
    config.stepThreshold = stepThreshold
    
    #file name prefix
    PDS = str(prefix) + "_data_" + data_value + "_stepThreshold_" + str(stepThreshold)

    try:
        # Create NEAT population
        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.Checkpointer(generation_interval=60, time_interval_seconds=1800,
                         filename_prefix=os.path.join("results", "neat-checkpoint"+PDS)+"-"))

        cpus = multiprocessing.cpu_count()
        pe = neat.ParallelEvaluator(cpus, eval_genome)

        gen = 0
        start_time = time.time()
        for generations in [80]:
            gen += generations
            winner = pop.run(pe.evaluate, generations)
            PDSG = PDS + "_gen_" + str(gen)

            # save some data
            with open(os.path.join("results", "winner_"+PDSG), 'wb') as f:
                pickle.dump(winner, f)

            with open(os.path.join("results", "stats_"+PDSG), 'wb') as f:
                pickle.dump(stats, f)

            visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(
                "results", "fitness_"+PDSG+".svg"))
            visualize.plot_species(stats, view=False, filename=os.path.join(
                "results", "speciation_"+PDSG+".svg"))
            visualize.draw_net(config, winner, view=False, filename=os.path.join(
                "results", "drawing_"+PDSG))

            result = eval_genome(winner, config)
            end_time = time.time()
            elapsed_time = int(end_time - start_time)
            with open("raw_results.txt", "a") as file:
                file.write(str(result) + ",")
                file.write(str(elapsed_time) + ",")
                file.write(PDSG + "\n")

    except Exception as e:
        time.sleep(2)
        with open("raw_results.txt", "a") as file:
            file.write(str(e))
            file.write("\n")
            file.write(str(0) + ",")
            file.write(PDS + "\n")
            
def repeat_learn():
    for data_value in ["VehicleNumber", "HaltingNumber","VehicleNumberAndHaltingNumber","HaltingNumberAndMeanSpeed","All"]:
        for stepThreshold in [3000, 4000]:
            for prefix in [14,15,16]:
                learn(prefix, data_value, stepThreshold)

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.makedirs("results", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    repeat_learn()
