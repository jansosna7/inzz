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

learningStepCount = 1
street_net_name = "double"
epsilon = 0.01


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

    harsh_time = config.harsh_time
    harsh_result = config.harsh_result

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
        if(step > step_threshold):
            traci.close()
            meanT = get_fitness(summary_file, "meanTravelTime")+200
            if os.path.exists(filename):
                os.remove(filename)
                os.remove(summary_file)
            return -meanT

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
    meanT = get_fitness(summary_file, "meanTravelTime")

    if os.path.exists(filename):
        os.remove(filename)
        os.remove(summary_file)
    return -meanT



def repeat_run_sim(genome, config, rou):
    # SUMO or multoprocessing can cause crashes, so try several times
    result = -2500
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
    config.step_threshold = config.t1
    fitness_1 = repeat_run_sim(genome, config, "_1")
    if(fitness_1 < config.result_f1):
        return fitness_1 - 1500
    
    config.step_threshold = config.t2
    fitness_2 = repeat_run_sim(genome, config, "_2")
    if(fitness_2 < config.result_f2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    
    config.step_threshold = config.t3
    fitness_3 = repeat_run_sim(genome, config, "_3")
    if(fitness_3 < config.result_f3):
        return (fitness_3 - 500) + int(fitness_2)/100000
    
    config.step_threshold = config.t4
    fitness_4 = repeat_run_sim(genome, config, "_4")
    return (fitness_4) + int(fitness_3)/100000


def learn(prefix, data_value, harsh_time,harsh_result, pop_size):
    t1 = 259 * harsh_time
    t2 = 425 * harsh_time
    t3 = 613 * harsh_time
    t4 = 1182 * harsh_time

    result_f1 = -1*(155*harsh_result)
    result_f2 = -1*(163*harsh_result)
    result_f3 = -1*(176*harsh_result)

    random.seed(42+prefix)

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
    config.harsh_time = harsh_time
    config.harsh_result = harsh_result
    config.t1 = t1
    config.t2 = t2
    config.t3 = t3
    config.t4 = t4
    config.result_f1 = result_f1
    config.result_f2 = result_f2
    config.result_f3 = result_f3

    PDTR = str(prefix) + "_pop_" + str(pop_size) + "_data_" + data_value + "_ht_" + str(harsh_time) + "_hr_" + str(harsh_result)
 
    # Create NEAT population
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(generation_interval=60, time_interval_seconds=1800, filename_prefix=os.path.join("results", "neat-checkpoint"+PDTR)+"-"))

    cpus = multiprocessing.cpu_count()

    pe = neat.ParallelEvaluator(cpus, eval_genome)

    gen = 0
    winner = None
    try:
        start_time = time.time()
        for generations in [10000/pop_size,10000/pop_size]:
            gen += generations
            winner = pop.run(pe.evaluate, generations)
            PDTRG = PDTR + "_gen_" + str(gen)
            with open(os.path.join("results","winner"+PDTRG), 'wb') as f:
                pickle.dump(winner, f)

            with open(os.path.join("results","stats"+PDTRG), 'wb') as f:
                pickle.dump(stats, f)
            visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(
                "results", "fitness_"+PDTRG+".svg"))
            visualize.plot_species(stats, view=False, filename=os.path.join(
                "results", "speciation_"+PDTRG+".svg"))
            visualize.draw_net(config, winner, view=False, filename=os.path.join(
                "results", "drawing_"+PDTRG))

        result = eval_genome(winner, config)
        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        with open("hipertuning.txt", "a") as file:
            file.write(str(result) + ",")
            file.write(str(elapsed_time) + ",")
            file.write(str(pop_size) + ",")
            file.write(data_value + ",")
            file.write(str(harsh_time) + ",")
            file.write(str(harsh_result) + ",")
            file.write(PDTRG + "\n")


    except Exception as e:
        time.sleep(2)
        with open("hipertuning.txt", "a") as file:
            file.write(str(e))
            file.write("\n")


def repeat_learn():
    for pop_size in [100,200]:
        for data_value in ["VehicleNumber", "HaltingNumber","VehicleNumberAndHaltingNumber","HaltingNumberAndMeanSpeed","All"]:
            for harsh_result in [1.05, 1.25, 1.5]:
                for harsh_time in [1.5, 2, 3]:
                    for prefix in range(2):
                        try:
                            learn(prefix, data_value, harsh_time,harsh_result, pop_size)
                        except Exception as e:
                            time.sleep(5)
                            print(e)


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
