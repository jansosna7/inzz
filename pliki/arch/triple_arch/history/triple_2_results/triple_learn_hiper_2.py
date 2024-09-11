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

learningStepCount = 1
street_net_name = "triple"

epsilon = 0.15

harsh_time = 1.9
harsh_result = 1.22

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


def sim(genome, config, num, data_value):
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S.%f")
    characters = string.ascii_letters
    random_string = ''.join(random.choices(characters, k=70))
    name = 'z' + street_net_name + random_string + time_string
    summary_file = os.path.join("summaries", f"{name}summary_{street_net_name}.xml")
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

        for i in range(learningStepCount):
            traci.simulationStep()
        step += learningStepCount

    traci.close()
    meanT = get_fitness(summary_file, "meanTravelTime")

    if os.path.exists(filename):
        os.remove(filename)
        os.remove(summary_file)
    return -meanT



def run_sim(genome, config, rou, data):
    result = -500
    try:
        result = sim(genome, config, rou, data)
    except Exception as e:
        print(e)
        with open("log.txt", "a") as file:
            file.write(str(e) + "\n")
        time.sleep(0.4)
        try:
            result = sim(genome, config, rou, data)
        except Exception as e2:
            print(e2)
            with open("log.txt", "a") as file:
                file.write(str(e2) + "\n")
            time.sleep(1)
            try:
                result = sim(genome, config, rou, data)
            except Exception as e3:
                print(e3)
                with open("log.txt", "a") as file:
                    file.write(str(e3) + "\n")
                time.sleep(2)
                try:
                    result = sim(genome, config, rou, data)
                except Exception as e4:
                    print(e4)
                    with open("log.txt", "a") as file:
                        file.write(str(e4) + "\n")
                    time.sleep(0.1)
                    traci.close()
    return result


def eval_genome_vn(genome, config):
    data = "VehicleNumber"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


def eval_genome_hn(genome, config):
    data = "HaltingNumber"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


def eval_genome_ms(genome, config):
    data = "MeanSpeed"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


def eval_genome_vh(genome, config):
    data = "VehicleNumberAndHaltingNumber"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


def eval_genome_hs(genome, config):
    data = "HaltingNumberAndMeanSpeed"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


def eval_genome_sv(genome, config):
    data = "MeanSpeedAndVehicleNumber"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


def eval_genome_all(genome, config):
    data = "All"
    fitness_1 = run_sim(genome, config, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1000
    fitness_2 = run_sim(genome, config, "_2", data)
    if(fitness_2 < result_f2_1):
        return (fitness_2 - 500) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, "_3", data)
    return fitness_3 + int(fitness_2)/100000


    
def eval_winner(winner,config,data_value):
    result = 0
    if(data_value == "VehicleNumber"):
        result = eval_genome_vn(winner, config)
    elif(data_value == "HaltingNumber"):
        result = eval_genome_hn(winner, config)
    elif(data_value == "MeanSpeed"):
        result = eval_genome_ms(winner, config)
    elif(data_value == "VehicleNumberAndHaltingNumber"):
        result = eval_genome_vh(winner, config)
    elif(data_value == "HaltingNumberAndMeanSpeed"):
        result = eval_genome_hs(winner, config)
    elif(data_value == "MeanSpeedAndVehicleNumber"):
        result = eval_genome_hs(winner, config)
    elif(data_value == "All"):
        result = eval_genome_all(winner, config)
    return result

def learn():

    seed = 42
    with open("seed_no.txt", "r") as fs:
        content = fs.read()
        try:
            seed = int(content)
        except ValueError:
            seed = random.randint(1, 1999)
    with open("seed_no.txt", "w") as fs:
        fs.write(str(seed + 1))
    random.seed(seed)





    for ra in range(20):
        for data_value in ["All"]:#["VehicleNumber","HaltingNumber","MeanSpeed","VehicleNumberAndHaltingNumber","HaltingNumberAndMeanSpeed","MeanSpeedAndVehicleNumber","All"]:
            data_value = "All"
            if(data_value == "VehicleNumber" or data_value == "HaltingNumber" or data_value == "MeanSpeed"):
                config_file = 'triple_config_1.cfg'
            elif(data_value == "VehicleNumberAndHaltingNumber" or data_value == "HaltingNumberAndMeanSpeed" or data_value == "MeanSpeedAndVehicleNumber"):
                config_file = 'triple_config_2.cfg'
            elif(data_value == "All"):
                config_file = 'triple_config_3.cfg'

            # NEAT configuration
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)
            config.pop_size = 400
            config.num_hidden = random.choice([0,3])
            config.initial_connection = random.choice(["full","partial_nodirect 0.5"])
            config.feed_forward = random.choice(["False","True"])
            config.compatibility_disjoint_coefficient = random.choice([0.4,0.5,0.6])
            config.compatibility_weight_coefficient = random.choice([0.7,0.8])
            config.conn_add_prob = random.choice([0.65,0.7,0.75])
            config.conn_del_prob = config.conn_add_prob*0.9
            config.node_add_prob = random.choice([0.35,0.4,0.45])
            config.node_del_prob = config.conn_add_prob*0.9
            config.bias_mutate_rate = random.choice([0.3,0.4,0.5,0.6,0.7])
            config.bias_mutate_power = random.choice([1,2,4])
            config.bias_max_value = random.choice([50,100])
            config.bias_min_value = -1 * config.bias_max_value
            config.response_mutate_rate = random.choice([0.3,0.4,0.5,0.6])
            config.response_mutate_power = random.choice([2,4])
            config.weight_mutate_rate = random.choice([0.7,0.8,0.9])
            config.weight_replace_rate = random.choice([0.3,0.4,0.5,0.6])
            config.weight_mutate_power = random.choice([1,2,4])
            config.compatibility_threshold = random.choice([9,10,12,15])
            config.max_stagnation = random.choice([15,20,25])
            config.species_elitism = random.choice([2,3,4])
            config.elitism = random.choice([1,2])
            config.survival_threshold = random.choice([0.2,0.33,0.5])
            config.min_species_size = random.choice([9,10,12,15])

            # Create NEAT population
            pop = neat.Population(config)
            stats = neat.StatisticsReporter()
            pop.add_reporter(stats)
            pop.add_reporter(neat.StdOutReporter(True))
            pop.add_reporter(neat.Checkpointer(generation_interval=50, time_interval_seconds=1800, filename_prefix=os.path.join("results","neat-checkpoint"+str(seed) + "_" + str(ra)+"_"+data_value+"_"+str(learningStepCount)+"-")))

            cpus = multiprocessing.cpu_count()
            if(data_value == "VehicleNumber"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_vn)
            elif(data_value == "HaltingNumber"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_hn)
            elif(data_value == "MeanSpeed"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_ms)
            elif(data_value == "VehicleNumberAndHaltingNumber"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_vh)
            elif(data_value == "HaltingNumberAndMeanSpeed"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_hs)
            elif(data_value == "MeanSpeedAndVehicleNumber"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_sv)
            elif(data_value == "All"):
                pe = neat.ParallelEvaluator(
                    cpus, eval_genome_all)

            gen = 0
            winner = None
            try:
                for generations in [250]:
                    gen += generations
                    winner = pop.run(pe.evaluate, generations)
                    prefix = str(seed) + "_" + str(ra)+"gen_"+str(gen)
                    
                result = eval_winner(winner,config,data_value)

                with open(os.path.join("results", f"winner{prefix}"), 'wb') as f:
                    pickle.dump(winner, f)

                with open("hipertuning"+str(seed)+"_"+str(ra)+".txt", "a") as file:
                    file.write(str(result) + ",")
                    file.write(str(seed) + ",")
                    file.write(str(ra) + ",")
                    file.write(str(config.initial_connection) + ",")
                    file.write(str(config.feed_forward) + ",")
                    file.write(str(config.compatibility_disjoint_coefficient) + ",")
                    file.write(str(config.compatibility_weight_coefficient) + ",")
                    file.write(str(config.conn_add_prob) + ",")
                    file.write(str(config.node_add_prob) + ",")
                    file.write(str(config.bias_mutate_rate) + ",")
                    file.write(str(config.bias_mutate_power) + ",")
                    file.write(str(config.bias_max_value) + ",")
                    file.write(str(config.response_mutate_rate) + ",")
                    file.write(str(config.response_mutate_power) + ",")
                    file.write(str(config.weight_mutate_rate) + ",")
                    file.write(str(config.weight_replace_rate) + ",")
                    file.write(str(config.weight_mutate_power) + ",")
                    file.write(str(config.compatibility_threshold) + ",")
                    file.write(str(config.max_stagnation) + ",")
                    file.write(str(config.species_elitism) + ",")
                    file.write(str(config.elitism) + ",")
                    file.write(str(config.survival_threshold) + ",")
                    file.write(str(config.min_species_size) + "\n")
                    #file.write(data_value + "\n")

            except Exception as e:
                time.sleep(2)
                with open("hipertuning"+str(seed)+".txt", "a") as file:
                    file.write(str(e))
                    file.write("\n")
                    file.write(str(0) + ",")
                    file.write(str(seed) + ",")
                    file.write(str(ra) + "\n")


def repeat_learn():
    for i in range(1):
        try:
            learn()
        except Exception as e:
            time.sleep(5)
            print(e)


if __name__ == "__main__":
    #print(os.environ)
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    repeat_learn()
