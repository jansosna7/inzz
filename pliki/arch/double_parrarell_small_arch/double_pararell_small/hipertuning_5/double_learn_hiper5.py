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
street_net_name = "double"
distance = "_500"

harsh_time = 1.7
harsh_result = 1.18


t1 = 260 * harsh_time
t2 = 430 * harsh_time
t3 = 620 * harsh_time
t4 = 1180 * harsh_time

result_f1_1 = -1*(155*harsh_result + 15)
result_f1_2 = -1*(155*harsh_result)
result_f2_1 = -1*(163*harsh_result + 10)
result_f2_2 = -1*(163*harsh_result)
result_f3_1 = -1*(176*harsh_result)
result_f4 = -237

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
        if(prediction > 0.7 and currentPhase == 0):
            # switch
            return True
        elif(prediction < 0.3 and currentPhase == 2):
            # switch
            return True
    # do nothing
    return False


def sim(genome, config, lenght, num, data_value):
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S.%f")
    characters = string.ascii_letters
    random_string = ''.join(random.choices(characters, k=50))
    name = 'z' + street_net_name + random_string + time_string
    summary_file = "summaries//"+name+"summary_"+street_net_name+".xml"
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

    step_threshold = t4  # for rou_4
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



def run_sim(genome, config, distance, rou, data):
    result = -500
    try:
        result = sim(genome, config, distance, rou, data)
    except Exception as e:
        print(e)
        time.sleep(3)
        try:
            result = sim(genome, config, distance, rou, data)
        except Exception as e2:
            print(e2)
            time.sleep(5)
            try:
                result = sim(genome, config, distance, rou, data)
            except Exception as e3:
                print(e3)
                time.sleep(0.1)
                traci.close()
    return result


def eval_genome_vn(genome, config):
    data = "VehicleNumber"
    fitness_1 = run_sim(genome, config, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(genome, config, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(genome, config, distance, "_4", data)
    return fitness_4


def eval_genome_hn(genome, config):
    data = "HaltingNumber"
    fitness_1 = run_sim(genome, config, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(genome, config, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(genome, config, distance, "_4", data)
    return fitness_4


def eval_genome_ms(genome, config):
    data = "MeanSpeed"
    fitness_1 = run_sim(genome, config, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(genome, config, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(genome, config, distance, "_4", data)
    return fitness_4


def eval_genome_vh(genome, config):
    data = "VehicleNumberAndHaltingNumber"
    fitness_1 = run_sim(net, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(net, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(net, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(net, distance, "_4", data)
    return fitness_4


def eval_genome_hs(genome, config):
    data = "HaltingNumberAndMeanSpeed"
    fitness_1 = run_sim(genome, config, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(genome, config, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(genome, config, distance, "_4", data)
    return fitness_4


def eval_genome_sv(genome, config):
    data = "MeanSpeedAndVehicleNumber"
    fitness_1 = run_sim(genome, config, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(genome, config, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(genome, config, distance, "_4", data)
    return fitness_4


def eval_genome_all(genome, config):
    data = "All"
    fitness_1 = run_sim(genome, config, distance, "_1", data)
    if(fitness_1 < result_f1_1):
        return fitness_1 - 1500
    fitness_2 = run_sim(genome, config, distance, "_2", data)
    if(fitness_2 < result_f2_1 or fitness_1 < result_f1_2):
        return (fitness_2 - 1000) + int(fitness_1)/100000
    fitness_3 = run_sim(genome, config, distance, "_3", data)
    if(fitness_3 < result_f3_1 or fitness_2 < result_f2_2):
        return (fitness_3 - 500) + int(fitness_2)/100000
    fitness_4 = run_sim(genome, config, distance, "_4", data)
    if(fitness_4 < result_f4):
        return (fitness_4) + int(fitness_3)/100000
    return fitness_4


def learn():
    pop_sizes = [100, 150, 200]
    num_hidden = [0]
    comp_d_c = [0.2, 0.5, 0.8, 1]
    comp_w_c = [0.1, 0.2, 0.5, 0.6, 0.8, 1]
    probs = [0.02, 0.05, 0.1, 0.2, 0.33, 0.4, 0.5,
             0.6, 0.7]  # conn add & del, node add & del
    diff = [0.99, 0.98, 0.95, 0.9, 0.8, 0.66, 0.5, 0.1]
    node_diff = [0.99, 0.98, 0.95, 0.9, 0.8, 0.66, 0.5]
    activations = [0, 0.03, 0.3]
    aggregations = [0, 0.03, 0.3]
    replace = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    rate = [0.01, 0.05, 0.1, 0.33, 0.5, 0.66, 0.8, 0.9]
    power = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4, 6]
    biases = [10,50,100]
    responses = [1, 8, 10, 12]
    weights = [1, 10, 20, 30]

    bias_mutate_rate = [0.3, 0.5, 0.66]

    enabled_mutate_rate = [0.01, 0.1]

    compatibility = [4, 4.5, 5, 5.5, 6]
    stagnation = [15, 22, 33]
    species_elitism = [0, 1, 2, 3, 4]
    elitism = [0, 1, 2, 3]
    species_size = [11, 13, 15, 17]
    survival_t = [0.33, 0.5, 0.66]

    dividers = [1,1,1]

    noise = 0.13
    noise_small = 0.02
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

    for ra in range(100):
        for i in range(1):
            data_value = "All"
            pop_size = random.choice(pop_sizes)

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

            config.pop_size = pop_size
            config.num_hidden = random.choice(num_hidden)
            config.compatibility_disjoint_coefficient = random.choice(
                comp_d_c) * (1 + random.uniform(-noise_small, noise_small))
            config.compatibility_weight_coefficient = random.choice(
                comp_w_c) * (1 + random.uniform(-noise_small, noise_small))
            config.conn_add_prob = random.choice(
                probs) * (1 + random.uniform(-noise_small, noise_small))
            config.conn_delete_prob = random.choice(
                diff) * config.conn_add_prob
            config.node_add_prob = random.choice(
                probs) * (1 + random.uniform(-noise_small, noise_small))
            config.node_delete_prob = random.choice(
                node_diff) * config.node_add_prob
            config.activation_mutate_rate = random.choice(activations)
            config.aggregation_mutate_rate = random.choice(aggregations)

            config.bias_replace_rate = random.choice(
                replace) * (1 + random.uniform(-noise_small, noise_small))
            config.bias_mutate_rate = random.choice(
                bias_mutate_rate) * (1 + random.uniform(-noise_small, noise_small))
            config.bias_mutate_power = random.choice(
                power) * (1 + random.uniform(-noise, noise))
            config.response_replace_rate = random.choice(
                replace) * (1 + random.uniform(-noise_small, noise_small))
            config.response_mutate_rate = random.choice(
                rate) * (1 + random.uniform(-noise_small, noise_small))
            config.response_mutate_power = random.choice(
                power) * (1 + random.uniform(-noise, noise))
            config.weight_replace_rate = random.choice(
                replace) * (1 + random.uniform(-noise_small, noise_small))
            config.weight_mutate_rate = random.choice(
                rate) * (1 + random.uniform(-noise_small, noise_small))
            config.weight_mutate_power = random.choice(
                power) * (1 + random.uniform(-noise, noise))

            config.enabled_mutate_rate = random.choice(
                enabled_mutate_rate) * (1 + random.uniform(-noise, noise))

            m_bias = random.choice(biases) * (1 + random.uniform(-noise, noise))
            m_response = random.choice(responses) * (1 + random.uniform(-noise, noise))
            m_weight = random.choice(weights) * (1 + random.uniform(-noise, noise))

            config.bias_max_value = m_bias
            config.bias_min_value = -m_bias
            config.response_max_value = m_response 
            config.response_min_value = -m_response
            config.weight_max_value = m_weight 
            config.weight_min_value = -m_weight

            init_divider = [1.5, 2, 3, 5, 10]
            dividers[0] = random.choice(init_divider)
            dividers[1] = random.choice(init_divider)
            dividers[2] = random.choice(init_divider)

            config.bias_init_stdev = m_bias / \
                dividers[0] * (1 + random.uniform(-noise, noise))
            config.response_init_stdev = m_response / \
                dividers[1] * (1 + random.uniform(-noise, noise))
            config.weight_init_stdev = m_weight / \
                dividers[2] * (1 + random.uniform(-noise, noise))

            config.compatibility_threshold = random.choice(compatibility)
            config.max_stagnation = random.choice(stagnation)
            config.species_elitism = random.choice(species_elitism)
            config.elitism = random.choice(elitism)
            config.survival_threshold = random.choice(survival_t)
            config.min_species_size = random.choice(species_size)

            # Create NEAT population
            pop = neat.Population(config)
            stats = neat.StatisticsReporter()
            pop.add_reporter(stats)
            pop.add_reporter(neat.StdOutReporter(True))
            #pop.add_reporter(neat.Checkpointer(generation_interval=50, filename_prefix='results/neat-checkpoint'+str(seed) + "_" + str(ra)+'_pop_size_'+str(pop_size)+"_"+data_value+"_"+str(learningStepCount)+"-"))

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
                for generations in [32000/pop_size]:
                    gen += generations
                    winner = pop.run(pe.evaluate, generations)
                    prefix = str(seed) + "_" + str(ra)+"_pop_size_"+str(pop_size) + \
                        "_"+data_value+"step_" + \
                        str(learningStepCount)+"gen_"+str(gen)
                    with open("results/winner"+prefix, 'wb') as f:
                        pickle.dump(winner, f)

                    with open("results/stats"+prefix, 'wb') as f:
                        pickle.dump(stats, f)
                    visualize.plot_stats(
                         stats, ylog=False, view=False, filename="results/fitness"+prefix+".svg")

                result = 0
                if(data_value == "VehicleNumber"):
                    result = eval_genome_vn(winner, config)
                if(data_value == "HaltingNumber"):
                    result = eval_genome_hn(winner, config)
                if(data_value == "MeanSpeed"):
                    result = eval_genome_ms(winner, config)
                if(data_value == "VehicleNumberAndHaltingNumber"):
                    result = eval_genome_vh(winner, config)
                if(data_value == "HaltingNumberAndMeanSpeed"):
                    result = eval_genome_hs(winner, config)
                if(data_value == "MeanSpeedAndVehicleNumber"):
                    result = eval_genome_sh(winner, config)
                if(data_value == "All"):
                    result = eval_genome_all(winner, config)

                with open("hipertuning"+str(seed)+".txt", "a") as file:
                    file.write(str(result) + ",")
                    file.write(str(seed) + ",")
                    file.write(str(ra) + ",")

                    file.write(str(pop_size) + ",")
                    
                    file.write(str(config.compatibility_threshold) + ",")
                    file.write(str(config.max_stagnation) + ",")
                    file.write(str(config.species_elitism) + ",")
                    file.write(str(config.elitism) + ",")
                    file.write(str(config.min_species_size) + ",")
                    file.write(str(config.survival_threshold) + ",")

                    file.write(str(config.conn_add_prob) + ",")
                    file.write(str(config.conn_delete_prob) + ",")
                    file.write(str(config.node_add_prob) + ",")
                    file.write(str(config.node_delete_prob) + ",")

                    file.write(str(config.enabled_mutate_rate) + ",")
                    file.write(str(config.activation_mutate_rate) + ",")
                    file.write(str(config.aggregation_mutate_rate) + ",")

                    file.write(
                        str(config.compatibility_disjoint_coefficient) + ",")
                    file.write(
                        str(config.compatibility_weight_coefficient) + ",")


                    file.write(str(config.bias_init_stdev) + ",")
                    file.write(str(dividers[0]) + ",")
                    file.write(str(config.bias_max_value) + ",")
                    file.write(str(config.bias_mutate_power) + ",")
                    file.write(str(config.bias_mutate_rate) + ",")
                    file.write(str(config.bias_replace_rate) + ",")

                    file.write(str(config.response_init_stdev) + ",")
                    file.write(str(dividers[1]) + ",")                   
                    file.write(str(config.response_max_value) + ",")
                    file.write(str(config.response_mutate_power) + ",")
                    file.write(str(config.response_mutate_rate) + ",")
                    file.write(str(config.response_replace_rate) + ",")

                    file.write(str(config.weight_init_stdev) + ",")
                    file.write(str(dividers[2]) + ",")
                    file.write(str(config.weight_max_value) + ",")
                    file.write(str(config.weight_mutate_power) + ",")
                    file.write(str(config.weight_mutate_rate) + ",")
                    file.write(str(config.weight_replace_rate) + "\n")

            except Exception as e:
                time.sleep(2)
                with open("hipertuning"+str(seed)+".txt", "a") as file:
                    file.write(str(e))
                    file.write("\n")
                    file.write(str(0) + ",")
                    file.write(str(seed) + ",")
                    file.write(str(ra) + "\n")


def repeat_learn():
    for i in range(100):
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
