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
import uuid

street_net_name = "double"

def get_meanTime(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if lines and lines[-1].strip() != "</summary>":
        # Append "</summary>" to the file if it's not the last line
        with open(file_path, 'a') as file:
            file.write("</summary>\n")
    
    tree = ET.parse(file_path)
    root = tree.getroot()

    last_step = root.findall('step')[-1]

    return float(last_step.get('meanTravelTime'))

def run_sim(num,tls_type):
    #prepare sumo config file:
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d%H%M%S")
    characters = string.ascii_letters
    random_string = str(uuid.uuid4())
    name = 'z' + street_net_name + random_string + time_string
    summary_file = os.path.join("control_summaries","summary_"+street_net_name+tls_type+num+".xml")
    tripinfos_file = os.path.join("control_summaries","tripinfos_"+street_net_name+tls_type+num+".xml")
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="'''+street_net_name+tls_type+'''.net.xml"/>
            <route-files value="'''+street_net_name+num+'''.rou.xml"/>
        </input>
        <output>
            <summary-output value="'''+summary_file+'''"/>
            <tripinfo-output value="'''+tripinfos_file+'''"/>
        </output>
    </configuration>'''
    filename = name + "single_cross.sumocfg"
    with open(filename, 'w') as file:
        file.write(xml_content)

    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary, "-c", filename,
                 "--time-to-teleport","-1"])
    
    #execute the TraCI control loop
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    meanT = get_meanTime(summary_file)
    if os.path.exists(filename):
        os.remove(filename)
    return -meanT


def learn():
    filename = "control_test_results.txt"
    with open(filename, 'w') as file:
        for tls_type in ["_no_lights","_static","_static_offset","_actuated"]:
            for length in ["_500"]:
                results = []
                results.append(run_sim('_1', tls_type))
                results.append(run_sim('_2', tls_type))
                results.append(run_sim('_3', tls_type))
                results.append(run_sim('_4', tls_type))

                print(tls_type,length,results)
                
                file.write(str(tls_type)+" "+str(length)+" "+str(results)+"\n")

if __name__ == "__main__":
    os.makedirs("control_summaries", exist_ok=True)
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    learn()
