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


street_net_name = "single"



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

def run_sim(num,tls_type):
    #prepare sumo config file:
    name = 'z' + tls_type + num
    summary_file = os.path.join(
        "emissions_test", f"{name}_emissions_tripinfo_{street_net_name}.xml")
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="'''+street_net_name+tls_type+'''.net.xml"/>
            <route-files value="'''+street_net_name+num+'''.rou.xml"/>
        </input>
        <output>
            <tripinfo-output value="'''+summary_file+'''"/>
        </output>
        <emissions>
            <device.emissions.probability value="1.0"/>
        </emissions>
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
    fuels = extract_emissions(summary_file, "fuel_abs")
    meanF = sum(fuels) / len(fuels)
    if os.path.exists(filename):
        os.remove(filename)
    return meanF



def run_control():
    filename = os.path.join("emissions_test","control_results.txt")
    with open(filename, 'w') as file:
        for tls_type in ["_no_lights","_static","_actuated"]:
            results = []
            results.append(run_sim('_1', tls_type))
            results.append(run_sim('_2', tls_type))
            results.append(run_sim('_3', tls_type))
            results.append(run_sim('_4', tls_type))
            results.append(run_sim('_5', tls_type))
            
            file.write(str(tls_type)+" "+str(results)+"\n")

                    

if __name__ == "__main__":
    os.makedirs("control_summaries", exist_ok=True)
    os.makedirs("emissions_test", exist_ok=True)

    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    run_control()
