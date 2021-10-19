#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import traci
import sys
import os
from sumolib import checkBinary
import xml.etree.ElementTree as ET

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def getMeanSpeed():
    tree = ET.parse("statistics.xml")
    root = tree.getroot()
    return root.find('vehicleTripStatistics').attrib['speed']

def changeDuration(duration):
    tree = ET.parse("circularControlNet.net.xml")
    root = tree.getroot()
    for counter, phase in enumerate(root.iter("phase")):
        if counter % 2 == 0:
            phase.set("duration", str(duration))
    tree.write("circularControlNet.net.xml")

def run():
    step = 0
    traci.trafficlight.setPhase("m", 0)
    detectedVehicles = 0
    lastVehicleIDs = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        '''
        #detectedVehicles += traci.inductionloop.getLastStepVehicleNumber("detect")
        vehicleIDs = traci.inductionloop.getLastStepVehicleIDs("detect")
        for id in vehicleIDs:
            if not id in lastVehicleIDs:
                detectedVehicles += 1
        lastVehicleIDs = vehicleIDs
        '''
        step += 1
    #print("Vehicles passed detector: ", detectedVehicles)
    traci.close()
    sys.stdout.flush()


if __name__ == '__main__':
    sumoBinary = checkBinary('sumo')
    configPath = os.path.abspath("circularControl.sumocfg")
    netPath = os.path.abspath("circularControlNet.net.xml")

    #optimization loop
    greenDuration = 20
    stepSize = 3
    direction = 1
    changeDuration(greenDuration)

    #starting point
    traci.start([sumoBinary, "-c", configPath,
                                "--tripinfo-output", "tripinfo.xml",
                                "--statistic-output", "statistics.xml"])
    run()
    avgSpeed = getMeanSpeed()
    currentBestSpeed = avgSpeed

    greenDuration -= stepSize
    changeDuration(greenDuration)

    traci.start([sumoBinary, "-c", configPath,
                                "--tripinfo-output", "tripinfo.xml",
                                "--statistic-output", "statistics.xml"])
    run()
    avgSpeed = getMeanSpeed()
    
    print(avgSpeed, currentBestSpeed)
    if avgSpeed > currentBestSpeed:
        direction = -stepSize
        currentBestSpeed = avgSpeed
    else:
        direction = stepSize

    while True:
        greenDuration -= direction
        changeDuration(greenDuration)

        #start server
        traci.start([sumoBinary, "-c", configPath,
                                "--tripinfo-output", "tripinfo.xml",
                                "--statistic-output", "statistics.xml"])

        #one simulation run
        run()

        avgSpeed = getMeanSpeed()
        if avgSpeed < currentBestSpeed:
            greenDuration += direction
            break
        else:
            currentBestSpeed = avgSpeed
        print("Best green phase duration: ", greenDuration)
        print("Best Avg Speed: ", currentBestSpeed)
    
    print("Best green phase duration: ", greenDuration)
    print("Best Avg Speed: ", currentBestSpeed)