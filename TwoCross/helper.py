#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import traci
import sys
import os
from sumolib import checkBinary
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

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
    tree = ET.parse("twoCross.net.xml")
    root = tree.getroot()
    for counter, phase in enumerate(root.iter("phase")):
        if counter % 4 == 0:
            phase.set("duration", str(duration))
    tree.write("twoCross.net.xml")