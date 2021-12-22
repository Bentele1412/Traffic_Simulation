#!/usr/bin/env python

import numpy as np
import time
import ray
ray.init()

@ray.remote
def calcX(x):
    return x**2

if __name__ == '__main__':
    runs = 1000000
    start = time.time()
    results = []
    for i in range(runs):
        results.append(i**2)
    timeNeeded = time.time()-start

    start = time.time()
    results = ray.get([calcX.remote(x) for x in range(runs)])
    rayTimeNeeded = time.time()-start
    print(timeNeeded)
    print(rayTimeNeeded)