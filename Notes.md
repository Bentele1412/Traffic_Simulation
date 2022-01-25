# Ideas for autonomous controller
* SOTL + self-adapting threshold:
  * calculate traffic density at runtime to determine threshold value and green phase times 
  * $\theta$ = k * trafficDensity --> "optimize" k to control the impact of $\theta$-change


## Open Questions
* definition of traffic density
* number of traffic density calculations


## Keep in mind
* Identify stable system state and only measure if system in stable state
* LPSolve provides the inflow rates for internal edges<br/>--> use them to calculate junction utilization<br/> --> calculate minCycleTime and greenPhaseRatios with junction utilization 


## Questions to Vollbrecht
* Where to depart/arrive? Middle of edges - junctions - only outer junctions? 
* LPSolve Output for 2x3Grid and $\epsilon$ = 0.5:
  * for a: 0.200002666702
  * for b: 0.200002666702
  * for c: 0.200002666702

## To-do
* **Done:** Read paper until tuesday
* **Done:** reimplement 2X3 grid from paper with flows
* **Done:** implement SOTL with well defined object structure
* **Done:** implement fixed cycle based control
* Hillclimbing: 
  * **Done:** define gradient per dimension with (positive direction fitness deviation - negative direction fitness deviation)
  * **Done:** Optimization strategies:
    * **Done:** calc all directions, take best and multiply with gradient --> new fitness evaluation needed after updates
    * **Done:** calc all directions, get all fitness increasing gradients and summarize as one gradient --> updates are performed in several directions at once
    * **Done:** iterate over directions and make the step and update for one direction immediately, if a better fitness value is achieved --> fitnessDynamics???
* **Done:** Consider phase switches at switching routes (t=1200/2400) --> maybe green light gets red immediately, but that should not be the case
* **Declined:** Parallelization --> file writings are issues to face
* Bigger Phase shifts --> 50 secs per node
* Random optimization starts
* **Freezed:** Handle more than 2 green phases + more than one red lane
* **Freezed:** Create different scenario
* **In Progress:** Waiting time calculated differently in paper than provided mean waiting time of statistics --> avg per intersection and then avg over all intersections
* compare SOTL and FIX + comparison to paper results --> Cycle based difficult due to flow switches --> phase shifts?
* **Done:** ensure minCycleTime is never undercut
* termination condition of HillClimbing 
* implement all platoon based algorithms
* arterial network to validate CB and learn mor about AdaSOTL behavior --> verification 
* structure helperTwoPhases.py into more files


## Possible paper content
* Szenarios: TwoCross, 2x3Grid, FH bridge
* Traffic load: 900, 1200, 1500 + real life data
* Controller: Cycle based, SOTL, PBSS, AdaSOTL
* Optimizers: HillClimbing, GridSearch, ES (?)


## Experiment results
### HillClimbing 2x3Grid: 20 iters, 5 runs
Strategy 0:<br/>
Optimal fitness: 10.1<br/>
Optimal params: [ 0.7712  9.912  19.954   9.898  20.096  29.88  ]<br/>

Strategy 1:<br/>
Optimal fitness: 10.204<br/>
Optimal params: [ 0.6688 10.099  20.023  10.215  19.992  29.667 ]


### HillClimbing 2x3Grid: single Flows
Flow 1:
66 min and 7.585176 seconds needed.
Found optimum with:
Optimal fitness: 10.93
Optimal params: [ 0.7875 10.225  19.785   9.75   20.47   30.18  ]

Flow 2:
59 min and 34.063972 seconds needed.
Found optimum with:
Optimal fitness: 10.83
Optimal params: [ 0.755 10.305 20.     9.715 19.895 30.135]

Flow 3:
59 min and 38.805740 seconds needed.
Found optimum with:
Optimal fitness: 10.89
Optimal params: [ 0.694 10.235 20.09  10.175 20.155 30.655]


### HillClimbing 2x3Grid: AdaSOTL, 50 iters, 5 runs, strategy 1, 900 vehicles
423 min and 57.934498 seconds needed.
Last evaluation:
Fitness: 8.67
Params: [4.2035 1.1912]

Best:
Optimal fitness: 8.67
Optimal params: [4.2035 1.1912]

Next best after rerunning optimization with previous params:
Optimal fitness: 8.682
Optimal params: [4.183  1.1866]

### HillClimbing 2x3Grid: SOTL, 50 iters, 5 runs, strategy 1, 900 vehicles
189 min and 38.304668 seconds needed.
Last evaluation:
Fitness: 8.48
Params: [31.97]

Best:
Optimal fitness: 8.495999999999999
Optimal params: [31.84]

## Experiment results with setup similar to paper
### ES 2x3Grid: AdaSOTL, 100 iters, 5 runs, 900 vehicles, mu=2, lambda=6
1102 min and 33.877640 seconds needed.
Best:
Optimal fitness: 8.286
Optimal params: [3.18689112 1.61760802]

### ES 2x3Grid: SOTL, 50 iters, 5 runs, 900 vehicles, mu=3, lambda=6
554 min and 32.572487 seconds needed.
Best:
Optimal fitness: 6.798
Optimal params: [-57829.36829758]
--> Sigma exploded

### HillClimbing 2x3Grid: AdaSOTL, 50 iters, 5 runs, strategy 1, 900 vehicles
376 min and 21.285167 seconds needed.
Last evaluation:
Fitness: 8.282
Params: [3.5925 1.4584]

Best:
Optimal fitness: 8.312000000000001
Optimal params: [3.5665  1.45495]

Rerun:
380 min and 5.536006 seconds needed.
Last evaluation:
Fitness: 8.288
Params: [3.95    1.49215]

Best:
Optimal fitness: 8.315999999999999
Optimal params: [3.601  1.4591]

### HillClimbing 2x3Grid: SOTL, 50 iters, 5 runs, strategy 1, 900 vehicles
237 min and 34.809808 seconds needed.
Last evaluation:
Fitness: 6.756
Params: [28.77]

Best:
Optimal fitness: 7.47
Optimal params: [30]