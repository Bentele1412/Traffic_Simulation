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
* Bigger Phase shifts --> 50 secs per node
* Random optimization starts
* **Freezed:** Handle more than 2 green phases + more than one red lane
* **Freezed:** Create different scenario
* **Done:** Waiting time calculated differently in paper than provided mean waiting time of statistics --> avg per intersection and then avg over all intersections
* compare SOTL and FIX + comparison to paper results --> Cycle based difficult due to flow switches --> phase shifts?
* **Done:** ensure minCycleTime is never undercut
* termination condition of HillClimbing 
* **Done:** implement all platoon based algorithms
* **Done:** arterial network to validate CB and learn mor about AdaSOTL behavior --> verification 
* **Done:** structure helperTwoPhases.py into more files
* Parallelization 
* HillClimbing updates
* Additional evaluation functions --> e.g. minimize meanWaitingTime, consider stds in evalFuncs


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
### Distribution of optimization experiments
2x3 Grid:
<table>
<tr>
  <th></th><th>900 vehicles</th><th>1200 vehicles</th><th>1500 vehicles</th>
<tr>
<tr>
  <td>CB</td><td></td><td></td><td></td>
</tr>
<tr>
  <td>SOTL</td><td>Marcus</td><td>Georg</td><td></td>
</tr>
<tr>
  <td>AdaSOTL</td><td>Marcus</td><td>Georg</td><td></td>
</tr>
</table>
Arterial Grid:
<table>
<tr>
  <th></th><th>deltaR_t=0</th><th>deltaR_t=1/16</th><th>deltaR_t=2/16</th>
<tr>
<tr>
  <td>CB</td><td></td><td></td><td></td>
</tr>
<tr>
  <td>SOTL</td><td></td><td></td><td></td>
</tr>
<tr>
  <td>AdaSOTL</td><td></td><td></td><td></td>
</tr>
</table>

### ES 2x3Grid: AdaSOTL, 100 iters, 5 runs, 900 vehicles, mu=2, lambda=6
1102 min and 33.877640 seconds needed.
Best:
Optimal fitness: 8.286
Optimal params: [3.18689112 1.61760802]

### ES 2x3Grid: AdaSOTL, 50 iters, 5 runs, 1200 vehicles, mu=3, lambda=8
525 min and 9.521330 seconds needed.
Best:
Optimal fitness: 8.036
Optimal params: [1.44550967 1.93555081]
--> see dynamics, maybe even better mean speed achieved

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

Corrected version:
275 min and 10.550787 seconds needed.
Last evaluation:
Fitness: 8.142
Params: [3.24260892 1.31881679]

Best:
Optimal fitness: 8.222
Optimal params: [3.20760892 1.31576679]

### HillClimbing 2x3Grid: AdaSOTL, 50 iters, 5 runs, strategy 1, 1200 vehicles
252 min and 15.884611 seconds needed.
Last evaluation:
Fitness: 7.994
Params: [3.98760892 1.41701679]

Best:
Optimal fitness: 8.042
Optimal params: [3.93910892 1.41431679]

### HillClimbing 2x3Grid: AdaSOTL, 50 iters, 5 runs, strategy 1, 1500 vehicles
260 min and 28.675490 seconds needed.
Last evaluation:
Fitness: 7.75
Params: [4.49560892 1.48061679]

Best:
Optimal fitness: 7.802
Optimal params: [4.49260892 1.48011679]

### HillClimbing 2x3Grid: SOTL, 50 iters, 5 runs, strategy 1, 900 vehicles
237 min and 34.809808 seconds needed.
Last evaluation:
Fitness: 6.756
Params: [28.77]

Best:
Optimal fitness: 7.47
Optimal params: [30]

### HillClimbing 2x3Grid: SOTL, 50 iters, 5 runs, strategy 1, 1200 vehicles
79 min and 24.685507 seconds needed.
Last evaluation:
Fitness: 5.464
Params: [50]

Best:
Optimal fitness: 5.52
Optimal params: [45]

### HillClimbing 2x3Grid: CB, 50 iters, 5 runs, strategy 1, 1500 veh
959 min and 30.552535 seconds needed.
Last evaluation:
Fitness: 6.843999999999999
Params: [  0.8  36.   59.  105.   86.  129. ]

Best:
Optimal fitness: 7.0200000000000005
Optimal params: [  0.8  48.   55.   93.   86.  125. ]

### HillClimbing 2x3Grid: CB, 50 iters, 5 runs, strategy 1, 900 veh
688 min and 16.009214 seconds needed.
Last evaluation:
Fitness: 7.616
Params: [  1.5 164.  147.  187.  170.  237. ]

Best:
Optimal fitness: 7.754
Optimal params: [  1.40806634   88.  71.  111.  94.  161.]

### HillClimbing arterial: AdaSOTL, 50 iters, 5 runs, strategy 1, 900 veh --> invalid
207 min and 37.548431 seconds needed.
Last evaluation:
Fitness: 8.372
Params: [3.08710892 1.33361679]

Best:
Optimal fitness: 8.399999999999999
Optimal params: [2.68710892 1.41361679]

### HillClimbing arterial: AdaSOTL, 50 iters, 5 runs, strategy 1, delta_r_t=0
425 min and 29.947640 seconds needed.
Last evaluation:
Fitness: 8.22
Params: [1.18710892 1.89361679]

Best:
Optimal fitness: 8.286000000000001
Optimal params: [1.38710892 1.79361679]

### HillClimbing arterial: AdaSOTL, 50 iters, 5 runs, strategy 1, delta_r_t=1/16
250 min and 1.750037 seconds needed.
Last evaluation:
Fitness: 8.148
Params: [3.48710892 1.43361679]

Best:
Optimal fitness: 8.193999999999999
Optimal params: [3.48710892 1.39361679]

### HillClimbing arterial: AdaSOTL, 50 iters, 5 runs, strategy 1, delta_r_t=2/16
251 min and 9.850723 seconds needed.
Last evaluation:
Fitness: 8.128
Params: [2.58710892 1.53361679]

Best:
Optimal fitness: 8.172
Optimal params: [2.68710892 1.55361679]

### HillClimbing Arterial: SOTL, 50 iters, 5 runs, strategy 1, Delta = 0
95 min and 30.440051 seconds needed.
Last evaluation:
Fitness: 7.453999999999999
Params: [74]

Best:
Optimal fitness: 7.83
Optimal params: [24]

### HillClimbing Arterial: SOTL, 50 iters, 5 runs, strategy 1, Delta = 1/16
65 min and 6.446793 seconds needed.
Last evaluation:
Fitness: 7.628
Params: [59]

Best:
Optimal fitness: 7.83
Optimal params: [24] 

 ### HillClimbing Arterial: SOTL, 50 iters, 5 runs, strategy 1, Delta = 2/16
151 min and 20.474988 seconds needed.
Last evaluation:
Fitness: 7.136
Params: [74]

Best:
Optimal fitness: 7.36
Optimal params: [24]


  ### HillClimbing Arterial: CB, 50 iters, 5 runs, strategy 1, Delta = 0 (mayby do a rerun)
381 min and 10.183828 seconds needed.
Last evaluation:
Fitness: 7.5600000000000005
Params: [  1.5 164.  147.  187.  170.  237. ]

Best:
Optimal fitness: 8.0
Optimal params: [0.8080663378899781, 64, 47, 87, 70, 137]

  ### HillClimbing Arterial: CB, 50 iters, 5 runs, strategy 1, Delta = 1/16
455 min and 59.874841 seconds needed.
Last evaluation:
Fitness: 7.4879999999999995
Params: [  1.5 164.  147.  187.  170.  237. ]

Best:
Optimal fitness: 7.97
Optimal params: [0.8080663378899781, 64, 47, 87, 70, 137]

  ### HillClimbing Arterial: CB, 50 iters, 5 runs, strategy 1, Delta = 2/16
428 min and 40.581102 seconds needed.
Last evaluation:
Fitness: 7.544
Params: [  1.5 164.  147.  187.  170.  237. ]

Best:
Optimal fitness: 7.95
Optimal params: [0.8080663378899781, 64, 47, 87, 70, 137]

### HillClimbing arterial meanWaitingTime: AdaSOTL, 50 iters, 5 runs, strategy 1, delta_r_t=2/16
255 min and 5.082778 seconds needed.
Last evaluation:
Fitness: 8.136
Params: [3.58710892 1.49361679]

Best:
Optimal fitness: 8.196000000000002
Optimal params: [3.48710892 1.51361679]

### HillClimbing 2x3 fixed: AdaSOTL, 50 iters, 5 runs, strategy 1, 900 veh
274 min and 52.418194 seconds needed.
Last evaluation:
Fitness: 8.315999999999999
Params: [3.28710892 1.57361679]

Best:
Optimal fitness: 8.348
Optimal params: [3.38710892 1.55361679]

### HillClimbing 2x3 fixed: AdaSOTL, 50 iters, 5 runs, strategy 1, 1200 veh
279 min and 57.165212 seconds needed.
Last evaluation:
Fitness: 8.129999999999999
Params: [3.28710892 1.61361679]

Best:
Optimal fitness: 8.144
Optimal params: [3.48710892 1.61361679]

### HillClimbing 2x3 fixed: AdaSOTL, 50 iters, 5 runs, strategy 1, 1500 veh
292 min and 59.228068 seconds needed.
Last evaluation:
Fitness: 7.814
Params: [3.28710892 1.59361679]

Best:
Optimal fitness: 7.874
Optimal params: [3.08710892 1.59361679]

### HillClimbing 2x3Grid_fixed: CB, 50 iters, 5 runs, strategy 1, 900 veh
404 min and 2.141102 seconds needed.
Last evaluation:
Fitness: 8.608
Params: [0.90806634  78.          39.          99.54.107.        ]

Best:
Optimal fitness: 8.618
Optimal params: [0.85806634 ,80,49,99,52,107]

### HillClimbing 2x3Grid_fixed: CB, 50 iters, 5 runs, strategy 1, 1200 veh
429 min and 51.646476 seconds needed.
Last evaluation:
Fitness: 8.294
Params: [0.90806634  34.   51.  99.    58. 153. ]

Best:
Optimal fitness: 8.384
Optimal params: [0.85806634  36.    53.  101.   56. 155.]


### HillClimbing 2x3Grid_fixed: CB, 50 iters, 5 runs, strategy 1, 1500 veh
779 min and 5.406524 seconds needed.
Last evaluation:
Fitness: 7.822
Params: [  0.75  60.    23.    51.    78.   137.  ]

Best:
Optimal fitness: 7.952
Optimal params: [  0.75  74.    25.    53.    84.   135.  ]

### HillClimbing 2x3Grid_fixed: SOTL, 50 iters, 5 runs, strategy 1, 900 vehicles
96 min and 8.348288 seconds needed.
Last evaluation:
Fitness: 6.474000000000001
Params: [25.]

Best:
Optimal fitness: 6.824
Optimal params: [28.]

### HillClimbing 2x3Grid_fixed: SOTL, 50 iters, 5 runs, strategy 1, 1200 vehicles
134 min and 51.604621 seconds needed.
Last evaluation:
Fitness: 5.09
Params: [25.]

Best:
Optimal fitness: 5.180000000000001
Optimal params: [30.]

### HillClimbing 2x3Grid_fixed: SOTL, 50 iters, 5 runs, strategy 1, 1500 vehicles
146 min and 30.462012 seconds needed.
Last evaluation:
Fitness: 4.92
Params: [37.]

Best:
Optimal fitness: 4.965999999999999
Optimal params: [34.]