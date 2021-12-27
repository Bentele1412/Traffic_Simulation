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
* Read paper until tuesday
* reimplement 2X3 grid from paper with flows
* implement SOTL with well defined object structure
* implement fixed cycle based control
* compare SOTL and FIX + comparison to paper results
* Hillclimbing: 
  * define gradient per dimension with (positive direction fitness deviation - negative direction fitness deviation)
  * ensure minCycleTime is never undercut
  * Optimization strategies:
    * calc all directions, take best and multiply with gradient --> new fitness evaluation needed after updates
    * calc all directions, get all fitness increasing gradients and summarize as one gradient --> updates are performed in several directions at once
    * iterate over directions and make the step and update for one direction immediately, if a better fitness value is achieved --> fitnessDynamics???
* Consider phase switches at switching routes (t=1200/2400) --> maybe green light gets red immediately, but that should not be the case
* Parallelization --> file writings are issues to face

* Szenarios: TwoCross??, 2x3Grid, FH bridge
* Traffic load: 900, 1200, 1500 (?)
* Controller: Cycle based, SOTL, PBSS
* Optimizers: HillClimbing, GridSearch, ES (?)

Optimal fitness: 10.204
Optimal params: [ 0.6688 10.099  20.023  10.215  19.992  29.667 ]