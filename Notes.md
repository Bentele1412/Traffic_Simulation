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
* Consider phase switches at switching routes (t=1200/2400) --> maybe green light gets red immediately, but that should not be the case