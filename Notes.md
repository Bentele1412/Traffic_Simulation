# Ideas for autonomous controller
* SOTL + self-adapting threshold:
  * calculate traffic density at runtime to determine threshold value and green phase times 
  * $\theta$ = k * trafficDensity --> "optimize" k to control the impact of $\theta$-change


## Open Questions
* definition of traffic density
* number of traffic density calculations


## Keep in mind
* Identify stable system state and only measure if system in stable state


## Questions to Vollbrecht
* Where to depart/arrive? Middle of edges - junctions - only outer junctions? 

## To-do
* Read paper until tuesday
* reimplement 2X3 grid from paper with flows
* implement SOTL with well defined object structure
* implement fixed cycle based control
* compare SOTL and FIX + comparison to paper results