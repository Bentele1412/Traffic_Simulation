# Traffic Simulation Project
This is the README.md of the repository "Traffic_Simulation".

## File generation
To generate flow.xml files, run e.g.

```
cd "C:\Program Files (x86)\Sumo\tools"
randomTrips.py -n circularControlNet.net.xml -o "C:\Users\Marcus\Desktop\Hauptordner\Studium\Masterstudium\3. Semester\Projekt_Sim-Opt\Traffic_Simulation\ZUS_Test\circularControlFlow.xml" --begin 0 --end 36000 --random --binomial 1 -p 6
```

To generate the corresponding routes, run e.g.

```
cd "C:\Users\Marcus\Desktop\Hauptordner\Studium\Masterstudium\3. Semester\Projekt_Sim-Opt\Traffic_Simulation\ZUS_Test"
jtrrouter -c circularControl.jtrrcfg
```
