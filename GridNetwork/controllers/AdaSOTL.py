#!/usr/bin/env python

class AdaSOTL():
    '''
    Adaptive threshold self-organizing traffic light controller

    Problem of basic SOTL:
        - Green phases get shorter if traffic load gets heavier but should get longer
        - leads to more yellow phases and crossing clearing --> less time for queue clearing --> lost time for every crossway participant to move on

    Idea:
        - adaptive threshold of integrated cars on lane
        - the more cars driving to a red light of the crossway per lane, the bigger the threshold gets (biggest threshold for all (red) lanes at the crossway)

    Implementation strategy:
        - sum up all cars driving towards crossway and multiply with a constant (constant to be optimized)
        --> threshold =  avgCarsTowardsCrossway**self.beta * self.alpha
        - alpha 
            - constant to adapt threshold to needed scale range
        - beta 
            - exponent to ensure increasing threshold of running averaged cars towards crossway
            - otherwise kappa overshooting theta would need always the same time (linear dependency of carsTowardsCrossway and alpha)

    Open Questions: 
        - Adaptation strength
    '''
    def __init__(self, tl, mu, alpha, beta):
            self.tl = tl
            self.mu = mu
            self.theta = 0
            self.alpha = alpha
            self.beta = beta

            self.kappa = 0
            self.phi_min = self.tl.minGreenTime
            self.phi = 0

    def step(self):
        avgCarsTowardsCrossway = 0
        currentPhase = self.tl.getCurrentPhase()
        if currentPhase % 2 == 0: #don´t execute SOTL if TL in yellow phase
            self.phi += 1
        for lane in self.tl.lanes:
            lane.updateCarCount()
            if lane.isRed: 
                self.kappa += lane.carsOnLane #change to more kappas if more than one direction has red
            
            #adaptivity
            avgCarsTowardsCrossway += lane.runningAvgCoL
        self.theta = avgCarsTowardsCrossway**self.beta * self.alpha

        if self.phi >= self.phi_min and currentPhase % 2 == 0:
            for lane in self.tl.lanes:
                if not lane.isRed:
                    if not(0 < lane.carsWithinOmega and lane.carsWithinOmega < self.mu) or self.phi > self.tl.maxGreenTime:
                        if self.kappa >= self.theta:
                            self.tl.switchLight(self.tl.getCurrentPhase())
                            self.resetParams()
                            break
    
    def resetParams(self):
        self.phi = 0
        self.kappa = 0