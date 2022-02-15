#!/usr/bin/env python

class PBSS():
    class Cluster():
        def __init__(self, tau_ps, n_pc, v_f, L_det=450, tau_pd=1):
            '''
            status: 
                0 = Tuple
                1 = Minor
                2 = Queue
                3 = Platoon
            '''
            self.L_det = L_det #length between intersection and detector
            self.v_f = v_f #speed
            self.tau_ps = tau_ps #sample period start time
            self.tau_pd = tau_pd #sample period duration
            self.tau_ps_t = self.getTau_ps_t(self.tau_ps)
            self.calcTau_pe() #calc sample period end time
            self.n_pc = n_pc #number of counted vehicles (drove over detector)
            self.calcQ_pc() #calc flow rate of vehicles
            self.status = 0 #tuple
        
        def getTau_ps_t(self, t):
            return self.tau_ps - t + self.L_det/self.v_f

        def calcTau_pe(self):
            self.tau_pe = self.tau_ps_t + self.tau_pd

        def calcQ_pc(self):
            self.q_pc = self.n_pc/self.tau_pd #flow rate of vehicles

    def __init__(self, tl, tau_g_min=5, tau_g_max=55, tau_y=5, tau_sl=3, tau_sh=3, v_f=9.5, c_thc=5, c_thpc=5, useAAC=True, usePBE=True, usePBS=True):
        self.tl = tl
        self.tau_g_min = tau_g_min #min green time
        self.tau_g_max = tau_g_max #max green time
        self.tau_y = tau_y #yellow time
        self.tau_sl = tau_sl #startup loss time
        self.tau_sh = tau_sh #saturation headway
        self.v_f = v_f #0.95*v_max --> here v_max = 10 m/s
        self.c_thc = 0 if useAAC and not usePBE and not usePBS else c_thc #specified threshold duration for creating clusters
        self.c_thpc = c_thpc #threshold of platoon count (in cars)
        self.c_thpd = 1/self.c_thc if self.c_thc != 0 else 0 #flow rate threshold
        self.useAAC = useAAC 
        self.usePBE = usePBE
        self.usePBS = usePBS

        self.S = []
        for _ in range(len(self.tl.lanes)):
            self.S.append([])
        self.tau_ge = 0
        self.tau_ext = 0

    def step(self, t):
        yellowPhase = True if self.tl.getCurrentPhase() % 2 != 0 else False
        self.tau_ge = self.tau_ge + 1 if not yellowPhase else self.tau_ge #only increase green phase duration count if tl is in a green phase
        self.tau_ext = self.tau_ext - 1 if not yellowPhase else self.tau_ext

        #perform incoming detection
        for counter, lane in enumerate(self.tl.lanes):
            incomingCars = lane.detectors[0].getCurrentCars()
            if incomingCars > 0:
                self.S[counter].append(self.Cluster(t, incomingCars, self.v_f))
            
            #perform outgoing detection
            outflowingCars = lane.detectors[1].getCurrentCars()
            while outflowingCars > 0:
                if len(self.S[counter]) > 0:
                    if self.S[counter][0].n_pc - outflowingCars <= 0:
                        outflowingCars -= self.S[counter][0].n_pc 
                        del self.S[counter][0]
                    else:
                        self.S[counter][0].n_pc -= outflowingCars
                        break
                else:
                    break
        
        #update tau_ps_t for all clusters
        for s in self.S:
            for c in s:
                c.tau_ps_t = c.getTau_ps_t(t)
        
        #reaching decision point
        if not yellowPhase and self.tau_ext <= 0:
            #aggregation
            for counter, s in enumerate(self.S):
                self.S[counter] = self.aggregateClusters(s)

            #policy application
            redLaneInd = 0 if self.tl.lanes[0].isRed else 1
            greenLaneInd = 1 if redLaneInd == 0 else 0
            if self.useAAC:
                self.tau_ext = self.performAAC(self.tau_ge, self.S[greenLaneInd])
            if self.usePBE and self.tau_ext <= 0:
                self.tau_ext = self.performPBE(self.tau_ge, self.S[redLaneInd], self.S[greenLaneInd])
            if self.usePBS and self.tau_ext <= 0:
                self.tau_ext = self.performPBS(self.tau_ge, self.S[redLaneInd])

            if self.tau_ext == 0 and self.tau_ge >= self.tau_g_min:
                self.tl.switchLight(self.tl.getCurrentPhase())
                self.tau_ge = 0

    def performAAC(self, tau_ge, s):
        tau_gremain = self.tau_g_max - tau_ge
        n_q = self.calcN_q(s)
        n_qa = self.estimateN_qa(s, tau_ge, tau_adv=0, n_qn=n_q)
        tau_ext = self.getTau_qc(n_qa, tau_ge)
        tau_ext = min(tau_ext, tau_gremain)
        return tau_ext

    def performPBE(self, tau_ge, s_r, s_g):
        tau_gremain = self.tau_g_max - tau_ge
        n_q = self.calcN_q(s_r)
        n_qa_r = self.estimateN_qa(s_r, tau_ge=0, tau_adv=self.tau_y, n_qn=n_q)
        tau_qa_r = self.getTau_qc(n_qa_r, 0)
        tau_qa_r = max(tau_qa_r, tau_gremain)
        platoons_g = [c for c in s_g if c.status == 3]
        if not len(platoons_g) == 0:
            n_m_g = sum([c.n_pc for c in s_g if c.status == 1 and c.tau_ps_t < platoons_g[0].tau_ps_t])
            tau_idle_g = platoons_g[0].tau_ps_t - self.getTau_qc(n_m_g, 0)
            delta_tau = (tau_qa_r + 2 * self.tau_y) - tau_idle_g
            if delta_tau > 0:
                n_m_r = sum([c.n_pc for c in s_r if c.status == 1])
                n_m_r = n_m_r + n_q - n_qa_r #questionable usage of n_q
                delta_r = n_m_r * delta_tau - n_qa_r * (platoons_g[0].tau_pe + n_m_r * self.tau_sl)
                delta_g = (delta_tau + self.tau_sl) * (platoons_g[0].n_pc + n_m_g) + n_m_g * tau_idle_g/2
                if delta_g - delta_r > 0: #check +
                    return platoons_g[0].tau_pe
        return 0

    def performPBS(self, tau_ge, s_r):
        platoons_r = [c for c in s_r if c.status == 3]
        if len(platoons_r) > 0:
            tau_gremain = self.tau_g_max - tau_ge
            n_q = self.calcN_q(s_r)
            n_m_r = sum([c.n_pc for c in s_r if c.status == 1 and c.tau_ps_t < platoons_r[0].tau_ps_t])
            tau_qa_r = self.getTau_qc(n_q + n_m_r, 0)
            tau_qa_r = max(tau_qa_r, tau_gremain)
            tau_idle_r = platoons_r[0].tau_ps_t - tau_qa_r - self.tau_y
            if tau_idle_r < (self.tau_g_min + 2 * self.tau_y):
                return tau_idle_r
        return 0

        
    def aggregateClusters(self, s):
        currentClusterInd = 0
        while(currentClusterInd < len(s)-1):
            currentCluster = s[currentClusterInd]
            nextCluster = s[currentClusterInd+1]
            currentTau_pe = currentCluster.tau_pe
            if nextCluster.tau_ps_t - currentTau_pe < self.c_thc:
                #merge tuples
                currentCluster.tau_pe = nextCluster.tau_pe
                currentCluster.tau_pd = currentCluster.tau_pe - currentCluster.tau_ps_t #changed from sum to this
                currentCluster.n_pc += nextCluster.n_pc
                currentCluster.calcQ_pc()
                del s[currentClusterInd+1]
            else:
                currentClusterInd += 1

        for cluster in s:
            if cluster.tau_ps_t <= 0:
                cluster.status = 2
            elif cluster.q_pc > self.c_thpd and cluster.n_pc > self.c_thpc:
                cluster.status = 3
            else:
                cluster.status = 1
        return s

    def estimateN_qa(self, s, tau_ge, tau_adv, n_qn):
        n_qa = n_qn
        tau_qc = self.getTau_qc(n_qa, tau_ge)
        for c in s:
            if c.tau_ps_t - tau_adv <= tau_qc:
                delta_d = 1/self.tau_sh - c.q_pc
                if delta_d <= 0 or c.tau_pe - tau_adv <= tau_qc:
                    n_qa += c.n_pc
                else:
                    delta_t = (tau_qc - (c.tau_ps_t - tau_adv)) * c.q_pc/delta_d
                    if delta_t < c.tau_pd:
                        n_qa += c.n_pc * delta_t/c.tau_pd
                        break
                    else:
                        n_qa += c.n_pc
        return n_qa

    def calcN_q(self, s):
        return sum([c.n_pc for c in s if c.tau_ps_t <= 0])


    def getTau_qc(self, n_q, tau_ge):
        #calculate queue clearing time
        return self.tau_sl - tau_ge + self.tau_sh * n_q if tau_ge < self.tau_sl else self.tau_sh * n_q