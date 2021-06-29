import time
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.pyplot as ptr
import enum
from scipy.integrate import odeint
from scipy.stats import pearsonr
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import networkx as nx

from multiprocessing import Pool

steps_per_day = 30

class InfectionMultiscaleModel(Model):
    """A model for infection spread."""

    def __init__(self, N=10, steps_per_day = 2,
                 v_threshold=300000, G='constant',
                 con_init=100, lin_init=1, log_init=1,
                 sig_init=1, delay=0, output_tree=None):

        self.num_agents = N
        self.steps_per_day = steps_per_day
        self.con_init = con_init
        self.lin_init = lin_init
        self.log_init = log_init
        self.sig_init = sig_init
        self.delay = delay
        self.schedule = RandomActivation(self)
        self.running = True
        self.dead_agents = []
        self.G = G
        self.r0 = 0
        # self.init_vs = []
        # self.avg_init_v = 0
        self.ptrans = 1.0 / steps_per_day
        self.viral_load_max = 25000
        self.inoc_max = 50000
        self.inoc_max_sig = np.power(self.inoc_max/2,self.sig_init)
        self.infected = 0
        # self.viral_load_corr = dict()
        self.viral_load_tree = dict()

        if G == 'constant':
            self.v_threshold = v_threshold
        elif G == 'random':
            self.v_threshold = v_threshold
        elif G == 'linear':
            # self.v_threshold = max(v_threshold, (1 / self.lin_init))
            self.v_threshold = 1 / self.lin_init
        elif G == 'log':
            # self.v_threshold = max(v_threshold, np.exp(1 / (self.log_init * self.inoc_max)))
            self.v_threshold = np.exp(1 / (self.log_init * self.inoc_max))
        elif G == 'sigmoid':
            beta = self.lin_init * self.inoc_max
            # self.v_threshold = max(v_threshold, np.power(self.inoc_max_sig/(beta * (1 - (1/beta))),1/self.sig_init) - self.delay)
            self.v_threshold = np.power(self.inoc_max_sig/(beta * (1 - (1/beta))),1/self.sig_init) - self.delay
        
        print(self.lin_init,self.sig_init,self.v_threshold)

        # Create agents
        for i in range(self.num_agents):
            a = MSMyAgent(i, self)
            self.schedule.add(a)
            if i == 1:
                a.state = MSState.INFECTED
                self.infected = 1
                a.infection_course = a.infect_stein('constant',a.unique_id,self.con_init,1)

        self.datacollector = DataCollector(
            # model_reporters={"r0" : "r0",
            #                 "init_v" : "avg_init_v",
            #                 "init_v_raw" : "init_vs",
            #                 "viral_load_corr" : "viral_load_corr",
            #                 "viral_load_tree" : "viral_load_tree"},
            model_reporters={"r0" : "r0",
                            "viral_load_tree" : "viral_load_tree"},
            agent_reporters={"State": "state"})
        
    # def get_init_v(self):
    #     if self.init_vs:
    #         return np.mean(self.init_vs)
    #     else:
    #         return 0

    def step(self):
        self.datacollector.collect(self)
        # self.avg_init_v = self.get_init_v()
        # self.init_vs = []
        # self.viral_load_corr = dict()
        self.viral_load_tree = dict()
        self.schedule.step()

        

class MSState(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

def differential_stein(n_stein, t, r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP):
    dV_dt = r*n_stein[0] - n_stein[0]*(((r*n_stein[0])/kV) + kI*n_stein[1] + kN*n_stein[2] + kP*n_stein[3])
    dI_dt = aI*n_stein[0] + bI*(1-(n_stein[1]/KI))
    dN_dt = aN*n_stein[0]*np.heaviside(t-tN,1)-dN*n_stein[2]
    dP_dt = aP*n_stein[0]*n_stein[3] + c*n_stein[2]*(t-tP)
    return dV_dt, dI_dt, dN_dt, dP_dt
    
class MSMyAgent(Agent):
    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)      
        self.state = MSState.SUSCEPTIBLE  
        self.infection_time = 0
        self.infection_course = []
        self.init_v = 0
    
    def infect_stein(self, G, donor_id, donor, donor_init_v):
        t = np.linspace(0, 24, num=24 * self.model.steps_per_day)
        y0 = ()
        r = 5.5
        kI = 0.05
        kN = 0.5
        kP = 2
        aI = 0.000000001
        aN = 0.00000001
        aP = 0.000005
        bI = 2
        dN = 0.05
        c = 0.01
        kV = 100000000000
        KI = 100
        tN = 2.5
        tP = 3
        n0 = 0
        p0 = 2
        
        if G == 'constant':
            init_v = self.model.con_init
        elif G == 'random':
            init_v = self.random.randint(1,3000000)
        elif G == 'bottleneck':
            init_v = np.minimum(1, donor * 0.000001)
        elif G == 'linear':
            init_v = self.model.lin_init * donor
        elif G == 'log':
            init_v = self.model.log_init * np.log(donor)
        elif G == 'sigmoid':
            init_v = self.model.lin_init * self.model.inoc_max * np.power(donor + self.model.delay,self.model.sig_init) / (np.power(donor + self.model.delay,self.model.sig_init) + self.model.inoc_max_sig)

        # self.model.init_vs.append(init_v)
        # self.model.infected += 1
        # self.model.viral_load_corr[donor_init_v] = init_v
        if self.unique_id != 1:
            self.model.viral_load_tree[(donor_id,self.unique_id)] = (donor_init_v,init_v,donor)

        self.init_v = init_v
            
        y0 = (init_v, 0, n0, p0)
        
        solution = odeint(differential_stein, y0, t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
        solution = [[row[i] for row in solution] for i in range(4)]

        return solution

    def status(self):
        """Check infection status"""

        if self.state == MSState.INFECTED: 
            t = self.model.schedule.time-self.infection_time
            if self.infection_course[0][t] < 1:
                self.state = MSState.REMOVED
                # self.model.infected -= 1
                if self.unique_id == 1:
                    self.model.infected = 0


    def contact(self):
        """Find close contacts and infect"""
        
        if self.state is MSState.INFECTED and self.unique_id == 1:
            cellmates = self.random.sample(self.model.schedule.agents,k=1)
            t = self.model.schedule.time-self.infection_time
            if t>0:
                var_ptrans = self.infection_course[0][t] / (self.model.viral_load_max * self.model.steps_per_day)
                for other in cellmates:
                    if other.state is MSState.SUSCEPTIBLE and self.infection_course[0][t] > self.model.v_threshold and self.random.random() < var_ptrans: 
                    # if other.state is MSState.SUSCEPTIBLE and self.infection_course[0][t] > model.v_threshold and self.random.random() < self.model.ptrans:                    
                        other.state = MSState.INFECTED
                        other.infection_time = self.model.schedule.time
                        other.infection_course = other.infect_stein(model.G,self.unique_id,self.infection_course[0][t],self.init_v)
                        if self.unique_id == 1:
                            self.model.r0 += 1
                    
    def step(self):
        self.status()
        self.contact()

def get_column_data(model):
    """pivot the model dataframe to get states count at each step"""
    agent_state = model.datacollector.get_agent_vars_dataframe()
    model_state = model.datacollector.get_model_vars_dataframe()

    X = pd.DataFrame()    
    X['r0'] = model_state['r0']
    # X['init_v'] = model_state['init_v']
    # X['init_v_raw'] = model_state['init_v_raw']
    # X['viral_load_corr'] = model_state['viral_load_corr']
    X['viral_load_tree'] = model_state['viral_load_tree']

    labels = ['R0','Viral Load Tree']
    X.columns = labels[:len(X.columns)]
    # labels = ['Susceptible','Infected','Removed','R0','Viral Load Tree']
    # X.columns = labels[:len(X.columns)]
    # X['Incidence'] = X['Susceptible'].diff() * -1
    # X['Recovery'] = X['Removed'].diff()


    # for j in range(X.shape[0],steps):
    #     X.loc[j] = 0
    #     X['Viral Load Tree'].loc[j] = []
    #     # X['Viral Load Correlation'].loc[j] = 0
    # X['Days'] = X.index
    # X['Days'] = X['Days'].div(steps_per_day)


    # X['Incidence Sum'] = X['Incidence']
    # X['Incidence Days'] = 0

    # for i in range(0,days):
    #     # print(i*steps,(i+1)*steps)
    #     X['Incidence Sum'].loc[i] = X['Incidence'][i*steps_per_day:(i+1)*steps_per_day].sum()
    #     # print(X['Incidence'][i*steps:(i+1)*steps])
    #     X['Incidence Days'].loc[i] = i

    # X['Beta'] = 0
    # X['Beta'].iloc[1:] = X['Incidence'].iloc[1:].div(X['Infected'].shift(periods=1))
    # X['Gamma'] = 0
    # X['Gamma'].iloc[1:] = X['Recovery'].iloc[1:].div(X['Infected'].shift(periods=1))
    # X['R0_new'] = pop * X['Beta']
    # X['R0_new'] = X['R0_new'].div(X['Gamma'])
    return X

def get_peak_viral_load(init):
    t = np.linspace(0, 12, num=12 * steps_per_day)
    y0 = ()
    r = 8
    kI = 0.05
    kN = 0.5
    kP = 2
    aI = 0.000000001
    aN = 0.00000001
    aP = 0.000005
    bI = 2
    dN = 0.05
    c = 0.01
    kV = 100000000000
    KI = 100
    tN = 2.5
    tP = 3
    n0 = 0
    p0 = 2

    y0 = (init, 0, n0, p0)
        
    solution = odeint(differential_stein, y0, t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
    solution = [[row[i] for row in solution] for i in range(4)]

    return max(solution[0])

days = 100
steps = steps_per_day * days
pop = 5000

ptr.close("all")

# HISTOGRAM GENERATOR
z = np.linspace(1,1401,15)
no_sims = 20

ptr.figure(figsize=(15,10), dpi=100)

length = 6
width = 5

size = length * width

# v_max = 500
# j_add = v_max / size
# i_add = width * j_add

for j in range(len(z)):

    data_whole = []
    r0s_inner = []
    for i in range(no_sims):
        print("Sim: " + str(i))
        model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='linear', con_init=z[j], sig_init=10, delay=0, lin_init=0.01, log_init=400)
        for k in range(steps):
            model.step()
            if model.infected == 0:
                model.step()
                break

        data = get_column_data(model)

        new_inits = data['Viral Load Tree']
        new_inits = new_inits[new_inits.astype(bool)]

        r0s_inner.append(data['R0'].iloc[-1])
        if i==0:
            data_whole = new_inits
        else:
            data_whole = data_whole.append(new_inits)
    
    # print(r0s_inner,data_whole)

    vals = [list(k.values())[0][2] for k in data_whole]

    print(r0s_inner, vals)

    row = 2 * int(j / 5)
    col = int(j % 5)

    # print(hor,ver)

    ax = ptr.subplot2grid((6,5), (row,col), title='V0 = ' + str(z[j]), xlabel='Number of Secondary Cases', ylim=(0,10))
    ax.hist(r0s_inner,bins=15,range=(0,15))
    ax = ptr.subplot2grid((6,5), (row+1,col), title='V0 = ' + str(z[j]), xlabel='Donor Viral Loads at transmission', ylim=(0,25))
    ax.hist(vals,bins=10,range=(0,50000))

ptr.tight_layout()

ptr.savefig(fname='hist')
ptr.show()