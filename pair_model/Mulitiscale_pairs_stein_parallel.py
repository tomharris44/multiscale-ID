import time
import os
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.pyplot as ptr
import enum
from scipy.integrate import odeint
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from multiprocessing import Pool
from multiprocessing.sharedctypes import Value, Array

steps_per_day = 5

class InfectionMultiscaleModel(Model):
    """A model for infection spread."""

    def __init__(self, N=10, steps_per_day = 2,
                 v_threshold=300000, G='constant',
                 con_init=1, lin_init=1):

        self.num_agents = N
        self.steps_per_day = steps_per_day
        self.v_threshold = v_threshold
        self.con_init = con_init
        self.lin_init = lin_init
        self.schedule = RandomActivation(self)
        self.running = True
        self.dead_agents = []
        self.G = G
        self.r0 = 0
        self.init_vs = []
        self.avg_init_v = 0
        self.ptrans = 1.0 / steps_per_day
        self.viral_load_max = 125000
        self.infected = 0
        
        # Create agents
        for i in range(self.num_agents):
            a = MSMyAgent(i, self)
            self.schedule.add(a)
            if i == 1:
                a.state = MSState.INFECTED
                a.infection_course = a.infect_stein('constant',1)

        self.datacollector = DataCollector(
            model_reporters={"r0" : "r0",
                            "init_v" : "avg_init_v"},
            agent_reporters={"State": "state"})
        
    def get_init_v(self):
        if self.init_vs:
            return np.mean(self.init_vs)
        else:
            return 0

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.avg_init_v = self.get_init_v()
        self.init_vs = []
        

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
    
    def infect_stein(self, G, donor):
        t = np.linspace(0, 12, num=12 * self.model.steps_per_day)
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
        
        if G == 'constant':
            init_v = self.model.con_init
        elif G == 'random':
            init_v = self.random.randint(1,3000000)
        elif G == 'bottleneck':
            init_v = np.minimum(1, donor * 0.000001)
        elif G == 'linear':
            init_v = np.maximum(self.model.lin_init * donor,1)
        elif G == 'log':
            init_v = np.log(donor)
        elif G == 'sigmoid':
            init_v = 3700000 * np.power(donor,10) / (np.power(donor,10) + 1850000)

        self.model.init_vs.append(init_v)
        self.model.infected += 1
            
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
                self.model.infected -= 1


    def contact(self):
        """Find close contacts and infect"""
        
        if self.state is MSState.INFECTED:
            cellmates = self.random.sample(self.model.schedule.agents,k=1)
            t = self.model.schedule.time-self.infection_time
            if t>0:
                var_ptrans = self.infection_course[0][t] / (self.model.viral_load_max * self.model.steps_per_day)
                for other in cellmates:
                    if other.state is MSState.SUSCEPTIBLE and self.infection_course[0][t] > self.model.v_threshold and self.random.random() < var_ptrans: 
                    # if other.state is MSState.SUSCEPTIBLE and self.infection_course[0][t] > self.model.v_threshold and self.random.random() < self.model.ptrans:                    
                        other.state = MSState.INFECTED
                        other.infection_time = self.model.schedule.time
                        other.infection_course = other.infect_stein(self.model.G,self.infection_course[0][t])
                        if self.unique_id == 1:
                            self.model.r0 += 1
                    
    def step(self):
        self.status()
        self.contact()



steps = steps_per_day * 100
pop = 2500

ptr.close("all")

data_whole = pd.DataFrame()

def get_column_data(model):
    """pivot the model dataframe to get states count at each step"""
    agent_state = model.datacollector.get_agent_vars_dataframe()
    model_state = model.datacollector.get_model_vars_dataframe()

    X = pd.pivot_table(agent_state.reset_index(),index='Step',columns='State',aggfunc=np.size,fill_value=0)    
    X['r0'] = model_state['r0']
    X['init_v'] = model_state['init_v']

    labels = ['Susceptible','Infected','Removed','R0','Mean Initial Viral Load']
    X.columns = labels[:len(X.columns)]
    X['Incidence'] = X['Susceptible'].diff() * -1
    X['Recovery'] = X['Removed'].diff()

    for j in range(X.shape[0],steps):
        X.loc[j] = 0
    X['Days'] = X.index
    X['Days'] = X['Days'].div(steps_per_day)

    # X['Beta'] = 0
    # X['Beta'].iloc[1:] = X['Incidence'].iloc[1:].div(X['Infected'].shift(periods=1))
    # X['Gamma'] = 0
    # X['Gamma'].iloc[1:] = X['Recovery'].iloc[1:].div(X['Infected'].shift(periods=1))
    # X['R0_new'] = pop * X['Beta']
    # X['R0_new'] = X['R0_new'].div(X['Gamma'])
    return X

#CONSTANT INITIAL VIRAL LOAD

z = [1,10,100,1000,10000]
no_sims = 20

r0s = Array('f',[0 for i in range(len(z))])

def constant_sim(j):
    ptr.figure(figsize=(10,10), dpi=100)

    axes_s = ptr.subplot(211)
    axes_s.set_ylabel("#individuals")

    axes_i = ptr.subplot(212)
    axes_i.set_ylabel("Incidence")
    axes_i.set_xlabel("Days")

    data_whole = pd.DataFrame()
    r0s_inner = []
    for i in range(no_sims):
        print("Sim: " + str(i))
        #Validation setting (allow dud runs)
        # model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='constant',con_init=j)
        # for i in range(steps):
        #     model.step()
        #     if model.infected == 0:
        #         model.step()
        #         break
        while(True):
            model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='constant',con_init=j)
            for i in range(steps):
                model.step()
                if model.infected == 0:
                    model.step()
                    break
            if model.r0 != 0:
                break

        data = get_column_data(model)
        r0s_inner.append(data["R0"].max())
        if i==1:
            data_whole = data
        else:
            data_whole = data_whole.append(data)
    # data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
    data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='V0=' + str(j), sharex=True)
    for i in range(5):
        data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': V0=' + str(j), sharex=True)
    data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='V0=' + str(j), sharex=True, logy=True, ylim=[1,1000])
    for i in range(5):
        data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': V0=' + str(j), sharex=True)
    # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
    print(data_whole.groupby(level=0).mean().max())
    print(r0s_inner)
    r0s[z.index(j)] = np.mean(r0s_inner)
    ptr.savefig(fname='pair_constant_var_ptrans_stein_' + str(no_sims) + '_' + str(steps_per_day) + '_' + str(j))
    # ptr.show()


with Pool(5) as p:
    p.map(constant_sim,z)
print([a for a in r0s])


# LINEAR VIRAL LOAD

# z = [0.0001, 0.001, 0.01, 0.1, 1.0]
# z_names = ['00001','0001','001','01','1']
# no_sims = 1

# r0s = Array('f',[0 for i in range(len(z))])

# def linear_sim(j):
#     ptr.figure(figsize=(10,10), dpi=100)

#     # make a subplot for the susceptible, infected and recovered individuals
#     axes_s = ptr.subplot(311)
#     axes_s.set_ylabel("#individuals")

#     axes_i = ptr.subplot(312)
#     axes_i.set_ylabel("Incidence")
#     axes_i.set_xlabel("Days")

#     axes_r = ptr.subplot(313)
#     axes_r.set_ylabel("Mean Initial Viral Load")
#     axes_r.set_xlabel("Days")
#     data_whole = pd.DataFrame()
#     r0s_inner = []

#     for i in range(no_sims):
#         print("Sim: " + str(i))
#         while(True):
#             model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='linear',lin_init=j)
#             for i in range(steps):
#                 model.step()
#                 if model.infected == 0:
#                     model.step()
#                     break
#             if model.r0 != 0:
#                 break

#         data = get_column_data(model)
#         r0s_inner.append(data["R0"].max())
#         if i==1:
#             data_whole = data
#         else:
#             data_whole = data_whole.append(data)

#     for i in range(no_sims):
#         data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='_nolegend_', color='lightcoral', sharex=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='alpha=' + str(j), sharex=True)

#     for i in range(no_sims):
#         data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='_nolegend_', color='lightcoral', sharex=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), sharex=True, logy=True)

#     for i in range(no_sims):
#         data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_r, title='Mean initial viral load (N=2500)', y="Mean Initial Viral Load", label='_nolegend_', color='lightcoral', logy=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_r, title='Mean initial viral load (N=2500)', y="Mean Initial Viral Load", label='alpha=' + str(j), logy=True)

#     print(data_whole.groupby(level=0).mean().max())
#     print(r0s_inner)
#     r0s[z.index(j)] = np.mean(r0s_inner)
#     k = z_names[z.index(j)]
#     ptr.savefig(fname='pair_linear_high_threshold_stein_' + str(no_sims) + '_' + str(steps_per_day) + '_' + k)


# with Pool(5) as p:
#     p.map(linear_sim,z)
# print([a for a in r0s])