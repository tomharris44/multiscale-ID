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
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

import networkx as nx

steps_per_day = 30

class InfectionMultiscaleModel(Model):
    """A model for infection spread."""

    def __init__(self, N=10, steps_per_day = 2,
                 v_threshold=300000, G='constant',
                 con_init=1000, lin_init=1, log_init=1,
                 sig_init=1, delay=0):

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
        self.init_vs = []
        self.avg_init_v = 0
        self.ptrans = 1.0 / steps_per_day
        self.viral_load_max = 25000
        self.inoc_max = 50000
        self.inoc_max_sig = np.power(self.inoc_max/2,self.sig_init)
        self.infected = 0
        self.viral_load_corr = dict()

        if G == 'constant':
            self.v_threshold = v_threshold
        elif G == 'random':
            self.v_threshold = v_threshold
        elif G == 'linear':
            self.v_threshold = max(v_threshold, (1 / self.lin_init))
        elif G == 'log':
            self.v_threshold = max(v_threshold, np.exp(1 / (self.log_init * self.inoc_max)))
        elif G == 'sigmoid':
            beta = self.lin_init * self.inoc_max
            self.v_threshold = max(v_threshold, np.power(self.inoc_max_sig/(beta * (1 - (1/beta))),1/self.sig_init) - self.delay)
        
        print(self.lin_init,self.sig_init,self.v_threshold)

        prob_link = 2 / N
        # self.Graph = nx.erdos_renyi_graph(n=self.num_agents, p=prob_link)
        # self.Graph = nx.complete_graph(n=self.num_agents)
        self.Graph = nx.random_regular_graph(d=3, n=self.num_agents)
        self.grid = NetworkGrid(self.Graph)

        # Create agents
        for i, node in enumerate(self.Graph.nodes()):
            a = MSMyAgent(i+1, self)
            self.schedule.add(a)
            if i == 1:
                a.state = MSState.INFECTED
                a.infection_course = a.infect_stein('constant',1, 1)
            self.grid.place_agent(a,node)

        self.datacollector = DataCollector(
            model_reporters={"r0" : "r0",
                            "init_v" : "avg_init_v",
                            "init_v_raw" : "init_vs",
                            "viral_load_corr" : "viral_load_corr"},
            agent_reporters={"State": "state"})
        
    def get_init_v(self):
        if self.init_vs:
            return np.mean(self.init_vs)
        else:
            return 0

    def step(self):
        self.datacollector.collect(self)
        self.avg_init_v = self.get_init_v()
        self.init_vs = []
        self.viral_load_corr = dict()
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
    
    def infect_stein(self, G, donor, donor_init_v):
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

        self.model.init_vs.append(init_v)
        self.model.infected += 1
        self.model.viral_load_corr[donor_init_v] = init_v

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
                self.model.infected -= 1


    def contact(self):
        """Find close contacts and infect"""

        if self.state is MSState.INFECTED:
            t = self.model.schedule.time-self.infection_time
            if t>0:
                neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
                susceptible_neighbors = [
                    agent
                    for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
                    if agent.state is MSState.SUSCEPTIBLE
                ]

                var_ptrans = self.infection_course[0][t] / (self.model.viral_load_max * self.model.steps_per_day)
                for other in susceptible_neighbors:
                    if self.infection_course[0][t] > self.model.v_threshold and self.random.random() < var_ptrans: 
                    # if self.infection_course[0][t] > model.v_threshold and self.random.random() < self.model.ptrans:
                        other.state = MSState.INFECTED
                        other.infection_time = self.model.schedule.time
                        other.infection_course = other.infect_stein(model.G,self.infection_course[0][t],self.init_v)
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
    X['init_v_raw'] = model_state['init_v_raw']
    X['viral_load_corr'] = model_state['viral_load_corr']

    labels = ['Susceptible','Infected','Removed','R0','Mean Initial Viral Load','Initial Viral Loads','Viral Load Correlation']
    X.columns = labels[:len(X.columns)]
    X['Incidence'] = X['Susceptible'].diff() * -1
    X['Recovery'] = X['Removed'].diff()

    for j in range(X.shape[0],steps):
        X.loc[j] = 0
        X['Initial Viral Loads'].loc[j] = []
        # X['Viral Load Correlation'].loc[j] = 0
    X['Days'] = X.index
    X['Days'] = X['Days'].div(steps_per_day)

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

#TIME STEP VALIDATION

# z = [1,5,10,15,20]
# no_sims = 3

# ptr.figure(figsize=(10,10), dpi=100)

# axes_s = ptr.subplot(211)
# axes_s.set_ylabel("#individuals")

# axes_i = ptr.subplot(212)
# axes_i.set_ylabel("Incidence")
# axes_i.set_xlabel("Days")

# r0s = []

# for j in z:
#     data_whole = pd.DataFrame()
#     r0s_inner = []

#     steps_per_day = j
#     steps = steps_per_day * 100

#     for i in range(no_sims):
#         print("Sim: " + str(i))
#         #Validation setting (allow dud runs)
#         # model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='constant',con_init=j)
#         # for i in range(steps):
#         #     model.step()
#         #     if model.infected == 0:
#         #         model.step()
#         #         break
#         while(True):
#             model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='constant',con_init=1)
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
# #     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
#     # data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='steps per day=' + str(j), sharex=True)
#     for i in range(3):
#         data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': V0=' + str(j), sharex=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='steps per day=' + str(j), sharex=True, logy=True, ylim=[1,1000])
#     for i in range(3):
#         data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': V0=' + str(j), sharex=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
#     print(data_whole.groupby(level=0).mean().max())
#     print(r0s_inner)
#     r0s.append(np.mean(r0s_inner))
# print(r0s)
# ptr.savefig(fname='pair_constant_high_threshold_stein_' + str(no_sims) + '_' + str(steps_per_day))
# ptr.show()

#CONSTANT INITIAL VIRAL LOAD

# z = [100] #,10,100,1000,10000]
# no_sims = 1

# ptr.figure(figsize=(10,10), dpi=100)

# axes_s = ptr.subplot(211)
# axes_s.set_ylabel("Prevalence (number of people)")

# axes_i = ptr.subplot(212)
# axes_i.set_ylabel("Incidence (number of people)")
# axes_i.set_xlabel("Days")

# r0s = []

# for j in z:
#     data_whole = pd.DataFrame()
#     r0s_inner = []
#     for i in range(no_sims):
#         print("Sim: " + str(i))
#         #Validation setting (allow dud runs)
#         # model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='constant',con_init=j)
#         # for i in range(steps):
#         #     model.step()
#         #     if model.infected == 0:
#         #         model.step()
#         #         break
#         while(True):
#             model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='constant',con_init=j)
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
# #     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
#     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='Prevalence (N=2500)', x='Days', y='Infected', label='V0=' + str(j), sharex=True)
#     # for i in range(5):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': V0=' + str(j), sharex=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='V0=' + str(j), sharex=True, logy=True, ylim=[1,1000])
#     # for i in range(5):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': V0=' + str(j), sharex=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
#     print(data_whole.groupby(level=0).mean().max())
#     print(r0s_inner)
#     r0s.append(np.mean(r0s_inner))
# print(r0s)
# ptr.savefig(fname='pair_constant_var_ptrans_stein_' + str(no_sims) + '_' + str(steps_per_day))
# ptr.show()


#LINEAR INITIAL VIRAL LOAD

# z = [(0.0001, 'blue'), (0.001, 'orange'), (0.01, 'green'), (0.1, 'red')]
# no_sims = 1

# ptr.figure(figsize=(10,10), dpi=100)

# # make a subplot for the susceptible, infected and recovered individuals
# axes_s = ptr.subplot(311, xlim=(0,100))
# axes_s.set_ylabel("Incidence")
# axes_s.set(xlabel="Days")

# axes_i = ptr.subplot(312, yscale='log', title='Initial Viral Loads', xlim=(0,100))
# axes_i.set_ylabel("Initial Viral Loads")
# axes_i.set(xlabel="Days")

# axes_r = ptr.subplot(313, xscale='log', yscale='log', title='Donor vs Recipient Initial Viral Loads',)
# # axes_r = ptr.subplot(313, title='Donor vs Recipient Initial Viral Loads',)
# axes_r.set_ylabel("Recipient Initial Viral Loads")
# axes_r.set_xlabel("Donor Initial Viral Loads")

# r0s = []


# for j, col in z:
    
#     # x_max = np.linspace(10000*j,10000000*j,1000)
#     # y_max = [j * get_peak_viral_load(i) for i in x_max]
#     # axes_r.plot(x_max,y_max)

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
# #     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
#     # data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='alpha=' + str(j), sharex=True)
#     # for i in range(3):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
#     # for i in range(3):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
#     # data_whole.groupby(level=0).plot(ax=axes_r, kind='scatter', title='Mean initial viral load (N=2500)', x='Days', y='Mean Initial Viral Load', label='alpha=' + str(j))
#     # print(data_whole.iloc[100])
#     corr_a = list()
#     corr_b = list()
#     for i, row in data_whole.iterrows():
#         # print(i)
#         if not len(row['Initial Viral Loads']) == 0:
#             axes_i.scatter(np.repeat(row['Days'],len(row['Initial Viral Loads'])),row['Initial Viral Loads'], c=col, label='alpha=' + str(j), alpha=0.3)
#             axes_r.scatter(row['Viral Load Correlation'].keys(),row['Viral Load Correlation'].values(), c=col, label='alpha=' + str(j), alpha=0.3)
#             for i in row['Viral Load Correlation']:
#                 corr_a.append(i)
#                 corr_b.append(row['Viral Load Correlation'][i])
#     print(str(j) + ' : ' + str(pearsonr(corr_a,corr_b)))
#     print(data_whole.groupby(level=0).mean().max())
#     # print(r0s_inner)
#     r0s.append(np.mean(r0s_inner))

# handles, labels = axes_r.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes_r.legend(by_label.values(), by_label.keys())

# handles, labels = axes_i.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes_i.legend(by_label.values(), by_label.keys())

# print(r0s)
# ptr.tight_layout()

# ptr.savefig(fname='paired_linear_varied_stein_' + str(no_sims) + '_' + str(steps_per_day))
# ptr.show()

#LOG INITIAL VIRAL LOAD

# z = [(1, 'blue'), (10, 'orange'), (100, 'green'), (1000, 'red')]
# no_sims = 1

# ptr.figure(figsize=(10,10), dpi=100)

# # make a subplot for the susceptible, infected and recovered individuals
# axes_s = ptr.subplot(311, xlim=(0,100))
# axes_s.set_ylabel("Incidence")
# axes_s.set(xlabel="Days")

# axes_i = ptr.subplot(312, yscale='log', title='Initial Viral Loads', xlim=(0,100))
# axes_i.set_ylabel("Initial Viral Loads")
# axes_i.set(xlabel="Days")

# axes_r = ptr.subplot(313, xscale='log', yscale='log', title='Donor vs Recipient Initial Viral Loads',)
# # axes_r = ptr.subplot(313, title='Donor vs Recipient Initial Viral Loads',)
# axes_r.set_ylabel("Recipient Initial Viral Loads")
# axes_r.set_xlabel("Donor Initial Viral Loads")

# r0s = []


# for j, col in z:
    
#     # x_max = np.linspace(10000*j,10000000*j,1000)
#     # y_max = [j * get_peak_viral_load(i) for i in x_max]
#     # axes_r.plot(x_max,y_max)

#     data_whole = pd.DataFrame()
#     r0s_inner = []
#     for i in range(no_sims):
#         print("Sim: " + str(i))
#         while(True):
#             model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='log',log_init=j)
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
# #     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
#     # data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='alpha=' + str(j), sharex=True)
#     # for i in range(3):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
#     # for i in range(3):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
#     # data_whole.groupby(level=0).plot(ax=axes_r, kind='scatter', title='Mean initial viral load (N=2500)', x='Days', y='Mean Initial Viral Load', label='alpha=' + str(j))
#     # print(data_whole.iloc[100])
#     corr_a = list()
#     corr_b = list()
#     for i, row in data_whole.iterrows():
#         # print(i)
#         if not len(row['Initial Viral Loads']) == 0:
#             axes_i.scatter(np.repeat(row['Days'],len(row['Initial Viral Loads'])),row['Initial Viral Loads'], c=col, label='alpha=' + str(j), alpha=0.3)
#             axes_r.scatter(row['Viral Load Correlation'].keys(),row['Viral Load Correlation'].values(), c=col, label='alpha=' + str(j), alpha=0.3)
#             for i in row['Viral Load Correlation']:
#                 corr_a.append(i)
#                 corr_b.append(row['Viral Load Correlation'][i])
#     print(str(j) + ' : ' + str(pearsonr(corr_a,corr_b)))
#     print(data_whole.groupby(level=0).mean().max())
#     # print(r0s_inner)
#     r0s.append(np.mean(r0s_inner))

# handles, labels = axes_r.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes_r.legend(by_label.values(), by_label.keys())

# handles, labels = axes_i.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes_i.legend(by_label.values(), by_label.keys())

# print(r0s)
# ptr.tight_layout()

# ptr.savefig(fname='paired_log_constant_stein_' + str(no_sims) + '_' + str(steps_per_day))
# ptr.show()

#SIGMOID INITIAL VIRAL LOAD

# z = [(1, 'blue'), (2, 'orange'), (5, 'green'), (10, 'red')]
# no_sims = 1

# ptr.figure(figsize=(10,10), dpi=100)

# # make a subplot for the susceptible, infected and recovered individuals
# axes_s = ptr.subplot(311, xlim=(0,100))
# axes_s.set_ylabel("Incidence")
# axes_s.set(xlabel="Days")

# axes_i = ptr.subplot(312, yscale='log', title='Initial Viral Loads', xlim=(0,100))
# axes_i.set_ylabel("Initial Viral Loads")
# axes_i.set(xlabel="Days")

# axes_r = ptr.subplot(313, xscale='log', yscale='log', title='Donor vs Recipient Initial Viral Loads',)
# # axes_r = ptr.subplot(313, title='Donor vs Recipient Initial Viral Loads',)
# axes_r.set_ylabel("Recipient Initial Viral Loads")
# axes_r.set_xlabel("Donor Initial Viral Loads")

# r0s = []


# for j, col in z:
    
#     # x_max = np.linspace(10000*j,10000000*j,1000)
#     # y_max = [j * get_peak_viral_load(i) for i in x_max]
#     # axes_r.plot(x_max,y_max)

#     data_whole = pd.DataFrame()
#     r0s_inner = []
#     for i in range(no_sims):
#         print("Sim: " + str(i))
#         while(True):
#             model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G='sigmoid',sig_init=j,lin_init=0.001)
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
# #     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
#     # data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='alpha=' + str(j), sharex=True)
#     # for i in range(3):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
#     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
#     # for i in range(3):
#     #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
#     # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
#     # data_whole.groupby(level=0).plot(ax=axes_r, kind='scatter', title='Mean initial viral load (N=2500)', x='Days', y='Mean Initial Viral Load', label='alpha=' + str(j))
#     # print(data_whole.iloc[100])
#     corr_a = list()
#     corr_b = list()
#     for i, row in data_whole.iterrows():
#         # print(i)
#         if not len(row['Initial Viral Loads']) == 0:
#             axes_i.scatter(np.repeat(row['Days'],len(row['Initial Viral Loads'])),row['Initial Viral Loads'], c=col, label='alpha=' + str(j), alpha=0.3)
#             axes_r.scatter(row['Viral Load Correlation'].keys(),row['Viral Load Correlation'].values(), c=col, label='alpha=' + str(j), alpha=0.3)
#             for i in row['Viral Load Correlation']:
#                 corr_a.append(i)
#                 corr_b.append(row['Viral Load Correlation'][i])
#     print(str(j) + ' : ' + str(pearsonr(corr_a,corr_b)))
#     print(data_whole.groupby(level=0).mean().max())
#     # print(r0s_inner)
#     r0s.append(np.mean(r0s_inner))

# handles, labels = axes_r.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes_r.legend(by_label.values(), by_label.keys())

# handles, labels = axes_i.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# axes_i.legend(by_label.values(), by_label.keys())

# print(r0s)
# ptr.tight_layout()

# ptr.savefig(fname='paired_sig_varied_stein_' + str(no_sims) + '_' + str(steps_per_day))
# ptr.show()

#COMPARISON INITIALISING FUNCTION

# z = [('sigmoid', 0, 'blue', '', 10), ('linear', 0, 'orange', '', 10), ('log', 0, 'green', '', 10)]
# z = [('sigmoid', -14000, 'blue', 'Late', 10), ('sigmoid', 0, 'orange', 'Mid', 10), ('sigmoid', 14000, 'green', 'Early', 10)]
z = [('sigmoid', 0, 'blue', '', 10), ('sigmoid', 0, 'orange', '', 5), ('sigmoid', 0, 'green', '', 2), ('sigmoid', 0, 'red', '', 1)]
no_sims = 1

ptr.figure(figsize=(10,10), dpi=100)

# make a subplot for the susceptible, infected and recovered individuals
axes_s = ptr.subplot(311, xlim=(0,100))
axes_s.set_ylabel("Incidence")
axes_s.set(xlabel="Days")

axes_i = ptr.subplot(312, yscale='log', title='Initial Viral Loads', xlim=(0,100))
axes_i.set_ylabel("Initial Viral Loads")
axes_i.set(xlabel="Days")

axes_r = ptr.subplot(313, xscale='log', yscale='log', title='Donor vs Recipient Initial Viral Loads',)
# axes_r = ptr.subplot(313, title='Donor vs Recipient Initial Viral Loads',)
axes_r.set_ylabel("Recipient Initial Viral Loads")
axes_r.set_xlabel("Donor Initial Viral Loads")

r0s = []


for j, d, col, per, s in z:
    
    # x_max = np.linspace(10000*j,10000000*j,1000)
    # y_max = [j * get_peak_viral_load(i) for i in x_max]
    # axes_r.plot(x_max,y_max)

    data_whole = pd.DataFrame()
    r0s_inner = []
    for i in range(no_sims):
        print("Sim: " + str(i))
        while(True):
            model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G=j, sig_init=s, delay=d, lin_init=0.01, log_init=400)
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
#     data_whole.groupby(level=0).mean().plot(ax=axes_s, title='v0 = ' + str(j), y=['Susceptible','Infected','Removed'],figsize=(8,6))
    # data_whole.groupby(level=0).mean().plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='alpha=' + str(j), sharex=True)
    # for i in range(3):
    #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_s, title='#infected (N=2500)', x='Days', y='Infected', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
    data_whole.groupby(level=0).mean().plot(ax=axes_s, title='Incidence (N=2500)', x='Days', y='Incidence', label=j + ': zeta=' + str(per) + str(s), logy=True)
    # data_whole.groupby(level=0).mean().plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='alpha=' + str(j), logy=True)
    # for i in range(3):
    #     data_whole.iloc[i*steps:(i+1)*steps-1].plot(ax=axes_i, title='Incidence (N=2500)', x='Days', y='Incidence', label='Run ' + str(i+1) +': alpha=' + str(j), sharex=True)
    # data_whole.groupby(level=0).mean().plot(ax=axes_r, title='R0 (N=2500)', x='Days', y="R0", label='V0=' + str(j))
    # data_whole.groupby(level=0).plot(ax=axes_r, kind='scatter', title='Mean initial viral load (N=2500)', x='Days', y='Mean Initial Viral Load', label='alpha=' + str(j))
    # print(data_whole.iloc[100])
    corr_a = list()
    corr_b = list()
    for i, row in data_whole.iterrows():
        # print(i)
        if not len(row['Initial Viral Loads']) == 0:
            axes_i.scatter(np.repeat(row['Days'],len(row['Initial Viral Loads'])),row['Initial Viral Loads'], c=col, label=j + ': zeta=' + str(per) + str(s), alpha=0.3)
            axes_r.scatter(row['Viral Load Correlation'].keys(),row['Viral Load Correlation'].values(), c=col, label=j + ': zeta=' + str(per) + str(s), alpha=0.3)
            for i in row['Viral Load Correlation']:
                corr_a.append(i)
                corr_b.append(row['Viral Load Correlation'][i])
    print(str(j) + ' : ' + str(pearsonr(corr_a,corr_b)))
    print(data_whole.groupby(level=0).mean().max())
    # print(r0s_inner)
    r0s.append(np.mean(r0s_inner))

handles, labels = axes_r.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes_r.legend(by_label.values(), by_label.keys())

handles, labels = axes_i.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes_i.legend(by_label.values(), by_label.keys())

print(r0s)
ptr.tight_layout()

ptr.savefig(fname='network_sig_varied_stein_' + str(no_sims) + '_' + str(steps_per_day))
ptr.show()