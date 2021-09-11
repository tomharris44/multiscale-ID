import time
import random
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.pyplot as ptr
import enum
import re
from scipy.integrate import odeint
from scipy.stats import pearsonr, spearmanr, poisson, ks_2samp, uniform
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import seaborn as sns

import networkx as nx

from multiprocessing import Pool, current_process

steps_per_day = 30
viral_load_max = 50000

p_trans_div = viral_load_max * steps_per_day

class InfectionMultiscaleModel(Model):
    """A model for infection spread."""

    def __init__(self, N=10, steps_per_day = 2,
                 v_threshold=300000, G='constant', F='linear',
                 con_init=5, lin_init=1, log_init=1,
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
        self.F = F
        self.r0 = 0
        self.ptrans = 1.0 / steps_per_day
        self.viral_load_max = 25000
        self.inoc_max = 50000
        self.inoc_max_sig = np.power(self.inoc_max/2,self.sig_init)
        self.infected = 0
        self.removed = 0
        self.viral_load_tree = dict()

        self.sig_infector_multiplier = self.lin_init * self.inoc_max
        self.sig_ptrans_multiplier = self.inoc_max / p_trans_div

        self.con_ptrans_multiplier = self.sig_ptrans_multiplier / 2

        if G == 'constant':
            # self.v_threshold = 1 / self.lin_init
            self.v_threshold = 5000
        elif G == 'random':
            # self.v_threshold = 1 / self.lin_init
            self.v_threshold = 5000
        elif G == 'linear':
            # self.v_threshold = 1 / self.lin_init
            self.v_threshold = 5000
        elif G == 'log':
            self.v_threshold = np.exp(1 / (self.log_init * self.inoc_max))
        elif G == 'sigmoid':
            beta = self.lin_init * self.inoc_max
            self.v_threshold = np.power(self.inoc_max_sig/(beta * (1 - (1/beta))),1/self.sig_init) - self.delay
            # self.v_threshold = 5000

        # print(self.lin_init,self.sig_init,self.v_threshold)

        # Create agents
        for i in range(self.num_agents):
            a = MSMyAgent(i, self)
            self.schedule.add(a)
            if i == 1:
                a.state = MSState.INFECTED
                a.infection_course = a.infect_stein_init('constant',a.unique_id,1,1)

        self.datacollector = DataCollector(
            model_reporters={"r0" : "r0",
                            "viral_load_tree" : "viral_load_tree"},
            agent_reporters={"State": "state"})

    def step(self):
        self.datacollector.collect(self)
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
            init_v = self.random.randint(1,450)
        elif G == 'bottleneck':
            init_v = np.minimum(1, donor * 0.000001)
        elif G == 'linear':
            # init_v = self.model.lin_init * donor
            foo = self.model.lin_init * donor
            init_v = np.mean(poisson.rvs(foo, size=100))
        elif G == 'log':
            init_v = self.model.log_init * np.log(donor)
        elif G == 'sigmoid':
            # init_v = self.model.lin_init * self.model.inoc_max * np.power(donor + self.model.delay,self.model.sig_init) / (np.power(donor + self.model.delay,self.model.sig_init) + self.model.inoc_max_sig)
            foo = self.model.sig_infector_multiplier * np.power(donor + self.model.delay,self.model.sig_init) / (np.power(donor + self.model.delay,self.model.sig_init) + self.model.inoc_max_sig)
            init_v = np.mean(poisson.rvs(foo, size=100))

        self.model.infected += 1
        self.model.viral_load_tree[(donor_id,self.unique_id)] = (donor_init_v,init_v,donor,self.infection_time)

        self.init_v = init_v
            
        y0 = (init_v, 0, n0, p0)
        
        solution = odeint(differential_stein, y0, t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
        solution = [[row[i] for row in solution] for i in range(4)]

        return solution

    def infect_stein_init(self, G, donor_id, donor, donor_init_v):
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

        self.model.infected += 1
        # self.model.viral_load_tree[(donor_id,self.unique_id)] = (donor_init_v,init_v,donor)

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
                self.model.removed += 1


    def contact(self):
        """Find close contacts and infect"""
        
        if self.state is MSState.INFECTED:
            cellmates = self.random.sample(self.model.schedule.agents,k=1)
            t = self.model.schedule.time-self.infection_time
            if t>0:
                var_ptrans = self.get_ptrans(self.infection_course[0][t], self.model.F)
                for other in cellmates:
                    # if other.state is MSState.SUSCEPTIBLE and self.infection_course[0][t] > self.model.v_threshold and self.random.random() < var_ptrans:                 
                    if other.state is MSState.SUSCEPTIBLE and self.random.random() < var_ptrans:
                        other.state = MSState.INFECTED
                        other.infection_time = self.model.schedule.time
                        other.infection_course = other.infect_stein(self.model.G,self.unique_id,self.infection_course[0][t],self.init_v)
                        if self.unique_id == 1:
                            self.model.r0 += 1
                        # print(self.init_v,var_ptrans,self.infection_course[0][t],other.init_v)
                    
    def step(self):
        self.status()
        self.contact()

    def get_ptrans(self,V,F):
        if V < self.model.v_threshold:
            return 0
        if F == 'linear':
            return V / p_trans_div
        if F == 'sigmoid':
            return self.model.sig_ptrans_multiplier * np.power(V,self.model.sig_init) / (np.power(V,self.model.sig_init) + np.power(self.model.inoc_max/2,self.model.sig_init))
        if F == 'random':
            return self.random.random() * self.model.sig_ptrans_multiplier
        if F == 'constant':
            return self.model.con_ptrans_multiplier

def get_column_data(model):
    """pivot the model dataframe to get states count at each step"""
    agent_state = model.datacollector.get_agent_vars_dataframe()
    model_state = model.datacollector.get_model_vars_dataframe()

    X = pd.pivot_table(agent_state.reset_index(),index='Step',columns='State',aggfunc=np.size,fill_value=0)    
    X['r0'] = model_state['r0']
    X['viral_load_tree'] = model_state['viral_load_tree']

    labels = ['Susceptible','Infected','Removed','R0','Viral Load Tree']
    X.columns = labels[:len(X.columns)]
    X['Incidence'] = X['Susceptible'].diff() * -1
    X['Recovery'] = X['Removed'].diff()


    for j in range(X.shape[0],steps):
        X.loc[j] = 0
        X['Viral Load Tree'].loc[j] = []
    X['Days'] = X.index
    X['Days'] = X['Days'].div(steps_per_day)


    X['Incidence Sum'] = X['Incidence']
    X['Incidence Days'] = 0

    for i in range(0,days):
        X['Incidence Sum'].loc[i] = X['Incidence'][i*steps_per_day:(i+1)*steps_per_day].sum()
        X['Incidence Days'].loc[i] = i
    return X

def get_peak_viral_load(init):
    t = np.linspace(0, 12, num=12 * steps_per_day)
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

    y0 = (init, 0, n0, p0)
        
    solution = odeint(differential_stein, y0, t, args=(r,kI,kN,kP,aI,aN,aP,bI,dN,c,kV,KI,tN,tP))
    solution = [[row[i] for row in solution] for i in range(4)]

    return max(solution[0])


def get_all_lengths(G,root):

    paths = []

    for node in G:
        if G.out_degree(node)==0:
            paths.append(nx.shortest_path_length(G, source=root, target=node))
    
    return paths

def get_avg_length(G,root):
    paths = []

    for node in G:
        if G.out_degree(node)==0:
            paths.append(nx.shortest_path_length(G, source=root, target=node))

    return np.mean(paths)



days = 100
steps = steps_per_day * days
pop = 1000

# ind_inits = [4.5,14.2,45,142.2,450]
# ind_inits_strs = ['4_5','14_2','45','142_2','450']
ind_inits = np.logspace(np.log10(4.5),np.log10(450),9)
ind_inits = [450]
ind_inits_strs = [re.sub('\.','_',str(round(h,2))) for h in ind_inits]
# ptrans = ['sigmoid','linear','random','constant']
ptrans = ['linear','random'] 
# dv_rvs = ['sigmoid','linear','random']
dv_rvs = ['sigmoid']

work = [[c,a,ind_inits[b],ind_inits_strs[b]] for b in range(len(ind_inits)) for a in ptrans for c in dv_rvs]

print(work)

# print(work)

ptr.close("all")

data_whole = pd.DataFrame()

# sig_results =  [0,10.5,9.777777777777779, 7.571428571428571, 10.38888888888889]

# SUBTREE ANALYSIS

no_sims = 5

def diffusion(p):
    ptr.figure(figsize=(10,10), dpi=100)

    trees = []

    data_whole = pd.DataFrame()
    r0s_inner = []
    for i in range(no_sims):
        # print("Sim: " + str(i))
        counter = 0
        while(True):
            model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G=p[0], F=p[1], con_init=p[2], sig_init=10, delay=0, lin_init=0.01, log_init=400)
            for i in range(steps):
                model.step()
                if model.infected == 0:
                    model.step()
                    break
            if model.r0 != 0 and model.removed > 150:
                break
            counter += 1
            if counter == 20:
                print('No result: ' + str(p[2]))
                break

        data = get_column_data(model)
        tree = nx.DiGraph()

        for i, row in data.iterrows():
            if not len(row['Viral Load Tree']) == 0:
                for i in row['Viral Load Tree']:
                    if tree.number_of_nodes() < 10000:
                        if tree.has_node(i[0]):
                            tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2), time=round(row['Viral Load Tree'][i][3],2), gen=tree.nodes[i[0]]['gen'] + 1)
                            tree.add_edge(i[0],i[1], rec_init=round(row['Viral Load Tree'][i][1],2))
                        else:
                            tree.add_node(i[0], init_v=round(row['Viral Load Tree'][i][0],2), time=0, gen=0)
                            tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2), time=round(row['Viral Load Tree'][i][3],2), gen=tree.nodes[i[0]]['gen'] + 1)
                            tree.add_edge(i[0],i[1],rec_init=round(row['Viral Load Tree'][i][1],2)) 

        trees.append(tree)
        
        if i==0:
            data_whole = data
        else:
            data_whole = data_whole.append(data)

        print(p[2], max(data['Removed']))

        ptr.close("all")

        # ax = plt.subplot(111)
        # tree_display = trees[0]
        # ax.set_title('Transmission Tree')
        # ax.set_xticks([])

        # # for i, row in data_whole.iterrows():
        # #     if not len(row['Viral Load Tree']) == 0:
        # #         for i in row['Viral Load Tree']:
        # #             if tree.number_of_nodes() < 10000:
        # #                 if tree.has_node(i[0]):
        # #                     tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2), time=round(row['Viral Load Tree'][i][3],2), gen=tree.nodes[i[0]]['gen'] + 1)
        # #                     tree.add_edge(i[0],i[1], rec_init=round(row['Viral Load Tree'][i][1],2))
        # #                 else:
        # #                     tree.add_node(i[0], init_v=round(row['Viral Load Tree'][i][0],2), time=0, gen=0)
        # #                     tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2), time=round(row['Viral Load Tree'][i][3],2), gen=tree.nodes[i[0]]['gen'] + 1)
        # #                     tree.add_edge(i[0],i[1],rec_init=round(row['Viral Load Tree'][i][1],2)) 

        # pos=nx.drawing.nx_agraph.graphviz_layout(tree_display, prog='dot')
        # max_time = max([tree_display.nodes[i]['time'] for i in tree_display.nodes()])

        # for node in tree_display.nodes():
        #     pos[node] = (pos[node][0],-1*tree_display.nodes[node]['time'])

        # nodes = nx.draw_networkx_nodes(tree_display, ax=ax, pos=pos, node_size=50,
        #                 node_color=list(nx.get_node_attributes(tree_display, 'init_v').values()),
        #                 cmap=ptr.cm.viridis, vmin=0)
        # edges = nx.draw_networkx_edges(tree_display, ax=ax, pos=pos, node_size=50)
        # edge_labels_dict = nx.get_edge_attributes(tree_display,'rec_init')
        # edge_labels = nx.draw_networkx_edge_labels(tree_display, ax=ax, pos=pos, font_size=7, edge_labels=edge_labels_dict)
        
        # ptr.colorbar(nodes)

        # yticks = range(0,-1*max_time,(-3 * steps_per_day))
        # labelsy = [ round(-1*i/steps_per_day) for i in yticks]
        
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(labelsy)

        # ax.set_ylabel('Days')

        # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # ptr.tight_layout()

        # ptr.savefig(fname='trans_tree_' + p[0] + '_' + p[1] + '_' + p[3])
        # ptr.show()

    binned_time = []
    bins_time = np.linspace(0,days,num=21)
    for g in bins_time:
        binned_time.append([])

    binned_gen = []
    # bins_gen = list(range(0,max_gen+3,2))
    bins_gen = list(range(0,31,1))
    # print(max_gen,bins_gen)
    for g in bins_gen:
        binned_gen.append([])

    
    # coherence = []
    # rand = uniform.rvs(loc=1,scale=499,size=1000)
    # rand_mean = np.mean(rand)
    # rand_std = np.std(rand)
    # rand_q3 = np.percentile(rand,75)
    # rand_q1 = np.percentile(rand,25)

    # sig_sig = [21.95, 445.06, 423.3, 242.42, 333.05, 9.25, 273.99, 462.91, 387.95, 166.09, 473.02, 145.35, 388.79, 405.45, 445.74, 468.24, 468.98, 437.21, 251.74, 280.08, 453.87, 296.85, 470.34, 463.05, 460.68, 450.29, 382.23, 294.97, 465.76, 469.97, 186.41, 464.76, 470.48, 303.5, 121.07, 385.71, 458.2, 35.63, 202.51, 446.06, 455.86, 473.04, 385.39, 258.18, 456.71, 70.64, 457.08, 194.55, 359.54, 388.93, 216.82, 467.27, 464.08, 422.74, 330.78, 464.6, 446.05, 450.22, 439.89, 259.55, 449.65, 467.81, 257.94, 363.86, 353.84, 438.11, 459.06, 460.98, 373.65, 407.65, 463.56, 470.02, 396.05, 463.69, 389.46, 36.03, 445.54, 466.95, 465.96, 447.49, 468.18, 404.95, 444.86, 366.44, 451.18, 291.16, 454.61, 235.47, 231.77, 254.33, 460.58, 453.18, 465.19, 456.6, 394.83, 470.9, 473.33, 353.26, 445.66, 277.16, 450.29, 432.3, 360.09, 322.69, 430.05, 438.71, 354.52, 417.46, 470.0, 360.93, 179.44, 116.88, 472.51, 316.99, 439.59, 378.56, 316.21, 299.68, 336.73, 367.55, 438.63, 415.36, 353.52, 293.04, 440.01, 465.75, 473.0, 429.06, 208.32, 413.62, 445.01, 425.33, 459.26, 332.22, 387.16, 402.03, 459.76, 465.84, 408.91, 470.88, 437.0, 465.39, 396.46, 160.93, 449.67, 322.42, 381.61, 455.45, 424.93, 464.54, 361.41, 410.97, 430.67, 469.77, 340.26, 449.55, 444.95, 233.11, 318.38, 389.67, 121.34, 471.52, 462.21, 235.83, 467.45, 135.94, 204.7, 462.27, 435.82, 470.31, 187.47, 105.71, 472.87, 459.01, 460.61, 17.5, 403.8, 350.51, 299.17, 374.73, 74.97, 427.45, 456.42, 290.29, 293.66, 299.88, 443.27, 340.98, 445.74, 456.45, 190.77, 442.69, 371.26, 460.85, 466.53, 456.06, 361.51, 289.83, 458.25, 450.33, 396.63, 470.71, 458.57, 442.18, 463.41, 225.75, 32.56, 115.48, 366.78, 451.54, 449.9, 432.17, 462.97, 192.21, 163.52, 442.47, 467.45, 459.9, 435.76, 468.43, 452.91, 364.63, 436.07, 425.12,
    # ]

    # deco_times = []
    # deco_times_kp = []


    for tree in trees:
        coherence_gen = []
        coh_bin_gen = []

        for g in bins_gen:
            coh_bin_gen.append([])

        for node in tree.nodes():
            sub = nx.dfs_tree(tree,node)
            bin_no_time = np.digitize(tree.nodes[node]['time']/steps_per_day,bins_time)
            bin_no_gen = np.digitize(tree.nodes[node]['gen'],bins_gen)

            binned_time[bin_no_time].append(tree.nodes[node]['init_v'])
            if bin_no_gen < len(binned_gen):
                binned_gen[bin_no_gen].append(tree.nodes[node]['init_v'])
                coh_bin_gen[bin_no_gen].append(tree.nodes[node]['init_v'])

        bin_sizes = [len(h) for h in coh_bin_gen]

        # print('SIZES: ', bin_sizes)

        largest_bin = bin_sizes.index(max(bin_sizes))

        # print('LARGEST: ', largest_bin)
        print(p[1], coh_bin_gen[largest_bin])

        # sig_sig = [21.95, 445.06, 423.3, 242.42, 333.05, 9.25, 273.99, 462.91, 387.95, 166.09, 473.02, 145.35, 388.79, 405.45, 445.74, 468.24, 468.98, 437.21, 251.74, 280.08, 453.87, 296.85, 470.34, 463.05, 460.68, 450.29, 382.23, 294.97, 465.76, 469.97, 186.41, 464.76, 470.48, 303.5, 121.07, 385.71, 458.2, 35.63, 202.51, 446.06, 455.86, 473.04, 385.39, 258.18, 456.71, 70.64, 457.08, 194.55, 359.54, 388.93, 216.82, 467.27, 464.08, 422.74, 330.78, 464.6, 446.05, 450.22, 439.89, 259.55, 449.65, 467.81, 257.94, 363.86, 353.84, 438.11, 459.06, 460.98, 373.65, 407.65, 463.56, 470.02, 396.05, 463.69, 389.46, 36.03, 445.54, 466.95, 465.96, 447.49, 468.18, 404.95, 444.86, 366.44, 451.18, 291.16, 454.61, 235.47, 231.77, 254.33, 460.58, 453.18, 465.19, 456.6, 394.83, 470.9, 473.33, 353.26, 445.66, 277.16, 450.29, 432.3, 360.09, 322.69, 430.05, 438.71, 354.52, 417.46, 470.0, 360.93, 179.44, 116.88, 472.51, 316.99, 439.59, 378.56, 316.21, 299.68, 336.73, 367.55, 438.63, 415.36, 353.52, 293.04, 440.01, 465.75, 473.0, 429.06, 208.32, 413.62, 445.01, 425.33, 459.26, 332.22, 387.16, 402.03, 459.76, 465.84, 408.91, 470.88, 437.0, 465.39, 396.46, 160.93, 449.67, 322.42, 381.61, 455.45, 424.93, 464.54, 361.41, 410.97, 430.67, 469.77, 340.26, 449.55, 444.95, 233.11, 318.38, 389.67, 121.34, 471.52, 462.21, 235.83, 467.45, 135.94, 204.7, 462.27, 435.82, 470.31, 187.47, 105.71, 472.87, 459.01, 460.61, 17.5, 403.8, 350.51, 299.17, 374.73, 74.97, 427.45, 456.42, 290.29, 293.66, 299.88, 443.27, 340.98, 445.74, 456.45, 190.77, 442.69, 371.26, 460.85, 466.53, 456.06, 361.51, 289.83, 458.25, 450.33, 396.63, 470.71, 458.57, 442.18, 463.41, 225.75, 32.56, 115.48, 366.78, 451.54, 449.9, 432.17, 462.97, 192.21, 163.52, 442.47, 467.45, 459.9, 435.76, 468.43, 452.91, 364.63, 436.07, 425.12,
        # 384.43, 164.17, 470.92, 251.74, 462.4, 470.51, 455.58, 459.86, 455.79, 439.68, 218.05, 417.46, 389.61, 276.26, 406.34, 324.87, 193.63, 108.76, 8.01, 472.79, 455.83, 394.17, 451.98, 470.11, 423.03, 406.47, 440.3, 457.85, 458.68, 452.4, 400.65, 189.55, 400.16, 464.93, 320.11, 239.87, 435.97, 434.06, 443.42, 340.44, 463.7, 473.15, 467.18, 467.91, 472.17, 452.17, 474.06, 462.88, 286.31, 278.14, 424.07, 472.69, 175.43, 374.46, 473.16, 465.77, 369.64, 448.79, 68.69, 370.59, 468.9, 470.54, 372.04, 429.02, 368.61, 426.25, 389.24, 305.18, 344.46, 363.57, 473.58, 471.96, 395.66, 465.36, 463.06, 147.11, 473.61, 448.56, 369.41, 469.0, 456.91, 416.57, 441.02, 457.73, 241.59, 465.52, 431.29, 230.36, 391.12, 379.6, 464.5, 473.12, 469.67, 357.92, 378.85, 420.24, 243.27, 457.13, 445.14, 458.4, 466.29, 363.81, 388.36, 473.36, 451.29, 465.28, 439.68, 447.35, 445.14, 468.96, 263.61, 377.16, 402.54, 119.66, 471.68, 423.69, 472.29, 340.97, 412.59, 433.9, 473.82, 356.45, 292.96, 38.57, 468.63, 469.38, 424.95, 467.03, 454.07, 471.65, 468.66, 311.93, 292.95, 461.66, 361.83, 293.49, 265.98, 470.62, 468.98, 385.79, 436.37, 182.54, 264.2, 466.98, 360.85, 436.02, 467.53, 295.7, 473.26, 459.34, 293.31, 264.22, 390.95, 469.74, 391.3, 256.47, 466.79, 408.06, 115.7, 458.91, 471.35, 457.56, 334.93, 465.9,
        # 231.51, 424.52, 352.99, 469.84, 464.01, 471.91, 427.99, 95.55, 277.08, 465.13, 402.86, 250.98, 452.49, 457.87, 427.05, 443.69, 466.37, 382.81, 119.56, 469.29, 460.5, 92.18, 466.87, 470.37, 432.45, 455.96, 224.07, 142.75, 263.84, 174.47, 468.27, 322.19, 455.95, 349.44, 278.02, 448.01, 453.83, 440.98, 471.09, 443.72, 419.48, 474.84, 110.41, 230.46, 449.95, 454.07, 337.11, 474.94, 442.51, 189.98, 381.48, 100.87, 64.9, 13.07, 400.09, 426.99, 464.86, 456.41, 463.38, 457.21, 242.24, 136.34, 400.15, 457.69, 474.53, 440.99, 443.57, 451.46, 381.44, 365.63, 424.79, 365.82, 201.03, 412.45, 418.09, 213.58, 466.39, 365.02, 467.1, 421.88, 464.67, 470.48, 466.17, 472.34, 454.61, 427.52, 267.24, 303.87, 466.86, 421.01, 117.07, 435.14, 466.94, 275.96, 420.08, 462.56, 451.82, 474.18, 461.89, 472.66, 388.86, 447.49, 442.04, 462.86, 470.15, 475.05, 367.83, 469.86, 428.87, 96.76, 460.98, 448.99, 55.7, 463.31, 364.56, 468.9, 456.98, 281.55, 438.83, 418.36, 420.16, 463.48, 446.64, 323.11, 432.23, 389.09, 471.77, 324.89, 319.14, 391.93, 468.82, 411.14, 460.69, 469.96, 445.37, 442.26, 461.29, 469.69, 317.61, 295.04, 205.51, 343.47, 469.48, 425.1, 375.63, 7.49, 395.46, 466.59, 369.16, 465.01, 455.75, 458.91, 256.62, 443.14, 460.04, 454.93, 273.56, 409.77, 442.8, 390.48, 202.28, 398.5, 459.84, 178.1, 179.47, 450.42, 301.28, 434.01, 417.91, 364.99, 441.22, 459.19, 419.48, 411.79, 421.98, 386.39, 330.79, 472.32, 279.36, 419.69, 394.09, 264.83, 159.47, 205.56, 279.47, 425.88, 244.18, 469.52, 376.64, 69.36, 388.68, 466.97, 465.61, 375.85, 445.63, 268.51, 445.38, 443.47, 422.22, 468.34, 418.83, 421.63, 345.81, 470.12, 296.47, 410.5, 367.88, 458.57, 12.42, 279.52, 7.79, 238.03, 232.8, 162.6, 316.33, 453.0, 467.84, 416.03, 455.77, 396.99, 451.16, 450.55, 299.62, 466.07, 224.44, 342.36, 368.57, 386.56, 201.8, 466.62, 468.0, 465.45, 468.88, 451.18, 468.37, 454.08, 448.44, 147.43, 399.23, 37.3,
        # 459.27, 353.38, 454.72, 24.85, 455.92, 453.54, 435.67, 425.06, 333.87, 105.9, 450.51, 449.95, 277.63, 473.5, 434.69, 431.15, 473.65, 439.86, 463.7, 375.51, 462.99, 373.22, 429.93, 426.69, 314.06, 6.69, 283.97, 472.78, 208.04, 357.56, 177.86, 468.85, 469.09, 230.56, 470.7, 374.54, 457.04, 346.98, 471.65, 345.64, 467.33, 474.53, 118.44, 467.29, 427.36, 455.55, 288.66, 183.99, 338.8, 244.85, 238.08, 466.93, 466.79, 439.68, 258.26, 471.24, 430.84, 403.15, 462.87, 338.5, 443.61, 412.11, 343.21, 300.87, 384.26, 468.32, 429.02, 444.43, 460.38, 448.17, 441.07, 464.32, 447.49, 329.17, 469.03, 150.39, 450.45, 469.72, 473.61, 222.28, 241.12, 446.51, 289.94, 448.64, 468.08, 448.19, 232.68, 328.0, 406.18, 465.79, 471.28, 426.38, 459.51, 457.23, 362.36, 458.19, 177.24, 388.24, 440.77, 470.07, 371.49, 455.22, 255.43, 388.13, 394.86, 447.12, 382.29, 300.01, 19.49, 457.6, 415.59, 218.36, 464.94, 444.83, 438.22, 457.51, 402.31, 462.92, 469.4, 470.76, 461.9, 435.77, 228.42, 429.17, 436.1, 440.83, 321.56, 456.95, 457.87, 444.62, 236.3, 289.09, 450.54, 456.87, 327.26, 409.66, 464.52, 194.91, 447.45, 100.47, 204.57, 474.3, 460.52, 383.38, 465.34, 466.62, 435.16, 398.32, 461.08, 337.41, 458.27, 438.35, 426.65, 220.03, 357.76, 456.97, 447.48, 411.54, 117.39, 297.16, 320.91, 448.24, 127.09, 380.68, 444.7, 474.81, 447.61, 224.84, 465.02, 471.42, 333.11, 460.72, 378.31, 94.27, 319.4, 457.14, 73.66, 442.27, 207.97, 360.35, 415.32, 454.08, 391.88, 65.77, 459.16, 465.22, 240.7, 136.88, 450.38, 388.28, 418.96, 289.97, 357.2, 340.88, 279.37, 380.72, 468.08, 463.36, 469.75, 402.57, 471.76, 375.55, 389.54, 466.88, 366.54, 468.4, 192.08, 299.43, 466.15, 463.57,
        # 163.37, 341.71, 358.14, 32.19, 467.54, 460.2, 354.4, 438.41, 453.52, 435.21, 467.15, 306.19, 51.1, 296.12, 400.89, 127.27, 474.97, 386.78, 241.05, 412.36, 436.54, 373.1, 231.71, 405.54, 463.52, 406.82, 444.54, 454.18, 461.77, 465.95, 467.97, 425.02, 430.38, 437.64, 370.52, 441.8, 388.24, 464.54, 430.25, 422.67, 414.82, 432.77, 392.35, 336.49, 429.99, 445.8, 411.14, 463.35, 156.92, 360.82, 463.88, 425.53, 454.34, 398.85, 384.05, 95.56, 457.97, 328.9, 376.92, 465.07, 428.06, 456.24, 467.73, 166.88, 376.2, 427.88, 243.0, 394.67, 419.32, 260.44, 465.9, 465.82, 384.46, 444.44, 218.89, 406.35, 404.92, 434.18, 263.51, 418.99, 449.02, 418.11, 473.98, 308.78, 386.41, 389.19, 253.8, 440.63, 439.48, 195.26, 367.58, 301.94, 391.06, 418.47, 453.02, 470.05, 470.3, 135.16, 462.23, 460.62, 219.47, 474.97, 426.61, 465.5, 384.46, 430.57, 366.04, 267.08, 23.88, 452.6, 467.98, 470.8, 459.13, 231.34, 461.0, 422.14, 215.95, 434.34, 360.94, 300.72, 462.57, 458.09, 293.04, 115.48, 446.4, 443.22, 457.39, 454.65, 416.64, 468.13, 349.49, 445.9, 301.89, 460.5, 51.55, 410.27, 459.59, 473.15, 348.56, 462.85, 422.76, 426.31, 471.25, 416.81, 459.46, 469.37, 470.48, 429.07, 400.22, 458.76, 459.96, 464.9, 453.33, 378.33, 105.0, 431.11, 320.64, 470.92, 454.58, 163.22, 373.4, 423.0, 457.89, 464.55, 473.27, 467.64, 466.64, 427.19, 328.2, 465.39, 291.22, 474.02, 469.66, 249.0, 465.98, 466.86, 459.16, 387.85, 386.32, 59.28, 444.87, 109.96, 467.81, 447.44, 426.95, 455.76, 339.7, 457.53, 402.01, 360.3, 408.25, 434.81, 455.87, 462.3, 407.53, 468.0, 465.39, 463.37, 361.13, 464.9, 470.53, 378.09, 413.18, 466.7, 440.37, 400.44, 469.16, 447.72, 451.21, 361.95, 350.5, 465.65, 448.63, 455.08, 462.94, 454.65, 395.66, 452.74, 455.61, 397.64, 461.52, 450.52, 176.73, 439.86, 151.95, 454.75, 435.84, 400.5, 225.49, 463.25, 128.37, 466.27, 330.76, 358.46, 266.11, 273.51, 466.07, 441.02, 407.15,
        # 112.6, 404.35, 351.95, 470.4, 454.14, 353.3, 445.99, 416.09, 407.39, 467.52, 430.34, 455.73, 341.71, 472.15, 466.49, 468.8, 457.95, 307.05, 267.93, 292.69, 450.7, 357.68, 457.83, 469.7, 127.87, 472.58, 453.02, 8.61, 250.64, 350.32, 471.72, 447.54, 467.84, 366.83, 448.51, 386.02, 471.31, 377.12, 471.35, 468.22, 460.0, 459.2, 458.99, 454.95, 438.61, 470.59, 457.53, 330.07, 425.33, 452.25, 471.76, 387.93, 427.09, 357.83, 471.26, 13.6, 465.44, 421.59, 401.34, 413.83, 459.25, 440.03, 388.05, 451.45, 463.05, 464.26, 468.68, 419.42, 272.14, 473.58, 300.7, 416.27, 457.84, 154.39, 425.27, 471.41, 471.71, 407.2, 377.15, 455.56, 470.97, 40.66, 429.97, 365.06, 403.79, 412.98, 467.81, 442.99, 443.04, 404.52, 298.9, 347.77, 250.45, 447.89, 130.42, 464.57, 445.32, 436.9, 451.71, 353.92, 311.77, 377.0, 466.8, 422.09, 392.25, 441.04, 307.69, 455.33, 346.44, 467.56, 424.26, 415.74, 275.85, 358.49, 465.72, 391.04, 450.98, 466.83, 466.69, 48.18, 222.14, 85.93, 387.9, 193.59, 318.06, 455.34, 408.08, 439.29, 461.74, 306.52, 443.46, 123.59, 130.3, 374.09, 465.17, 463.04, 439.53, 115.59, 454.26, 417.4, 455.98, 460.03, 459.55, 241.21, 466.96, 351.78, 472.58, 466.52, 336.82, 439.67, 381.38, 402.71, 448.46, 472.09, 373.76, 323.53, 321.33, 421.41, 470.08, 39.01, 467.11, 440.64, 466.66, 304.72, 401.64, 29.71, 286.58, 277.89, 457.95, 439.47, 351.57, 425.0, 407.74, 432.59, 444.93, 46.1, 473.11, 412.24, 273.38, 447.28, 461.17, 257.58, 470.73, 374.36, 118.78, 434.27, 411.8]

        # all_bins = [h for g in coh_bin_gen for h in g]

        # gen_threshold = 20
        # gen_count = True

        # gen_threshold_kp = 0.99
        # gen_count_kp = True
        
        # for b in range(1,len(coh_bin_gen)-1):
        #     # print(coh_bin_gen[b])
        #     if coh_bin_gen[b] and coh_bin_gen[b-1] and coh_bin_gen[b+1]:
        #         # current_bin = coh_bin_gen[b] + coh_bin_gen[b-1] + coh_bin_gen[b+1]
        #         current_bin = coh_bin_gen[b]
        #         # current_bin_q1 = np.percentile(current_bin,25)
        #         # current_bin_q3 = np.percentile(current_bin,75)

        #         # # rand_res = ks_2samp(coh_bin_gen[b],rand)
        #         # rand_res = ks_2samp(current_bin,rand)
        #         # # prev_bin_res = ks_2samp(coh_bin_gen[b],coh_bin_gen[b-1])
        #         # prev_bin_res = ks_2samp(current_bin,coh_bin_gen[b])
        #         # # max_bin_res = ks_2samp(coh_bin_gen[b],coh_bin_gen[largest_bin])
        #         # max_bin_res = ks_2samp(current_bin,coh_bin_gen[largest_bin])
        #         # # whole_res = ks_2samp(coh_bin_gen[b],all_bins)
        #         # whole_res = ks_2samp(rand,sig_sig)
        #         # if whole_res[1] > gen_threshold_kp and gen_count_kp and len(current_bin) > 20:
        #         #     deco_times_kp.append(b)
        #         #     gen_count_kp = False
        #         # print('PREV BIN: ',p[3], b, prev_bin_res[1])
        #         # print('RAND: ',p[3], b, rand_res[1])
        #         # print('LARGE BIN: ',p[3], b, max_bin_res[1])
        #         # print('WHOLE: ',p[3], b, whole_res[1])
        #         # coherence_gen.append([rand_res[1],prev_bin_res[1], max_bin_res[1], whole_res[1]])
                

        #         # rand_mean_res = abs(rand_q1 - current_bin_q1) + abs(rand_q3 - current_bin_q3)
        #         # whole_mean_res = abs(np.percentile(all_bins,25) - current_bin_q1) + abs(np.percentile(all_bins,75) - current_bin_q3)
        #         # large_mean_res = abs(np.percentile(coh_bin_gen[largest_bin],25) - current_bin_q1) + abs(np.percentile(coh_bin_gen[largest_bin],75) - current_bin_q3)
        #         # prev_mean_res = abs(np.percentile(coh_bin_gen[b],25) - current_bin_q1) + abs(np.percentile(coh_bin_gen[b],75) - current_bin_q3)
        #         # whole_mean_res = abs(np.percentile(sig_sig,25) - current_bin_q1) + abs(np.percentile(sig_sig,75) - current_bin_q3)
        #         # whole_mean_res = abs(rand_q1 - current_bin_q1) + abs(rand_q3 - current_bin_q3)
        #         # if whole_mean_res < gen_threshold and gen_count:
        #         #     deco_times.append(b)
        #         #     gen_count = False
                    
        #         # print('PREV BIN: ',p[3], b, prev_mean_res)
        #         # print('RAND: ',p[3], b, rand_mean_res)
        #         # print('LARGE BIN: ',p[3], b, large_mean_res)
        #         # print('WHOLE: ',p[3], b, whole_mean_res)
        #         # coherence_gen.append([whole_mean_res])


        #         coherence_gen.append()

        #     else:
        #         coherence_gen.append([0,0,0,0])
        # coherence.append(coherence_gen)

    #TODO: take mean of all simulations in coherence

    # print(p[2], deco_times, np.mean(deco_times))
    # print(p[2], deco_times_kp, np.mean(deco_times_kp))


    medians_gen = [np.median(h) for h in binned_gen[1:] if h]
    qts_upper_gen = [np.percentile(h,97.5) for h in binned_gen[1:] if h]
    qts_lower_gen = [np.percentile(h,2.5) for h in binned_gen[1:] if h]

    print(medians_gen)
    print(qts_upper_gen)
    print(qts_lower_gen)

    ptr.figure(figsize=(15,10), dpi=100)

    axes_a = ptr.subplot(111, title='Index case V0 diffusion over generations')
    axes_a.set_ylabel("Initial Viral Load")
    axes_a.set(xlabel="Generation")

    labs_gen = [str(int(bins_gen[h-1])) for h in range(1,len(bins_gen))]

    # for sim in range(len(medians_gen)):
        # axes_a.plot(labs_gen,[h[1] for h in coherence[sim]],color='red')
        # axes_a.plot(labs_gen,[h[0] for h in coherence[sim]],color='blue')
        # axes_a.plot(labs_gen,[h[2] for h in coherence[sim]],color='green')
        # axes_a.plot(labs_gen,[h[3] for h in coherence[sim]],color='purple')
    axes_a.plot(labs_gen[:len(medians_gen)], medians_gen)


    ptr.savefig(fname='median_trajectory_' + p[0] + '_' + p[1] + '_' + p[3])

    # axes_a.legend()

    ptr.tight_layout()

    # ptr.show()

    ptr.close("all")

    # DIFFUSION

    ptr.figure(figsize=(15,10), dpi=100)

    axes_a = ptr.subplot(211, title='Index case V0 diffusion over generations')
    axes_a.set_ylabel("Initial Viral Load")
    axes_a.set(xlabel="Generation")

    axes_b = ptr.subplot(212, title='Index case V0 diffusion over time')
    axes_b.set_ylabel("Initial Viral Load")
    axes_b.set(xlabel="Time (days)")

    # TODO: plot upper and lower bounds
    # time_x = np.linspace(0,100*steps_per_day)
    # gen_x = np.linspace(0,20,num=21)
    # upper = [ind_init]

    # cur = ind_init
    # for f in gen_x[1:]:
    #     peak = get_peak_viral_load(cur)
    #     new_init = 0.01*50000 * np.power(peak,10) / (np.power(peak,10) + np.power(50000/2,10))
    #     upper.append(new_init)
    #     cur = new_init
    
    # axes_a.plot(gen_x,upper)

    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    labs_gen = [str(int(bins_gen[h-1])) for h in range(1,len(bins_gen))]
    labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]

    axes_a.boxplot(binned_gen[1:], labels=labs_gen, notch=True)#, positions=np.linspace(0,19,20))
    axes_b.boxplot(binned_time[1:], labels=labs_time, notch=True)
    ptr.tight_layout()

    ptr.savefig(fname='subtree_diffusion_bw_' + p[0] + '_' + p[1] + '_' + p[3])
    # ptr.show()

    ptr.close("all")

    ptr.figure(figsize=(15,10), dpi=100)

    axes_a = ptr.subplot(211, title='Index case V0 diffusion over generations')
    axes_a.set_ylabel("Initial Viral Load")
    axes_a.set(xlabel="Generation")

    axes_b = ptr.subplot(212, title='Index case V0 diffusion over time')
    axes_b.set_ylabel("Initial Viral Load")
    axes_b.set(xlabel="Time (days)")

    # TODO: plot upper and lower bounds
    # time_x = np.linspace(0,100*steps_per_day)
    # gen_x = np.linspace(0,20,num=21)
    # upper = [ind_init]

    # cur = ind_init
    # for f in gen_x[1:]:
    #     peak = get_peak_viral_load(cur)
    #     new_init = 0.01*50000 * np.power(peak,10) / (np.power(peak,10) + np.power(50000/2,10))
    #     upper.append(new_init)
    #     cur = new_init
    
    # axes_a.plot(gen_x,upper)

    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    labs_gen = [str(int(bins_gen[h-1])) for h in range(1,len(bins_gen))]
    labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]

    cols = {}
    for v in range(len(labs_time)):
        cols[str(v)] = labs_time[v]

    # print(cols)


    # axes_a.boxplot(binned_gen[1:], labels=labs_gen)#, positions=np.linspace(0,19,20))
    # axes_b.boxplot(binned_time[1:], labels=labs_time)

    gen_data = pd.DataFrame(data = binned_gen[1:])
    gen_data = gen_data.T

    time_data = pd.DataFrame(data = binned_time[1:])
    time_data = time_data.T
    time_data = time_data.rename(columns=cols)

    sns.stripplot(data=gen_data, ax=axes_a, jitter=0.4, size=3)
    sns.despine()

    sns.stripplot(data=time_data, ax=axes_b, jitter=0.4, size=3)
    sns.despine()

    ptr.tight_layout()

    axes_b.set_xticklabels(labs_time)
    axes_a.set_xticklabels(labs_gen)

    ptr.savefig(fname='subtree_diffusion_sp_' + p[0] + '_' + p[1] + '_' + p[3])
    # ptr.show()

    ptr.close("all")

with Pool(5) as p:
    p.map(diffusion,work)