import time
import random
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.pyplot as ptr
import enum
from scipy.integrate import odeint
from scipy.stats import pearsonr, spearmanr
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
        self.r0 = 0
        self.ptrans = 1.0 / steps_per_day
        self.viral_load_max = 25000
        self.inoc_max = 50000
        self.inoc_max_sig = np.power(self.inoc_max/2,self.sig_init)
        self.infected = 0
        self.viral_load_tree = dict()

        if G == 'constant':
            self.v_threshold = v_threshold
        elif G == 'random':
            self.v_threshold = v_threshold
        elif G == 'linear':
            self.v_threshold = 1 / self.lin_init
        elif G == 'log':
            self.v_threshold = np.exp(1 / (self.log_init * self.inoc_max))
        elif G == 'sigmoid':
            beta = self.lin_init * self.inoc_max
            self.v_threshold = np.power(self.inoc_max_sig/(beta * (1 - (1/beta))),1/self.sig_init) - self.delay
        
        print(self.lin_init,self.sig_init,self.v_threshold)

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


    def contact(self):
        """Find close contacts and infect"""
        
        if self.state is MSState.INFECTED:
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

ind_init = 450

ptr.close("all")

data_whole = pd.DataFrame()

# SUBTREE ANALYSIS

z = [('sigmoid', 0, 'blue', '', 10)]
no_sims = 5

ptr.figure(figsize=(10,10), dpi=100)

trees = []

for j, d, col, per, s in z:

    data_whole = pd.DataFrame()
    r0s_inner = []
    for i in range(no_sims):
        print("Sim: " + str(i))
        while(True):
            model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G=j, con_init=ind_init, sig_init=s, delay=d, lin_init=0.01, log_init=400)
            for i in range(steps):
                model.step()
                if model.infected == 0:
                    model.step()
                    break
            if model.r0 != 0:
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

        print(max(data['Removed']))

    ptr.close("all")

    ax = plt.subplot(111)
    tree_display = trees[0]
    ax.set_title('Transmission Tree')
    ax.set_xticks([])

    # for i, row in data_whole.iterrows():
    #     if not len(row['Viral Load Tree']) == 0:
    #         for i in row['Viral Load Tree']:
    #             if tree.number_of_nodes() < 10000:
    #                 if tree.has_node(i[0]):
    #                     tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2), time=round(row['Viral Load Tree'][i][3],2), gen=tree.nodes[i[0]]['gen'] + 1)
    #                     tree.add_edge(i[0],i[1], rec_init=round(row['Viral Load Tree'][i][1],2))
    #                 else:
    #                     tree.add_node(i[0], init_v=round(row['Viral Load Tree'][i][0],2), time=0, gen=0)
    #                     tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2), time=round(row['Viral Load Tree'][i][3],2), gen=tree.nodes[i[0]]['gen'] + 1)
    #                     tree.add_edge(i[0],i[1],rec_init=round(row['Viral Load Tree'][i][1],2)) 

    pos=nx.drawing.nx_agraph.graphviz_layout(tree_display, prog='dot')
    max_time = max([tree_display.nodes[i]['time'] for i in tree_display.nodes()])

    for node in tree_display.nodes():
        pos[node] = (pos[node][0],-1*tree_display.nodes[node]['time'])

    nodes = nx.draw_networkx_nodes(tree_display, ax=ax, pos=pos, node_size=50,
                    node_color=list(nx.get_node_attributes(tree_display, 'init_v').values()),
                    cmap=ptr.cm.viridis, vmin=0)
    edges = nx.draw_networkx_edges(tree_display, ax=ax, pos=pos, node_size=50)
    edge_labels_dict = nx.get_edge_attributes(tree_display,'rec_init')
    edge_labels = nx.draw_networkx_edge_labels(tree_display, ax=ax, pos=pos, font_size=7, edge_labels=edge_labels_dict)
    
    ptr.colorbar(nodes)

    yticks = range(0,-1*max_time,-100)
    labelsy = [ round(-1*i/steps_per_day) for i in yticks]
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(labelsy)

    ax.set_ylabel('Days')

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ptr.tight_layout()

    ptr.savefig(fname='trans_tree')
    ptr.show()

    binned_time = []
    bins_time = np.linspace(0,days,num=11)
    for g in bins_time:
        binned_time.append([])

    binned_gen = []
    bins_gen = np.linspace(0,20,num=21)
    for g in bins_gen:
        binned_gen.append([])


    for tree in trees:
        for node in tree.nodes():
            sub = nx.dfs_tree(tree,node)
            bin_no_time = np.digitize(tree.nodes[node]['time']/steps_per_day,bins_time)
            bin_no_gen = np.digitize(tree.nodes[node]['gen'],bins_gen)

            binned_time[bin_no_time].append(tree.nodes[node]['init_v'])
            binned_gen[bin_no_gen].append(tree.nodes[node]['init_v'])

    # DIFFUSION

    ptr.figure(figsize=(10,10), dpi=100)

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

    labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]

    axes_a.boxplot(binned_gen[1:], labels=labs_gen)#, positions=np.linspace(0,19,20))
    axes_b.boxplot(binned_time[1:], labels=labs_time)
    ptr.tight_layout()

    ptr.savefig(fname='subtree_diffusion')
    ptr.show()

    ptr.close("all")