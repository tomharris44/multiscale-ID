import time
import random
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.pyplot as ptr
import matplotlib as mpl
import enum
from scipy.integrate import odeint
from scipy.stats import pearsonr, spearmanr, poisson
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
                 sig_init=1, delay=0, output_tree=None, r=5.5):

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
        self.r = r

        if G == 'constant':
            self.v_threshold = v_threshold
        elif G == 'random':
            self.v_threshold = 1 / self.lin_init
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
        r = self.model.r
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
            foo = self.model.lin_init * self.model.inoc_max * np.power(donor + self.model.delay,self.model.sig_init) / (np.power(donor + self.model.delay,self.model.sig_init) + self.model.inoc_max_sig)
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
        r = self.model.r
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
            # init_v = self.model.lin_init * donor
            foo = self.model.lin_init * donor
            init_v = np.mean(poisson.rvs(foo, size=100))
        elif G == 'log':
            init_v = self.model.log_init * np.log(donor)
        elif G == 'sigmoid':
            # init_v = self.model.lin_init * self.model.inoc_max * np.power(donor + self.model.delay,self.model.sig_init) / (np.power(donor + self.model.delay,self.model.sig_init) + self.model.inoc_max_sig)
            foo = self.model.lin_init * self.model.inoc_max * np.power(donor + self.model.delay,self.model.sig_init) / (np.power(donor + self.model.delay,self.model.sig_init) + self.model.inoc_max_sig)
            init_v = np.mean(poisson.rvs(foo, size=100))

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


def get_all_lengths_gen(G,root):

    paths = []

    for node in G:
        if G.out_degree(node)==0:
            paths.append(nx.shortest_path_length(G, source=root, target=node))
    
    return paths

def get_avg_length_gen(G,root):
    paths = []

    for node in G:
        if G.out_degree(node)==0:
            paths.append(nx.shortest_path_length(G, source=root, target=node))

    return np.mean(paths)

def get_all_lengths_time(G,root):
    paths = []

    for node in G:
        if G.out_degree(node)==0:
            paths.append((G.nodes[node]['time'] - G.nodes[root]['time']) / steps_per_day)

    return paths

def get_avg_length_time(G,root,tree):
    paths = []

    for node in G:
        if G.out_degree(node)==0:
            paths.append((tree.nodes[node]['time'] - tree.nodes[root]['time']) / steps_per_day)

    return np.mean(paths)



days = 100
steps = steps_per_day * days
pop = 10000

ptr.close("all")

data_whole = pd.DataFrame()

# SUBTREE ANALYSIS

# z = [('sigmoid', 0, 'blue', '', 10), ('linear', 0, 'blue', '', 10), ('random', 0, 'blue', '', 10)]
# z = [('sigmoid', 0, 'blue', '', 10, 5.45), ('sigmoid', 0, 'blue', '', 10, 5.5), ('sigmoid', 0, 'blue', '', 10, 5.7), ('sigmoid', 0, 'blue', '', 10, 5.9)]
z = [('sigmoid', 0, 'blue', '', 10, 5.7), ('sigmoid', 0, 'blue', '', 10, 5.9)]

no_sims = 1

ptr.figure(figsize=(10,10), dpi=100)

for j, d, col, per, s, r in z:

    r_print = str(int(10 * (r-5)))

    data_whole = pd.DataFrame()
    r0s_inner = []
    for i in range(no_sims):
        print("Sim: " + str(i))
        while(True):
            model = InfectionMultiscaleModel(pop, steps_per_day=steps_per_day, v_threshold=10000, G=j, con_init=450, sig_init=s, delay=d, lin_init=0.01, log_init=400, r=r)
            for i in range(steps):
                model.step()
                if model.infected == 0:
                    model.step()
                    break
            if model.r0 != 0:
                break

        data = get_column_data(model)
        if i==0:
            data_whole = data
        else:
            data_whole = data_whole.append(data)

        print(max(data_whole['Removed']))

        # data_hist = data_whole['Viral Load Tree']
        # data_hist = data_hist[data_hist.astype(bool)]

        # length = 5
        # width = 5

        # size = length * width

        # v_max = 500
        # j_add = v_max / size
        # i_add = width * j_add
        
        # for i in range(5):
        #     for h in range(5):
        #         lower = i_add*i + j_add*h
        #         upper = i_add*i + j_add*(h+1)
                
        #         vals = [(list(k.values())[0][0],list(k.values())[0][1]) for k in data_hist if (list(k.values())[0][0] > lower) & (list(k.values())[0][0] <= upper)]

        #         dons = [i[0] for i in vals]
        #         recs = [i[1] for i in vals]

        #         ax = ptr.subplot2grid((5,5), (i,h), title='Donor V0 range: ' + str(int(lower)) + ' - ' + str(int(upper)), xlabel='Recipient V0', ylim=(0,50))
        #         ax.hist(recs,bins=20,range=(0,500))


    # ptr.tight_layout()

    # ptr.savefig(fname='hist')
    # ptr.show()

    # ptr.close("all")

    # ptr.figure(figsize=(15,10), dpi=100)

    # axes_s = ptr.subplot(221, xlim=(0,100))

    # axes_i = ptr.subplot(223, yscale='log', title='Initial Viral Loads', xlim=(0,100))
    # axes_i.set_ylabel("Initial Viral Loads")
    # axes_i.set(xlabel="Days")

    # axes_r = ptr.subplot(222, xscale='log', yscale='log', title='Donor vs Recipient Initial Viral Loads')
    # axes_r.set_ylabel("Recipient Initial Viral Loads")
    # axes_r.set_xlabel("Donor Initial Viral Loads")

    # ax = plt.subplot(224)
    # tree = nx.DiGraph()
    # ax.set_title('Transmission Tree')
    # ax.set_xticks([])
    # ax.set_yticks([])

    # r0s = []

    # data_whole.groupby(level=0).mean()[0:days].plot(ax=axes_s, xticks=range(0,days,10), kind='bar', 
    #                                                 title='Incidence (N=2500)', x='Incidence Days', y='Incidence Sum', 
    #                                                 label=str(j) + ': zeta=' + str(per) + str(s), logy=True)

    # for i, row in data_whole.iterrows():
    #     if not len(row['Viral Load Tree']) == 0:
    #         axes_i.scatter(np.repeat(row['Days'],len(row['Viral Load Tree'].values())),[i[1] for i in row['Viral Load Tree'].values()], c=col, label=str(j) + ': zeta=' + str(per) + str(s), alpha=0.3)
    #         axes_r.scatter([i[0] for i in row['Viral Load Tree'].values()], [i[1] for i in row['Viral Load Tree'].values()], c=col, label=str(j) + ': zeta=' + str(per) + str(s), alpha=0.3)
            
    #         for i in row['Viral Load Tree']:
    #             if tree.number_of_nodes() < 30:
    #                 if tree.has_node(i[0]):
    #                     tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2))
    #                     tree.add_edge(i[0],i[1], rec_init=round(row['Viral Load Tree'][i][1],2))
    #                 else:
    #                     tree.add_node(i[0], init_v=round(row['Viral Load Tree'][i][0],2))
    #                     tree.add_node(i[1], init_v=round(row['Viral Load Tree'][i][1],2))
    #                     tree.add_edge(i[0],i[1], rec_init=round(row['Viral Load Tree'][i][1],2)) 

    # r0s.append(np.mean(r0s_inner))
    
    # axes_s.set_ylabel("Incidence")
    # axes_s.set(xlabel="Days")

    # handles, labels = axes_r.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # axes_r.legend(by_label.values(), by_label.keys())

    # handles, labels = axes_i.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # axes_i.legend(by_label.values(), by_label.keys())

    # pos=nx.nx_agraph.graphviz_layout(tree, prog='dot')
    # nodes = nx.draw_networkx_nodes(tree, ax=ax, pos=pos, node_size=50,
    #                 node_color=list(nx.get_node_attributes(tree, 'init_v').values()),
    #                 cmap=ptr.cm.viridis, vmin=0)

    # edges = nx.draw_networkx_edges(tree, ax=ax, pos=pos, node_size=50)
    # edge_labels_dict = nx.get_edge_attributes(tree,'rec_init')
    # edge_labels = nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos, font_size=7, edge_labels=edge_labels_dict)
    
    # ptr.colorbar(nodes)

    # ptr.tight_layout()

    # ptr.savefig(fname='outbreak')
    # ptr.show()

    ptr.close("all")

    ax = plt.subplot(111)
    tree = nx.DiGraph()
    ax.set_title('Transmission Tree')
    ax.set_xticks([])

    for i, row in data_whole.iterrows():
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

    pos=nx.drawing.nx_agraph.graphviz_layout(tree, prog='dot')
    max_time = max([tree.nodes[i]['time'] for i in tree.nodes()])

    for node in tree.nodes():
        pos[node] = (pos[node][0],-1*tree.nodes[node]['time'])

    nodes = nx.draw_networkx_nodes(tree, ax=ax, pos=pos, node_size=50,
                    node_color=list(nx.get_node_attributes(tree, 'init_v').values()),
                    cmap=ptr.cm.viridis, vmin=0)
    edges = nx.draw_networkx_edges(tree, ax=ax, pos=pos, node_size=50)
    edge_labels_dict = nx.get_edge_attributes(tree,'rec_init')
    edge_labels = nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos, font_size=7, edge_labels=edge_labels_dict)
    
    ptr.colorbar(nodes)

    yticks = range(0,-1*max_time,-100)
    labelsy = [ round(-1*i/steps_per_day) for i in yticks]
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(labelsy)

    ax.set_ylabel('Days')



    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ptr.tight_layout()

    ptr.savefig(fname='trans_tree_' + r_print)
    # ptr.show()

    init_vs = []
    times = []
    generations = []
    sizes = []
    lengths = []
    sec_cases = []
    GFs = []
    SLs = []

    binned_init_vs = []
    bins_vs = np.linspace(0,500,num=11)
    for g in bins_vs:
        binned_init_vs.append([])

    binned_time = []
    bins_time = np.linspace(0,days,num=11)
    for g in bins_time:
        binned_time.append([])

    binned_gen = []
    bins_gen = np.linspace(0,25,num=11)
    for g in bins_gen:
        binned_gen.append([])


    for node in tree.nodes():
        # if tree.nodes[node]['time'] > 10*steps_per_day and tree.nodes[node]['time'] < 55*steps_per_day:
        sub = nx.dfs_tree(tree,node)
        bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)
        if bin_no_vs >= len(bins_vs):
            bin_no_vs = len(bins_vs) - 1
        bin_no_time = np.digitize(tree.nodes[node]['time']/steps_per_day,bins_time)
        bin_no_gen = np.digitize(tree.nodes[node]['gen'],bins_gen)

        # raw data sets
        init_vs.append(tree.nodes[node]['init_v'])
        times.append(tree.nodes[node]['time']/steps_per_day)
        generations.append(tree.nodes[node]['gen'])

        sizes.append(sub.size())
        lengths.append(get_avg_length_gen(sub,node))
        sec_cases.append(len(list(tree.successors(node))))
        GFs.append(np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'])
        SLs.append(sub.size() / get_avg_length_gen(sub,node))

        #binned data sets
        binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'], sub.size() / get_avg_length_gen(sub,node), get_avg_length_time(sub,node,tree)])
        binned_time[bin_no_time].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'], sub.size() / get_avg_length_gen(sub,node), get_avg_length_time(sub,node,tree)])
        binned_gen[bin_no_gen].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'], sub.size() / get_avg_length_gen(sub,node), get_avg_length_time(sub,node,tree)])
        
        

    ptr.close("all")

    nan_fil_inits = [[x[4] for x in y if not np.isnan(x[4])] for y in binned_init_vs[1:]]
    nan_fil_time = [[x[4] for x in y if not np.isnan(x[4])] for y in binned_time[1:]]
    nan_fil_gen = [[x[4] for x in y if not np.isnan(x[4])] for y in binned_gen[1:]]

    # SIZE

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, yscale='log', title='Effect of V0 of root node on subtree size', xlim=(0,500), ylim=(0.1,pop))
    # axes_a.set_ylabel("Subtree size")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, yscale='log', title='Effect of time on subtree size', ylim=(0.1,pop))
    # axes_b.set_ylabel("Subtree size")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, yscale='log', title='Effect of generation on subtree size', ylim=(0.1,pop))
    # axes_c.set_ylabel("Subtree size")
    # axes_c.set(xlabel="Generation")

    # axes_a.scatter(init_vs,sizes, c=col, alpha=0.3)
    # axes_b.scatter(times,sizes, c=col, alpha=0.3)
    # axes_c.scatter(generations,sizes, c=col, alpha=0.3)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")

    # SIZE B+W NOTCH

    ptr.figure(figsize=(7,10), dpi=100)

    axes_a = ptr.subplot(311, yscale='log', title='Effect of V0 of root node on subtree size')
    axes_a.set_ylabel("Subtree size")
    axes_a.set(xlabel="Initial Viral Load")

    axes_b = ptr.subplot(312, yscale='log', title='Effect of time on subtree size')
    axes_b.set_ylabel("Subtree size")
    axes_b.set(xlabel="Days")

    axes_c = ptr.subplot(313, yscale='log', title='Effect of generation on subtree size')
    axes_c.set_ylabel("Subtree size")
    axes_c.set(xlabel="Generation")

    labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    labs_vs[len(labs_vs) - 1] = '450+'
    labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    axes_a.boxplot([[h[0] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
    axes_b.boxplot([[h[0] for h in g] for g in binned_time[1:]], labels=labs_time, notch=True)
    axes_c.boxplot([[h[0] for h in g] for g in binned_gen[1:]], labels=labs_gen, notch=True)

    ptr.tight_layout()

    ptr.savefig(fname='subtree_analysis_size' + r_print)
    # ptr.show()

    ptr.close("all")

    # SIZE B+W

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, yscale='log', title='Effect of V0 of root node on subtree size')
    # axes_a.set_ylabel("Subtree size")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, yscale='log', title='Effect of time on subtree size')
    # axes_b.set_ylabel("Subtree size")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, yscale='log', title='Effect of generation on subtree size')
    # axes_c.set_ylabel("Subtree size")
    # axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    # axes_a.boxplot([[h[0] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    # axes_b.boxplot([[h[0] for h in g] for g in binned_time[1:]], labels=labs_time)
    # axes_c.boxplot([[h[0] for h in g] for g in binned_gen[1:]], labels=labs_gen)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")

    # LENGTH

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree path length', xlim=(0,500))
    # axes_a.set_ylabel("Path length")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, title='Effect of time on subtree path length')
    # axes_b.set_ylabel("Path length")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, title='Effect of generation on subtree path length')
    # axes_c.set_ylabel("Path length")
    # axes_c.set(xlabel="Generation")

    # axes_a.scatter(init_vs,lengths, c=col, alpha=0.3)
    # axes_b.scatter(times,lengths, c=col, alpha=0.3)
    # axes_c.scatter(generations,lengths, c=col, alpha=0.3)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")

    # LENGTH B+W NOTCH GEN

    ptr.figure(figsize=(7,10), dpi=100)

    axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree path length')
    axes_a.set_ylabel("Path length")
    axes_a.set(xlabel="Initial Viral Load")

    axes_b = ptr.subplot(312, title='Effect of time on subtree path length')
    axes_b.set_ylabel("Path length")
    axes_b.set(xlabel="Days")

    axes_c = ptr.subplot(313, title='Effect of generation on subtree path length')
    axes_c.set_ylabel("Path length")
    axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    axes_a.boxplot([[h[1] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
    axes_b.boxplot([[h[1] for h in g] for g in binned_time[1:]], labels=labs_time, notch=True)
    axes_c.boxplot([[h[1] for h in g] for g in binned_gen[1:]], labels=labs_gen, notch=True)

    ptr.tight_layout()

    ptr.savefig(fname='subtree_analysis_len_gen' + r_print)
    # ptr.show()

    ptr.close("all")


    # LENGTH B+W NOTCH TIME

    ptr.figure(figsize=(7,10), dpi=100)

    axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree path length')
    axes_a.set_ylabel("Path length")
    axes_a.set(xlabel="Initial Viral Load")

    axes_b = ptr.subplot(312, title='Effect of time on subtree path length')
    axes_b.set_ylabel("Path length")
    axes_b.set(xlabel="Days")

    axes_c = ptr.subplot(313, title='Effect of generation on subtree path length')
    axes_c.set_ylabel("Path length")
    axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    axes_a.boxplot([[h[5] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
    axes_b.boxplot([[h[5] for h in g] for g in binned_time[1:]], labels=labs_time, notch=True)
    axes_c.boxplot([[h[5] for h in g] for g in binned_gen[1:]], labels=labs_gen, notch=True)

    ptr.tight_layout()

    ptr.savefig(fname='subtree_analysis_len_time_' + r_print)
    # ptr.show()

    ptr.close("all")

    # LENGTH B+W GEN

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree path length')
    # axes_a.set_ylabel("Path length")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, title='Effect of time on subtree path length')
    # axes_b.set_ylabel("Path length")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, title='Effect of generation on subtree path length')
    # axes_c.set_ylabel("Path length")
    # axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    # axes_a.boxplot([[h[1] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    # axes_b.boxplot([[h[1] for h in g] for g in binned_time[1:]], labels=labs_time)
    # axes_c.boxplot([[h[1] for h in g] for g in binned_gen[1:]], labels=labs_gen)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")


    # LENGTH B+W TIME

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree path length')
    # axes_a.set_ylabel("Path length")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, title='Effect of time on subtree path length')
    # axes_b.set_ylabel("Path length")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, title='Effect of generation on subtree path length')
    # axes_c.set_ylabel("Path length")
    # axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    # axes_a.boxplot([[h[5] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    # axes_b.boxplot([[h[5] for h in g] for g in binned_time[1:]], labels=labs_time)
    # axes_c.boxplot([[h[5] for h in g] for g in binned_gen[1:]], labels=labs_gen)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")

    

    # SECONDARY CASE DISTRIBUTION

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree secondary cases', xlim=(0,500))
    # axes_a.set_ylabel("Secondary Cases")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, title='Effect of time on subtree secondary cases')
    # axes_b.set_ylabel("Secondary Cases")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, title='Effect of generation on subtree secondary cases')
    # axes_c.set_ylabel("Secondary Cases")
    # axes_c.set(xlabel="Generation")

    # axes_a.scatter(init_vs,sec_cases, c=col, alpha=0.3)
    # axes_b.scatter(times,sec_cases, c=col, alpha=0.3)
    # axes_c.scatter(generations,sec_cases, c=col, alpha=0.3)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")

    # SECONDARY CASE DISTRIBUTION B+W

    ptr.figure(figsize=(7,10), dpi=100)

    axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree secondary cases')
    axes_a.set_ylabel("Secondary Cases")
    axes_a.set(xlabel="Initial Viral Load")

    axes_b = ptr.subplot(312, title='Effect of time on subtree secondary cases')
    axes_b.set_ylabel("Secondary Cases")
    axes_b.set(xlabel="Days")

    axes_c = ptr.subplot(313, title='Effect of generation on subtree secondary cases')
    axes_c.set_ylabel("Secondary Cases")
    axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    # axes_a.boxplot([[h for h in g] for g in nan_fil_inits], labels=labs_vs, notch=True)
    # axes_b.boxplot([[h for h in g] for g in nan_fil_time], labels=labs_time, notch=True)
    # axes_c.boxplot([[h for h in g] for g in nan_fil_gen], labels=labs_gen, notch=True)

    axes_a.boxplot([[h[2] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
    axes_b.boxplot([[h[2] for h in g] for g in binned_time[1:]], labels=labs_time, notch=True)
    axes_c.boxplot([[h[2] for h in g] for g in binned_gen[1:]], labels=labs_gen, notch=True)

    ptr.tight_layout()

    ptr.savefig(fname='subtree_analysis_sec_cases' + r_print)
    # ptr.show()

    ptr.close("all")


    # SECONDARY CASE DISTRIBUTION B+W

    # ptr.figure(figsize=(7,10), dpi=100)

    # axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree secondary cases')
    # axes_a.set_ylabel("Secondary Cases")
    # axes_a.set(xlabel="Initial Viral Load")

    # axes_b = ptr.subplot(312, title='Effect of time on subtree secondary cases')
    # axes_b.set_ylabel("Secondary Cases")
    # axes_b.set(xlabel="Days")

    # axes_c = ptr.subplot(313, title='Effect of generation on subtree secondary cases')
    # axes_c.set_ylabel("Secondary Cases")
    # axes_c.set(xlabel="Generation")

    # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    # labs_time = [str(int(bins_time[h-1])) + '-' + str(int(bins_time[h])) for h in range(1,len(bins_time))]
    # labs_gen = [str(int(bins_gen[h-1])) + '-' + str(int(bins_gen[h])) for h in range(1,len(bins_gen))]
    

    # # axes_a.boxplot([[h for h in g] for g in nan_fil_inits], labels=labs_vs, notch=True)
    # # axes_b.boxplot([[h for h in g] for g in nan_fil_time], labels=labs_time, notch=True)
    # # axes_c.boxplot([[h for h in g] for g in nan_fil_gen], labels=labs_gen, notch=True)

    # axes_a.boxplot([[h[2] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    # axes_b.boxplot([[h[2] for h in g] for g in binned_time[1:]], labels=labs_time)
    # axes_c.boxplot([[h[2] for h in g] for g in binned_gen[1:]], labels=labs_gen)

    # ptr.tight_layout()

    # ptr.savefig(fname='subtree_analysis')
    # ptr.show()

    # ptr.close("all")

    # GROWTH FACTOR

    ptr.figure(figsize=(7,10), dpi=100)

    axes_a = ptr.subplot(311, title='Effect of V0 of root node on subtree growth factor', xlim=(0,500))
    axes_a.set_ylabel("Growth factor")
    axes_a.set(xlabel="Initial Viral Load")

    axes_b = ptr.subplot(312, title='Effect of time on subtree growth factor')
    axes_b.set_ylabel("Growth factor")
    axes_b.set(xlabel="Days")

    axes_c = ptr.subplot(313, title='Effect of generation on subtree growth factor')
    axes_c.set_ylabel("Growth factor")
    axes_c.set(xlabel="Generation")

    axes_a.scatter(init_vs,GFs, c=col, alpha=0.3)
    axes_b.scatter(times,GFs, c=col, alpha=0.3)
    axes_c.scatter(generations,GFs, c=col, alpha=0.3)

    ptr.tight_layout()

    ptr.savefig(fname='subtree_analysis_GF_' + r_print)
    # ptr.show()

    ptr.close("all")

    # ptr.figure(figsize=(15,10), dpi=100)

    # # while(True):
    # #     rand_node = random.choice(list(tree.nodes()))
    # #     sub = nx.dfs_tree(tree,rand_node)
    # #     if sub.size() > 500:
    # #         print(rand_node,sub.size())
    # #         break

    # # lengths_dist = get_all_lengths(sub,rand_node)
    # lengths_dist = get_all_lengths_time(tree,1)
    
    # ax = ptr.subplot(111, title='Length of path to leaf nodes distribution for larger tree (Size = ' + str(tree.size()) + ')')

    # ax.hist(lengths_dist,bins=20,range=(0,100))

    # ax.set_ylabel("Frequency")
    # ax.set_xlabel("Path length")

    # ptr.tight_layout()

    # ptr.savefig(fname='hist')
    # ptr.show()

    # ptr.close("all")

    # V0 COLOUR PLOT

    ptr.figure(figsize=(15,10), dpi=100)

    axes_a = ptr.subplot(111, title='Effect of V0 on size vs. average path length', xscale='log')
    axes_a.set_ylabel("Average path length")
    axes_a.set(xlabel="Size")

    cmap = ptr.cm.viridis
    norm = mpl.colors.Normalize(vmin=1, vmax=500)

    axes_a.scatter(sizes,lengths, cmap=cmap, c=init_vs, alpha=0.3)

    ptr.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    

    ptr.tight_layout()

    ptr.savefig(fname='subtree_colour_analysis_CP')
    ptr.show()

    ptr.close("all")
    

    ptr.figure(figsize=(15,10), dpi=100)

    length = 3
    width = 4

    size = length * width

    time_max = 60
    j_add = time_max / size
    i_add = width * j_add

    labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]
    labs_vs[-1] = '450+'
    
    for i in range(3):
        for h in range(4):
            init_vs = []
            sizes = []
            lengths = []
            sec_cases = []
            lower = i_add*i + j_add*h
            upper = i_add*i + j_add*(h+1)

            binned_init_vs = []
            bins_vs = np.linspace(0,500,num=11)
            for g in bins_vs:
                binned_init_vs.append([])
            
            for node in tree.nodes():
                if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
                    sub = nx.dfs_tree(tree,node)
                    bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)
                    if bin_no_vs >= len(bins_vs):
                        bin_no_vs = len(bins_vs) - 1

                    init_vs.append(tree.nodes[node]['init_v'])
                    sizes.append(sub.size())
                    lengths.append(get_avg_length_gen(sub,node))
                    sec_cases.append(len(list(tree.successors(node))))
                    binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v']])
        
            if len(init_vs) > 2:
                pear_r = pearsonr(init_vs,sizes)
                # pear_r = pearsonr(init_vs,lengths)
                # pear_r = pearsonr(init_vs,sec_cases)
                pear_r = round(pear_r[0],3)

                spear_r = spearmanr(init_vs,sizes)
                # pear_r = pearsonr(init_vs,lengths)
                # pear_r = pearsonr(init_vs,sec_cases)
                spear_r_p = round(spear_r[1],3)
                spear_r = round(spear_r[0],3)
            else:
                pear_r = 'NaN'
                spear_r = 'NaN'
                spear_r_p = 'NaN'

            # labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]


            ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Subtree size', xlabel='Root node V0',ylim=(0.1,1000), yscale='log')
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Mean path length', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Secondary cases', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
            # ax.scatter(init_vs,sizes, c=col, alpha=0.3)
            ax.boxplot([[h[0] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
            ptr.xticks(fontsize=8)


    ptr.tight_layout()

    ptr.savefig(fname='hist_size_' + r_print)
    # ptr.show()

    ptr.close("all")

    ptr.figure(figsize=(15,10), dpi=100)

    length = 3
    width = 4

    size = length * width

    time_max = 60
    j_add = time_max / size
    i_add = width * j_add
    
    for i in range(3):
        for h in range(4):
            init_vs = []
            sizes = []
            lengths = []
            sec_cases = []
            lower = i_add*i + j_add*h
            upper = i_add*i + j_add*(h+1)

            binned_init_vs = []
            bins_vs = np.linspace(0,500,num=11)
            for g in bins_vs:
                binned_init_vs.append([])
            
            for node in tree.nodes():
                if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
                    sub = nx.dfs_tree(tree,node)
                    bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)
                    if bin_no_vs >= len(bins_vs):
                        bin_no_vs = len(bins_vs) - 1

                    init_vs.append(tree.nodes[node]['init_v'])
                    sizes.append(sub.size())
                    lengths.append(get_avg_length_gen(sub,node))
                    sec_cases.append(len(list(tree.successors(node))))
                    binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v']])
        
            if len(init_vs) > 2:
                # pear_r = pearsonr(init_vs,sizes)
                pear_r = pearsonr(init_vs,lengths)
                # pear_r = pearsonr(init_vs,sec_cases)
                pear_r = round(pear_r[0],3)

                spear_r = spearmanr(init_vs,lengths)
                spear_r_p = round(spear_r[1],3)
                spear_r = round(spear_r[0],3)
            else:
                pear_r = 'NaN'
                spear_r = 'NaN'
                spear_r_p = 'NaN'
            
            # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
            # labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)), ylabel='Subtree size', xlabel='Root node V0', xlim=(0,500),ylim=(0.1,1000), yscale='log')
            ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Mean path length', xlabel='Root node V0',ylim=(-1,15))
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Secondary cases', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
            # ax.scatter(init_vs,lengths, c=col, alpha=0.3)
            ax.boxplot([[h[1] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
            ptr.xticks(fontsize=8)


    ptr.tight_layout()

    ptr.savefig(fname='hist_len_' + r_print)
    # ptr.show()

    ptr.close("all")

    ptr.figure(figsize=(15,10), dpi=100)

    length = 3
    width = 4

    size = length * width

    time_max = 60
    j_add = time_max / size
    i_add = width * j_add
    
    for i in range(3):
        for h in range(4):
            init_vs = []
            sizes = []
            lengths = []
            sec_cases = []
            lower = i_add*i + j_add*h
            upper = i_add*i + j_add*(h+1)

            binned_init_vs = []
            bins_vs = np.linspace(0,500,num=11)
            for g in bins_vs:
                binned_init_vs.append([])
            
            for node in tree.nodes():
                if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
                    sub = nx.dfs_tree(tree,node)
                    bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)
                    if bin_no_vs >= len(bins_vs):
                        bin_no_vs = len(bins_vs) - 1

                    init_vs.append(tree.nodes[node]['init_v'])
                    sizes.append(sub.size())
                    lengths.append(get_avg_length_gen(sub,node))
                    sec_cases.append(len(list(tree.successors(node))))
                    binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'], sub.size() / get_avg_length_gen(sub,node)])
        
            if len(init_vs) > 2:
                # pear_r = pearsonr(init_vs,sizes)
                # pear_r = pearsonr(init_vs,lengths)
                pear_r = pearsonr(init_vs,sec_cases)
                pear_r = round(pear_r[0],3)

                spear_r = spearmanr(init_vs,sec_cases)
                spear_r_p = round(spear_r[1],3)
                spear_r = round(spear_r[0],3)
            else:
                pear_r = 'NaN'
                spear_r = 'NaN'
                spear_r_p = 'NaN'

            # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
            # labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)), ylabel='Subtree size', xlabel='Root node V0', xlim=(0,500),ylim=(0.1,1000), yscale='log')
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Mean path length', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
            ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Secondary cases', xlabel='Root node V0',ylim=(-1,15))
            # ax.scatter(init_vs,sec_cases, c=col, alpha=0.3)
            ax.boxplot([[h[2] for h in g] for g in binned_init_vs[1:]], labels=labs_vs, notch=True)
            ptr.xticks(fontsize=8)


    ptr.tight_layout()

    ptr.savefig(fname='hist_sec' + r_print)
    # ptr.show()


    ptr.close("all")

    # ptr.figure(figsize=(15,10), dpi=100)

    # length = 3
    # width = 4

    # size = length * width

    # time_max = 60
    # j_add = time_max / size
    # i_add = width * j_add
    
    # for i in range(3):
    #     for h in range(4):
    #         init_vs = []
    #         sizes = []
    #         lengths = []
    #         sec_cases = []
    #         lower = i_add*i + j_add*h
    #         upper = i_add*i + j_add*(h+1)

    #         binned_init_vs = []
    #         bins_vs = np.linspace(0,500,num=11)
    #         for g in bins_vs:
    #             binned_init_vs.append([])
            
    #         for node in tree.nodes():
    #             if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
    #                 sub = nx.dfs_tree(tree,node)
    #                 bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)

    #                 init_vs.append(tree.nodes[node]['init_v'])
    #                 sizes.append(sub.size())
    #                 lengths.append(get_avg_length_gen(sub,node))
    #                 sec_cases.append(len(list(tree.successors(node))))
    #                 binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v']])
        
    #         if len(init_vs) > 2:
    #             pear_r = pearsonr(init_vs,sizes)
    #             # pear_r = pearsonr(init_vs,lengths)
    #             # pear_r = pearsonr(init_vs,sec_cases)
    #             pear_r = round(pear_r[0],3)

    #             spear_r = spearmanr(init_vs,sizes)
    #             # pear_r = pearsonr(init_vs,lengths)
    #             # pear_r = pearsonr(init_vs,sec_cases)
    #             spear_r_p = round(spear_r[1],3)
    #             spear_r = round(spear_r[0],3)
    #         else:
    #             pear_r = 'NaN'
    #             spear_r = 'NaN'
    #             spear_r_p = 'NaN'

    #         # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    #         labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]

    #         ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Subtree size', xlabel='Root node V0',ylim=(0.1,1000), yscale='log')
    #         # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Mean path length', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
    #         # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Secondary cases', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
    #         # ax.scatter(init_vs,sizes, c=col, alpha=0.3)
    #         ax.boxplot([[h[0] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    #         ptr.xticks(fontsize=8)


    # ptr.tight_layout()

    # ptr.savefig(fname='hist_size')
    # ptr.show()

    # ptr.close("all")

    # ptr.figure(figsize=(15,10), dpi=100)

    # length = 3
    # width = 4

    # size = length * width

    # time_max = 60
    # j_add = time_max / size
    # i_add = width * j_add
    
    # for i in range(3):
    #     for h in range(4):
    #         init_vs = []
    #         sizes = []
    #         lengths = []
    #         sec_cases = []
    #         lower = i_add*i + j_add*h
    #         upper = i_add*i + j_add*(h+1)

    #         binned_init_vs = []
    #         bins_vs = np.linspace(0,500,num=11)
    #         for g in bins_vs:
    #             binned_init_vs.append([])
            
    #         for node in tree.nodes():
    #             if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
    #                 sub = nx.dfs_tree(tree,node)
    #                 bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)

    #                 init_vs.append(tree.nodes[node]['init_v'])
    #                 sizes.append(sub.size())
    #                 lengths.append(get_avg_length_gen(sub,node))
    #                 sec_cases.append(len(list(tree.successors(node))))
    #                 binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v']])
        
    #         if len(init_vs) > 2:
    #             # pear_r = pearsonr(init_vs,sizes)
    #             pear_r = pearsonr(init_vs,lengths)
    #             # pear_r = pearsonr(init_vs,sec_cases)
    #             pear_r = round(pear_r[0],3)

    #             spear_r = spearmanr(init_vs,lengths)
    #             spear_r_p = round(spear_r[1],3)
    #             spear_r = round(spear_r[0],3)
    #         else:
    #             pear_r = 'NaN'
    #             spear_r = 'NaN'
    #             spear_r_p = 'NaN'
            
    #         # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    #         labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]
    #         # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)), ylabel='Subtree size', xlabel='Root node V0', xlim=(0,500),ylim=(0.1,1000), yscale='log')
    #         ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Mean path length', xlabel='Root node V0',ylim=(-1,15))
    #         # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Secondary cases', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
    #         # ax.scatter(init_vs,lengths, c=col, alpha=0.3)
    #         ax.boxplot([[h[1] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    #         ptr.xticks(fontsize=8)


    # ptr.tight_layout()

    # ptr.savefig(fname='hist_len')
    # ptr.show()

    # ptr.close("all")

    # ptr.figure(figsize=(15,10), dpi=100)

    # length = 3
    # width = 4

    # size = length * width

    # time_max = 60
    # j_add = time_max / size
    # i_add = width * j_add
    
    # for i in range(3):
    #     for h in range(4):
    #         init_vs = []
    #         sizes = []
    #         lengths = []
    #         sec_cases = []
    #         lower = i_add*i + j_add*h
    #         upper = i_add*i + j_add*(h+1)

    #         binned_init_vs = []
    #         bins_vs = np.linspace(0,500,num=11)
    #         for g in bins_vs:
    #             binned_init_vs.append([])
            
    #         for node in tree.nodes():
    #             if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
    #                 sub = nx.dfs_tree(tree,node)
    #                 bin_no_vs = np.digitize(tree.nodes[node]['init_v'],bins_vs)

    #                 init_vs.append(tree.nodes[node]['init_v'])
    #                 sizes.append(sub.size())
    #                 lengths.append(get_avg_length_gen(sub,node))
    #                 sec_cases.append(len(list(tree.successors(node))))
    #                 binned_init_vs[bin_no_vs].append([sub.size(),get_avg_length_gen(sub,node),len(list(tree.successors(node))),np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'], sub.size() / get_avg_length_gen(sub,node)])
        
    #         if len(init_vs) > 2:
    #             # pear_r = pearsonr(init_vs,sizes)
    #             # pear_r = pearsonr(init_vs,lengths)
    #             pear_r = pearsonr(init_vs,sec_cases)
    #             pear_r = round(pear_r[0],3)

    #             spear_r = spearmanr(init_vs,sec_cases)
    #             spear_r_p = round(spear_r[1],3)
    #             spear_r = round(spear_r[0],3)
    #         else:
    #             pear_r = 'NaN'
    #             spear_r = 'NaN'
    #             spear_r_p = 'NaN'

    #         # labs_vs = [str(int(bins_vs[h-1])) + '-' + str(int(bins_vs[h])) for h in range(1,len(bins_vs))]
    #         labs_vs = [str(int(bins_vs[h-1])) for h in range(1,len(bins_vs))]
    #         # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)), ylabel='Subtree size', xlabel='Root node V0', xlim=(0,500),ylim=(0.1,1000), yscale='log')
    #         # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Mean path length', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
    #         ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Secondary cases', xlabel='Root node V0',ylim=(-1,15))
    #         # ax.scatter(init_vs,sec_cases, c=col, alpha=0.3)
    #         ax.boxplot([[h[2] for h in g] for g in binned_init_vs[1:]], labels=labs_vs)
    #         ptr.xticks(fontsize=8)


    # ptr.tight_layout()

    # ptr.savefig(fname='hist_sec')
    # ptr.show()


    ptr.close("all")

    ptr.figure(figsize=(15,10), dpi=100)

    length = 3
    width = 4

    size = length * width

    time_max = 60
    j_add = time_max / size
    i_add = width * j_add
    
    for i in range(3):
        for h in range(4):
            init_vs = []
            sizes = []
            lengths = []
            sec_cases = []
            GFs = []

            lower = i_add*i + j_add*h
            upper = i_add*i + j_add*(h+1)
            
            for node in tree.nodes():
                if tree.nodes[node]['time'] > lower*steps_per_day and tree.nodes[node]['time'] < upper*steps_per_day:
                    sub = nx.dfs_tree(tree,node)

                    # init_vs.append(tree.nodes[node]['init_v'])
                    # sizes.append(sub.size())
                    # lengths.append(get_avg_length_gen(sub,node))
                    # sec_cases.append(len(list(tree.successors(node))))
                    if len(list(tree.successors(node))) > 0:
                        init_vs.append(tree.nodes[node]['init_v'])
                        GFs.append(np.mean([tree.nodes[i]['init_v'] for i in tree.successors(node)]) / tree.nodes[node]['init_v'])
            
            # GFs = [x for x in GFs if str(x) != 'nan']
            if len(init_vs) > 2:
                # pear_r = pearsonr(init_vs,sizes)
                # pear_r = pearsonr(init_vs,lengths)
                # pear_r = pearsonr(init_vs,sec_cases)
                pear_r = pearsonr(init_vs,GFs)
                pear_r = round(pear_r[0],3)

                spear_r = spearmanr(init_vs,GFs)
                spear_r_p = round(spear_r[1],3)
                spear_r = round(spear_r[0],3)
            else:
                pear_r = 'NaN'
                spear_r = 'NaN'
                spear_r_p = 'NaN'
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)), ylabel='Subtree size', xlabel='Root node V0', xlim=(0,500),ylim=(0.1,1000), yscale='log')
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Mean path length', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
            # ax = ptr.subplot2grid((3,4), (i,h), title='Time (days) range: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r), ylabel='Secondary cases', xlabel='Root node V0', xlim=(0,500),ylim=(-1,15))
            ax = ptr.subplot2grid((3,4), (i,h), title='Days: ' + str(int(lower)) + ' - ' + str(int(upper)) + ', r = ' + str(pear_r) + ', $\\rho$ = ' + str(spear_r) + ' (' + str(spear_r_p) + ')', ylabel='Growth factor', xlabel='Root node V0', xlim=(0,500),ylim=(-1,10))
            ax.scatter(init_vs,GFs, c=col, alpha=0.3)
            


    ptr.tight_layout()

    ptr.savefig(fname='hist_GF_' + r_print)
    # ptr.show()




        
