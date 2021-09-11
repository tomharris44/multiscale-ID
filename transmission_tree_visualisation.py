import pylab as plt
import matplotlib.pyplot as ptr
import networkx as nx

# initialise plot
ax = plt.subplot(111)
tree = nx.DiGraph()
ax.set_title('Transmission Tree')
ax.set_xticks([])

# add cases to transmission tree
tree.add_node(0,init_v=2,time=0,gen=0)
tree.add_node(1,init_v=200,time=2000,gen=1)
tree.add_edge(0,1, rec_init=200)

# define graph layout
pos=nx.drawing.nx_agraph.graphviz_layout(tree, prog='dot')
max_time = max([tree.nodes[i]['time'] for i in tree.nodes()])

# alter y positions to reflect infected time
for node in tree.nodes():
    pos[node] = (pos[node][0],-1*tree.nodes[node]['time'])

# draw graph
nodes = nx.draw_networkx_nodes(tree, ax=ax, pos=pos, node_size=50,
                node_color=list(nx.get_node_attributes(tree, 'init_v').values()),
                cmap=ptr.cm.viridis, vmin=0)
edges = nx.draw_networkx_edges(tree, ax=ax, pos=pos, node_size=50)
edge_labels_dict = nx.get_edge_attributes(tree,'rec_init')
edge_labels = nx.draw_networkx_edge_labels(tree, ax=ax, pos=pos, font_size=7, edge_labels=edge_labels_dict)

ptr.colorbar(nodes)

# setup y labels
yticks = range(0,-1*max_time,-100)
labelsy = [ round(-1*i/30) for i in yticks]

ax.set_yticks(yticks)
ax.set_yticklabels(labelsy)

ax.set_ylabel('Days')

ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

ptr.tight_layout()

ptr.show()