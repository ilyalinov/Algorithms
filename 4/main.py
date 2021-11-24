import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import string
import pygraphviz
from networkx.algorithms import approximation as approx

dt = [('len', float)]
m = np.loadtxt('dist.txt', dtype='float')
m = m.view(dt)

G = nx.from_numpy_matrix(m)
cycle = approx.simulated_annealing_tsp(G, init_cycle=list(G) + [next(iter(G))], weight='len', source=0, max_iterations=1000)
starting_cost = sum(G[n][nbr]["len"] for n, nbr in nx.utils.pairwise(list(G) + [next(iter(G))]))
cost = sum(G[n][nbr]["len"] for n, nbr in nx.utils.pairwise(cycle))
print(cycle)
print(starting_cost)
print(cost)
# rename nodes from 0123... to ABCD...
# G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), string.ascii_uppercase)))

G = nx.nx_agraph.to_agraph(G)

G.node_attr["fixedsize"] = "true"
# G.node_attr['overlap'] = 'false'
G.node_attr["width"] = "5"
G.node_attr["height"] = "5"
G.node_attr["style"] = "filled"
G.graph_attr["ratio"] = "1.0"

G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="transparent", width="2.0")
G.node_attr["fontsize"] = "150"
G.edge_attr["style"] = "setlinewidth(20)"
# for i in range(15):
#     e = G.get_edge(i, (i + 1) % 15)
#     e.attr['color'] = 'blue'

for i in range(15):
    e = G.get_edge(cycle[i], cycle[i + 1])
    e.attr['color'] = 'blue'

G.draw('distances.png', prog='neato')
G.draw('out.dot', format='dot', prog='neato')