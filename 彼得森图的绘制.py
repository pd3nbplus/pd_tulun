# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:48:05 2021

@author: 潘登
"""

#%%彼得森图
import networkx as nx
import matplotlib.pyplot as plt
Graph = nx.petersen_graph()
nx.draw_shell(Graph, nlist=[range(5, 10), range(5)],  
              font_weight='bold',node_color=range(10),
              cmap=plt.cm.Reds,font_color='r',style='dotted')
plt.show()
#%%彼得森图不是平面图的证明
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
G = nx.Graph()
points = {'A':['B', 'C', 'O'],
          'B':['A', 'D', 'I'],
          'C':['A', 'E', 'H'],
          'D':['B', 'E', 'F'],
          'E':['C', 'D', 'G'],
          'F':['D', 'H', 'O'],
          'G':['E', 'I', 'O'],
          'H':['F', 'I', 'C'],
          'I':['G', 'H', 'B'],
          'O':['A', 'G', 'F']}

for i in points:
    for j in points[i]:
        G.add_edge(i, j)

points_1 = ['A']
points_2 = ['B', 'C']
points_3 = ['D', 'E']
points_4 = ['F', 'G']
points_5 = ['H', 'I']
points_6 = ['O']

options = {
        "font_size": 24,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "white",
        "edge_color": 'blue',
        "font_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }

pos = {n: (2, 5) for i, n in enumerate(points_1)}
pos.update({n: (1+2*i, 4.2) for i, n in enumerate(points_2)})
pos.update({n: (4*i, 3.3) for i, n in enumerate(points_3)})
pos.update({n: (0.5+3*i, 2) for i, n in enumerate(points_4)})
pos.update({n: (1.25+1.5*i, 1) for i, n in enumerate(points_5)})
pos.update({n: (2, 3) for i, n in enumerate(points_6)})



plt.figure(figsize=(16,8))
plt.subplot(131)
nx.draw(G, pos, **options)
G.remove_node('O')
plt.title('Petersen图')

plt.subplot(132)
pos.pop('O')
options['edge_color'] = 'red'
options['font_color'] = 'black'
nx.draw(G, pos, **options)
plt.title('Petersen图的子图')

plt.subplot(133)
G3_3 = G = nx.Graph()
points3_3 = {'B':['D', 'C', 'I'],
          'C':['B', 'E', 'H'],
          'D':['B', 'E', 'H'],
          'E':['C', 'D', 'I'],
          'H':['C', 'D', 'I'],
          'I':['E', 'H', 'B']}

for i in points3_3:
    for j in points3_3[i]:
        G3_3.add_edge(i, j)

pos1 = {n: (1+2*i, 4.2) for i, n in enumerate(points_2)}
pos1.update({n: (4*i, 3.3) for i, n in enumerate(points_3)})
pos1.update({n: (1.25+1.5*i, 1) for i, n in enumerate(points_5)})

options['edge_color'] = 'green'
options['font_color'] = 'blue'
nx.draw(G3_3, pos1, **options)
plt.title('Petersen图的子图的细分同构')
#%%可平面性的判定例子
G = nx.Graph()
points_1 = {'A':['B', 'C', 'D', 'G'],
          'B':['A', 'D', 'C'],
          'C':['A', 'E', 'B'],
          'D':['B', 'A', 'F', 'G'],
          'E':['C', 'F', 'G'],
          'F':['D', 'E', 'H'],
          'G':['E', 'A', 'D', 'H'],
          'H':['G', 'F', 'C']}
# 用于显示边
G_edge = nx.Graph()
points_edge = {'A':['D', 'G'],
              'B':['C'],
              'C':['B'],
              'D':['A', 'G'],
              'E':['F',],
              'F':['E',],
              'G':['A', 'D',],
              'H':['C']}

G_bin = nx.Graph()
points_bin = {'GA': ['CB', 'FE'],
              'GD': ['FE'],
              'DA': ['CB']}

for i in points_1:
    for j in points_1[i]:
        G.add_edge(i, j)

for i in points_edge:
    for j in points_edge[i]:
        G_edge.add_edge(i, j, name=i + j)

for i in points_bin:
    for j in points_bin[i]:
        G_bin.add_edge(i, j)

points_1 = ['A', 'B']
points_2 = ['C', 'D']
points_3 = ['E', 'F']
points_4 = ['G', 'H']

pos = {n: (1+i, 5) for i, n in enumerate(points_1)}
pos.update({n: (0.5+2*i, 4.2) for i, n in enumerate(points_2)})
pos.update({n: (0.5+2*i, 2.8) for i, n in enumerate(points_3)})
pos.update({n: (1+i, 2) for i, n in enumerate(points_4)})

options = {
        "font_size": 24,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "white",
        "edge_color": 'red',
        "font_color": 'black',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
plt.figure(figsize=(16, 8))
plt.subplot(131)
nx.draw(G, **options)

plt.title('未处理过的图')

plt.subplot(132)
nx.draw(G, pos, **options)
edge_labels = nx.get_edge_attributes(G_edge, 'name')
nx.draw_networkx_edge_labels(G_edge, pos, edge_labels=edge_labels, 
                             font_size=20, font_color='blue')
plt.title('将哈密顿回路画成环的图')

plt.subplot(133)
points_bin1 = ['GA', 'GD', 'DA']
points_bin2 = ['CB', 'FE']

pos_bin = {n: (1, 5-2*i) for i, n in enumerate(points_bin1)}
pos_bin.update({n: (3, 5-2*i) for i, n in enumerate(points_bin2)})

options_bin = {
        "font_size": 30,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "white",
        "edge_color": 'blue',
        "font_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }

nx.draw(G_bin, pos_bin, **options_bin)
plt.title('新的图G`是二部图')
