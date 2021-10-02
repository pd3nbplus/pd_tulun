# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:43:32 2021

@author: 潘登
"""
#%%最大匹配二部图的图
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
a_dict = {'x1':['y3', 'y5'],
        'x2':['y1', 'y2', 'y3'],
        'x3':['y4'],
        'x4':['y4', 'y5'],
        'x5':['y5']}

for i in a_dict:
    for j in a_dict[i]:
        G.add_edge(i, j)

x_list = ['x1', 'x2', 'x3', 'x4', 'x5']
y_list = ['y1', 'y2', 'y3', 'y4', 'y5']

pos = {n: (1, 5-i) for i, n in enumerate(x_list)}
pos.update({n: (3, 5-i) for i, n in enumerate(y_list)})

plt.clf()
options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "green",
        "edgecolors": "white",
        "edge_color": 'blue',
        "font_color": 'red',
        "linewidths": 12,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
nx.draw(G, pos, **options)
#%%2-正则二部图
G = nx.Graph()
a_dict = {'A':['1', '2'],
        'B':['2', '3'],
        'C':['3', '4'],
        'D':['4', '5'],
        'E':['5', '1']}

for i in a_dict:
    for j in a_dict[i]:
        G.add_edge(i, j)

x_list = ['A', 'B', 'C', 'D', 'E']
y_list = ['1', '2', '3', '4', '5']

pos = {n: (i, 3) for i, n in enumerate(x_list)}
pos.update({n: (i, 1) for i, n in enumerate(y_list)})

plt.clf()
options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "green",
        "edgecolors": "white",
        "edge_color": 'blue',
        "font_color": 'red',
        "linewidths": 12,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
nx.draw(G, pos, **options)
#%%最大匹配与最小边覆盖之间的关系 证明用图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
G = nx.Graph()
G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])
G.add_edge('B', 'D')
G.add_edge('E', 'F')

x_list = ['A', 'B', 'C']
y_list = ['D', 'E', 'F']

pos = {n: (i, 2) for i, n in enumerate(x_list)}
pos.update({n: (i, 1) for i, n in enumerate(y_list)})

options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "green",
        "edgecolors": "red",
        "edge_color": 'blue',
        "font_color": 'black',
        "linewidths": 12,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }

plt.figure(figsize=(12,6))
plt.subplot(121)
nx.draw(G, pos, **options)
plt.title('最大匹配M')

plt.subplot(122)
G.add_edge('A', 'B')
G.add_edge('C', 'B')
options['edgecolors'] = 'green'
options['edge_color'] = 'gray'

nx.draw(G, pos, **options)
plt.title('最小边覆盖')
#%%
G = nx.Graph()
G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])
G.add_edge('D', 'E')
G.add_edge('B', 'C')


x_list = ['A', 'B', 'C']
y_list = ['D', 'E', 'F']

pos = {n: (i, 2) for i, n in enumerate(x_list)}
pos.update({n: (i, 1) for i, n in enumerate(y_list)})

options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "green",
        "edgecolors": "red",
        "edge_color": 'blue',
        "font_color": 'black',
        "linewidths": 12,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }

plt.figure(figsize=(12,6))
plt.subplot(122)
nx.draw(G, pos, **options)
plt.title('最大匹配M')

plt.subplot(121)
G.add_edge('A', 'B')
G.add_edge('E', 'F')
options['edgecolors'] = 'green'
options['edge_color'] = 'gray'

nx.draw(G, pos, **options)
plt.title('最小边覆盖')
