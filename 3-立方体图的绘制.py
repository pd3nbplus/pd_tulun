# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:06:01 2021

@author: 潘登
"""

#%%3-立方体图的绘制
import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
points = {'100':['101', '110', '000'],
          '110':['111', '100', '010'],
          '000':['001', '010', '100'],
          '001':['101', '011', '000'],
          '011':['001', '111', '010'],
          '101':['111', '100', '001'],
          '111':['110', '101', '011'],
          '010':['110', '011', '000']}
for i in points:
    for j in points[i]:
        G.add_edge(i, j)
 # 设置节点的位置
left = ['100', '110']
middle_left = ['000', '010']
middle_right = ['001', '011']
right = ['101', '111']

options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "black",
        "edge_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
pos1 = {n: (0, 5-3*i) for i, n in enumerate(left)}
pos1.update({n: (1, 4-i) for i, n in enumerate(middle_left)})
pos1.update({n: (2, 4-i) for i, n in enumerate(middle_right)})
pos1.update({n: (3, 5-3*i) for i, n in enumerate(right)})
plt.clf()
nx.draw(G, pos1, **options)
#%% 轮图的绘制
import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
points = {'0':['1','2','3','4'],
          '1':'2',
          '2':'3',
          '3':'4',
          '4':'1'}
for i in points:
    for j in points[i]:
        G.add_edge(i, j)
plt.clf()
pos = nx.spring_layout(G)
nx.draw(G, pos)
# %%全等图的绘制
import networkx as nx
import matplotlib.pyplot as plt

G1 = nx.Graph()
points1 = {'v1':[['v2', 'e4'], ['v4', 'e2']],
         'v2':[['v3', 'e3'], ['v4', 'e6']],
         'v3':[['v4', 'e1'], ['v1', 'e5']]}

for i in points1:
    for j in points1[i]:
        G1.add_edge(i, j[0], name = j[1])
options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "blue",
        "edgecolors": "black",
        "edge_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
left = ['v1', 'v2']
right = ['v3', 'v4']

pos1 = {n: (0, 5-3*i) for i, n in enumerate(left)}
pos1.update({n: (3, 4.5-2*i) for i, n in enumerate(right)})

plt.clf()
pos = nx.spring_layout(G1)
nx.draw(G1, pos1, **options)

# 显示edge的labels
edge_labels = nx.get_edge_attributes(G1, 'name')
nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels, font_size=20, font_color='blue')
#%%
G2 = nx.Graph()
points2 = {'a':[['b', 'E4'], ['d', 'E2']],
           'b':[['c', 'E3'], ['d', 'E6']],
           'c':[['d', 'E1'], ['a', 'E5']]}

for i in points2:
    for j in points2[i]:
        G2.add_edge(i, j[0], name = j[1])
options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "blue",
        "edgecolors": "black",
        "edge_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
upper = ['d']
middle = ['c']
lower = ['a', 'b']
pos = {n: (1.5, 5) for _, n in enumerate(upper)}
pos.update({n: (3*i, 1) for i, n in enumerate(lower)})
pos.update({n: (1.5, 2.5) for i, n in enumerate(middle)})

plt.clf()
nx.draw(G2, pos, **options)

# 显示edge的labels
edge_labels = nx.get_edge_attributes(G2, 'name')
nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels, font_size=20, font_color='blue')
# %%科学家排座问题
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
a_dict = {'A':['English'],
          'B':['English', 'Chinese'],
          'C':['English', 'Italian', 'Russian'],
          'D':['Japanese', 'Chinese'],
          'E':['German', 'Italian'],
          'F':['French', 'Japanese', 'Russian'],
          'G':['German', 'French']}

# 用于储存相邻节点, key是语言， value是people_name
neighbors_dict = {}
for people_name in a_dict:
    G.add_node(people_name)
    for language in a_dict[people_name]:
        # 如果这个语言在neighbors_dict, 那么就把会这门语言的其他人
        # 与这个人之间连接一条边， 否则就把这个人会的语言加到dict中
        if language in neighbors_dict:
            for people_name_exist in neighbors_dict[language]:
                G.add_edge(people_name_exist, people_name, language=language)
            neighbors_dict[language].append(people_name)
        else:
            neighbors_dict[language] = [people_name]

options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "white",
        "edge_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
pos = nx.spring_layout(G)
plt.clf()
nx.draw(G, pos, **options)

# 显示edge的labels
edge_labels = nx.get_edge_attributes(G, 'language')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='blue')





