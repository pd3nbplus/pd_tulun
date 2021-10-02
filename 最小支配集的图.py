# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 09:09:43 2021

@author: 潘登
"""

#%%最小支配集的图
import networkx as nx
# import matplotlib.pyplot as plt

G = nx.Graph()
points = {'V1':['V2', 'V6'],
          'V2':['V6'],
          'V3':['V4', 'V6', 'V7'],
          'V4':['V7'],
          'V5':['V6']}
for i in points:
    for j in points[i]:
        G.add_edge(i, j)

options = {
        "font_size": 36,
        "node_size": 1000,
        'alpha':0.8,
        'with_labels':True
    }
nx.draw(G, **options)
