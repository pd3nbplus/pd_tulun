# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:24:51 2021

@author: 潘登
"""

#%% 科学家排座问题
import sys


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.color = 'white'
        self.dist = sys.maxsize  # 无穷大
        self.pred = None

    def addNeighbor(self, nbr, weight=1):   # 增加相邻边，
        self.connectedTo[nbr] = weight

    def __str__(self):   # 显示设置
        return str(self.id) + 'connectTo:' + \
               str([x.id for x in self.connectedTo])

    def getConnections(self):   # 获得相邻节点
        return self.connectedTo.keys()

    def getId(self):   # 获得节点名称
        return self.id

    def getWeight(self, nbr):   # 获得相邻边数据
        return self.connectedTo[nbr]

    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color

    def setPred(self, p):
        self.pred = p

    def getPred(self):  # 这个前驱节点主要用于追溯，是记录离起始节点最短路径上
        return self.pred    # 该节点的前一个节点是谁

    def setDistance(self, d):
        self.dist = d

    def getDistance(self):
        return self.dist

class Graph:
    def __init__(self):
        self.vertList = {}  # 这个虽然叫list但是实质上是字典
        self.numVertices = 0

    def addVertex(self, key):   # 增加节点
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)   # 创建新节点
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, key):   # 通过key获取节点信息
        if key in self.vertList:
            return self.vertList[key]
        else:
            return None

    def __contains__(self, n):  # 判断节点在不在图中
        return n in self.vertList

    def addEdge(self, from_key, to_key, cost=1):    # 新增边
        if from_key not in self.vertList:      # 不再图中的顶点先添加
            self.addVertex(from_key)
        if to_key not in self.vertList:
            self.addVertex(to_key)
        # 调用起始顶点的方法添加邻边
        self.vertList[from_key].addNeighbor(self.vertList[to_key], cost)

    def addUndirectedEdge(self, key1, key2, cost=1):
        self.addEdge(key1, key2, cost)
        self.addEdge(key2, key1, cost=1)

    def getVertices(self):   # 获取所有顶点的名称
        return self.vertList.keys()

    def __iter__(self):  # 迭代取出
        return iter(self.vertList.values())

    def __len__(self):
        return self.numVertices


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, items):   # 往队列加入数据
        self.items.insert(0, items)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

class Solution():
    def createGraph(self, a_dict):
        '''
        input: a_dict type: dict, key: people_name, value: language
        return: graph
        '''
        # 创建图
        graph = Graph()
        # 用于储存相邻节点, key是语言， value是people_name
        neighbors_dict = {}
        for people_name in a_dict:
            graph.addVertex(people_name)
            for language in a_dict[people_name]:
                # 如果这个语言在neighbors_dict, 那么就把会这门语言的其他人
                # 与这个人之间连接一条边， 否则就把这个人会的语言加到dict中
                if language in neighbors_dict:
                    for people_name_exist in neighbors_dict[language]:
                        graph.addUndirectedEdge(people_name_exist, people_name)
                    neighbors_dict[language].append(people_name)
                else:
                    neighbors_dict[language] = [people_name]
        return graph


    def DFS(self, g, start):
       result = []
       # 深度优先搜索
       def trace(path, g, start):   # path是探索的路径 g是图，start是起始的节点
           start.setColor('gray')  # 将正在探索的节点设置为灰色
           path.append(start)
           # path中包含了所有顶点，且最后一个顶点到初始顶点要有路径就是最终结果
           if len(path) == len(g):
               if path[-1] in path[0].getConnections():
                   result.append(list(path))
               return
           else:
               for i in list(start.getConnections()):
                   if i.getColor() == 'white':
                       trace(path, g, i)
                       path.pop()
                       i.setColor('white')

       trace([], g, start)
       return result

if __name__ == '__main__':
    a_dict = {'A':['English'],
              'B':['English', 'Chinese'],
              'C':['English', 'Italian', 'Russian'],
              'D':['Japanese', 'Chinese'],
              'E':['German', 'Italian'],
              'F':['French', 'Japanese', 'Russian'],
              'G':['German', 'French']}
    s = Solution()
    graph = s.createGraph(a_dict)
    path = s.DFS(graph, graph.getVertex('A'))
    # 因为是无向图，如果存在一个回路那一定会有两条路径
    for n in range(len(path)):
        for i in path[n]:
            print(i.getId(), end=' ')
        print('\t')