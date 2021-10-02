# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 08:45:44 2021

@author: 潘登
"""

#%%韦尔奇-鲍威尔算法
import networkx as nx
import matplotlib.pyplot as plt
from Vertex import Vertex
from Graph import Graph
import matplotlib as mpl

class New_Vertex(Vertex):  # 某一个具体问题的数据结构需要继承原有数据结构
    def __init__(self, key):
        super().__init__(key)
        self.degree = 0   # 新增类属性(用于节点排序)
        self.color = 'white'  # 新增类属性(用于记录节点的颜色)

    # 重写类方法
    def addNeighbor(self, nbr, weight=0):   # 增加相邻边，默认weight为0
        '''
        input:
            nbr: Vertex object
            weight: int
        return:
            None
        '''
        self.connectedTo[nbr] = weight
        self.degree += 1

    # 新增类方法 (查看degree)
    def getDegree(self):
        return self.degree

    # 新增类方法, 设置节点颜色
    def setColor(self, color):
        self.color = color

    # 新增类方法, 查看节点颜色
    def getColor(self):
        return self.color
    
class colorGraph(Graph):
    def __init__(self):
        super().__init__()

    # 重载方法  因为原先Graph中新增节点用的是Vertex节点,但现在是用New_Vertex
    def addVertex(self, key):   # 增加节点
        '''
        input: Vertex key (str)
        return: Vertex object
        '''
        self.numVertices = self.numVertices + 1
        newVertex = New_Vertex(key)   # 创建新节点
        self.vertList[key] = newVertex
        return newVertex

# 队列数据结构
class Queue():
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return self.queue.pop(0)

    def isEmpty(self):
        return self.queue == []

    def size(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    # 查看队首元素
    def see(self):
        return self.queue[0]

class Solution():
    def createGraph(self, a_dict):
        graph = colorGraph()
        for i in a_dict:
            for j in a_dict[i]:
                graph.addEdge(i, j)
        return graph

    # 排序算法 -快速排序
    def quickSort(self, a_list):
        if len(a_list) <= 1:  # 有可能出现left或者right是空的情况
            return a_list
        else:
            mid = a_list[len(a_list)//2]
            left = []
            right = []
            a_list.remove(mid)
            for i in a_list:
                if i[1] > mid[1]:
                    right.append(i)
                else:
                    left.append(i)
            return self.quickSort(left) + [mid] + self.quickSort(right)

    def Welch_Powell(self, g):
        queue = Queue()
        Vertices_keys = g.getVertices()
        Vertices_obj = [g.getVertex(k) for k in Vertices_keys]
        # 用于储存顶点和他的degree
        Vertices_deg = [[i, i.getDegree()] for i in Vertices_obj]
        # 对Vertices_deg进行排序, 然后扔进队列里
        for i in self.quickSort(Vertices_deg)[::-1]:
            queue.enqueue(i[0])
        # 当队列非空
        color = 0  # 颜色标记
        # 已着色顶点
        color_done_vertex = []
        while not queue.isEmpty():
            # 对第一个点进行着色
            frist_vertex = queue.dequeue()
            frist_vertex.setColor(color)
            color_done_vertex.append(frist_vertex)
            for _ in range(queue.size()):
                # 如果color_done_vertex与i这个节点有连接
                Connections = []
                for k in color_done_vertex:
                    Connections += list(k.getConnections())
                if queue.see() in Connections:
                    # 将节点从队首加到队尾
                    queue.enqueue(queue.dequeue())
                else:
                    temp = queue.dequeue()
                    temp.setColor(color)
                    color_done_vertex.append(temp)
            color += 1
        # 输出结果
        result = []
        while Vertices_obj:
            temp_vertex = Vertices_obj.pop()
            result.append((temp_vertex.getId(), temp_vertex.getColor()))
            print(temp_vertex.getId(), ' 的颜色是:', temp_vertex.getColor())
        return result

if __name__ == '__main__':
    a_dict = {'a':['b', 'g', 'h'],
              'b':['a', 'd', 'g', 'h'],
              'c':['d', 'e'],
              'd':['b', 'c', 'f'],
              'e':['c', 'f'],
              'f':['d', 'e'],
              'g':['a', 'b', 'h'],
              'h':['a', 'b', 'g']}
    s = Solution()
    graph = s.createGraph(a_dict)
    result = s.Welch_Powell(graph)

    G = nx.Graph()
    for i in a_dict:
        for j in a_dict[i]:
            G.add_edge(i, j)
    
    color = list(mpl.colors.TABLEAU_COLORS.values())
    node_color = []
    for i in result:
        node_color.append(color[i[1]])
    plt.clf()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_color, font_size= 35,with_labels=True)