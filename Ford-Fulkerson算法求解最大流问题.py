# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:05:16 2021

@author: 潘登
"""

#%%Ford-Fulkerson算法求解最大流问题
from Vertex import Vertex # 导入Vertex
from Graph import Graph  # 导入之前实现的Graph
import sys

class New_Vertex(Vertex):  # 某一个具体问题的数据结构需要继承原有数据结构
    def __init__(self, key):
        super().__init__(key)
        self.color = 'white'  # 新增类属性(用于记录节点是否被走过)
        self.dist = sys.maxsize  # 新增类属性(用于记录strat到这个顶点的距离)初始化为无穷大
        self.pred = None  # 顶点的前驱 BFS需要

    # 新增类方法, 设置节点颜色
    def setColor(self, color):
        self.color = color

    # 新增类方法, 查看节点颜色
    def getColor(self):
        return self.color

    # 新增类方法, 设置节点前驱
    def setPred(self, p):
        self.pred = p

    # 新增类方法, 查看节点前驱
    def getPred(self):  # 这个前驱节点主要用于追溯，是记录离起始节点最短路径上
        return self.pred    # 该节点的前一个节点是谁

    # 新增类方法, 设置节点距离
    def setDistance(self, d):
        self.dist = d

    # 新增类方法, 查看节点距离
    def getDistance(self):
        return self.dist

class New_Graph(Graph):  # 继承Graph对象
    def __init__(self):
        super().__init__()

    # 重载方法  因为原先Graph中新增节点用的是Vertex节点,但现在是用New_Vertex
    def addVertex(self, key):   # 增加节点
        '''
        input: Vertex key (str)
        return: Vertex object
        '''
        if key in self.vertList:
            return
        self.numVertices = self.numVertices + 1
        newVertex = New_Vertex(key)   # 创建新节点
        self.vertList[key] = newVertex
        return newVertex

# %广度优先算法（先从距离为1开始搜索节点，搜索完所有距离为k才搜索距离为k+1）
'''
为了跟踪顶点的加入过程，并避免重复顶点，要为顶点增加三个属性
    距离distance:从起始顶点到此顶点路径长度
    前驱顶点predecessor:可反向追随到起点
    颜色color：标识了此顶点是尚未发现（白色）,已经发现（灰色）,还是已经完成探索（黑色）
还需用一个队列Queue来对已发现的顶点进行排列
    决定下一个要探索的顶点（队首顶点）

BFS算法过程
    从起始顶点s开始，作为刚发现的顶点，标注为灰色，距离为0，前驱为None，
    加入队列，接下来是个循环迭代过程：
        从队首取出一个顶点作为当前顶点；遍历当前顶点的邻接顶点，如果是尚未发现的白
        色顶点，则将其颜色改为灰色（已发现），距离增加1，前驱顶点为当前顶点，加入到队列中
    遍历完成后，将当前顶点设置为黑色（已探索过），循环回到步骤1的队首取当前顶点
'''

# 队列 -- 也是BFS需要的
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

class max_flow_graph():
    # 创建图和初始化流f
    # 注意图中的cost由[flow, cost] 组成 前者表示流量 后者表示容量
    # 既可以传入字典也能传入邻接矩阵
    def createGraph_f(self, g_dict=None, g_matrix=None):
        '''
        input: g_dict(邻接表) or g_matrix(邻接矩阵)
        output: directgraph(有向图)
        '''
        graph = New_Graph()
        # f = Graph()
        if g_dict:
            for from_key in g_dict:
                for to_key in g_dict[from_key]:
                    # 这里就是注意提到的f跟着c走所以原图的c是[f,c],这个很重要
                    graph.addEdge(from_key, to_key[0], [0, to_key[1]])

        elif g_matrix:
            # 先给顶点起个名字
            name = [str(i) for i in range(1, len(g_matrix)+1)]
            for i, from_key in enumerate(name):
                for j, to_key in enumerate(name):
                    if g_matrix[i][j] != float('inf'):
                        graph.addEdge(from_key, to_key, [0,g_matrix[i][j]])
        return graph

    # 得到简单的s-t道路P
    def BFS(self, gf, start, end):   # g是图，start是起始的节点
        '''
        g: Gf 剩余图
        start: key
        end: key
        return: p 以节点为元素的道路列表
        '''
        start.setDistance(0)  # 距离
        start.setPred(None)  # 前驱节点
        vertQueue = Queue()   # 队列
        vertQueue.enqueue(start)  # 把起始节点加入图中
        while vertQueue.size() > 0:   # 当搜索完所有节点时，队列会变成空的
            currentVert = vertQueue.dequeue()  # 取队首作为当前顶点
            for nbr in currentVert.getConnections():  # 遍历临接顶点
                if (nbr.getColor() == 'white'):   # 当邻接顶点是灰色的时候
                    nbr.setColor('gray')
                    nbr.setDistance(currentVert.getDistance() + 1)
                    nbr.setPred(currentVert)
                    vertQueue.enqueue(nbr)
            currentVert.setColor('balck')
        P = []
        P.append(end)
        while end.getPred():
            end = end.getPred()
            P.append(end)
        return P[::-1]

    # 由原图构造剩余图
    def create_remain_graph(self, g):
        '''
        input: g有向图(包含flow流量和cost容量)
        output: gf剩余图(就是把flow的方法反过来的一个图)
        '''
        # 剩余图gf的边只有一个属性就是cost
        gf = New_Graph()
        # 获得顶点集
        verlist = g.vertList
        for i in verlist:
            from_key = verlist[i]
            gf.addVertex(from_key.getId())
            for to_key in from_key.getConnections():
                gf.addVertex(to_key.getId())
                f = from_key.getWeight(to_key)[0]  # f表示流
                c = from_key.getWeight(to_key)[1]  # c表示容量
                # 前向边
                if f <  c:
                    c_new = c - f
                    gf.addEdge(from_key.getId(), to_key.getId(), c_new)
                # 后向边
                if f > 0:
                    gf.addEdge(to_key.getId(), from_key.getId(), f)
        return gf

    def print_flow(self, g):
        '''
        input: g 带流的图
        output: flow_matrix
        '''
        result = [[0]*len(g) for _ in range(len(g))]
        for i in range(1, len(g)):
            from_key = str(i)
            for j in range(i+1, len(g)+1):
                to_key = str(j)
                from_Vertex = g.getVertex(from_key)
                to_Vertex = g.getVertex(to_key)
                try:
                    result[i-1][j-1] = from_Vertex.getWeight(to_Vertex)[0]
                except:
                    pass
        return result
                
    def Ford_Fulkerson(self, g):
        '''
        算法主流程:
            1.初始化流为0 (f<--0)(我把这步放到createGraph中了)
            2.构造G关于f的剩余图gf
            3.若gf中存在增广道路P, 则由增广道路P构造G的一个新流f’
              f <-- f' 转步骤2
              否则输出f

        Parameters
        ----------
        g : graph
            由createGraph_f创建的图.
            
        Returns
        -------
        choice matrix
            流量方案的选择(每一行表示一个流出点, 每一列表示一个流入点).

        '''
        # 构造g关于f的剩余图Gf
        gf = self.create_remain_graph(g)
        # 求出Gf的增广道路
        P = self.BFS(gf, gf.getVertex('1'), gf.getVertex(str(len(g))))
        # 若存在增广道路 则由增广道路构造G的一个新流f
        while len(P)>1:
            # 计算增广路中的bottleneck
            bottleneck = 1e5
            for i in range(len(P)-1):
                if P[i].getWeight(P[i+1]) < bottleneck:
                    bottleneck = P[i].getWeight(P[i+1])
            # 更新f
            for i in range(len(P)-1):
                u_name = P[i].getId()
                v_name = P[i+1].getId()
                u = g.vertList[u_name]
                v = g.vertList[v_name]
                c = u.getWeight(v)
                # 如果(u,v)是g中的正向边
                if v in u.getConnections():
                    c[0] += bottleneck
                elif u in v.getConnections():
                    c[0] -= bottleneck
                # 流一定是正向的，只有剩余图的边有可能是反向的
                g.addEdge(u_name, v_name, c)
            # 更新gf
            gf = self.create_remain_graph(g)
            # 求出Gf的增广道路
            P = self.BFS(gf, gf.getVertex('1'), gf.getVertex(str(len(g))))
        return self.print_flow(g)

if __name__ == '__main__':
    g_dict ={'1':[['2', 10], ['3', 10]],
             '2':[['3', 2], ['4', 4], ['5', 8]],
             '3':[['5', 9]],
             '4':[['6', 10]],
             '5':[['4', 6], ['6', 10]]}
    s = max_flow_graph()
    g = s.createGraph_f(g_dict)
    print(s.Ford_Fulkerson(g))
