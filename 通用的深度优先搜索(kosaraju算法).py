# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 21:33:51 2021

@author: 潘登
"""

# %%通用的深度优先搜索(kosaraju算法)
import sys


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.color = 'white'
        self.dist = sys.maxsize  # 无穷大
        self.pred = None
        self.disc = 0   # 发现时间
        self.fin = 0  # 结束时间

    def addNeighbor(self, nbr, weight=0):   # 增加相邻边，
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

    def setDiscovery(self, dtime):
        self.disc = dtime

    def setFinish(self, ftime):
        self.fin = ftime

    def getFinish(self):
        return self.fin

    def getDiscovery(self):  # 设置发现时间
        return self.disc


class DFSGraph:
    def __init__(self):
        self.vertList = {}  # 这个虽然叫list但是实质上是字典
        self.numVertices = 0
        self.time = 0   # DFS图新增time 用于记录执行步骤

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

    def getVertices(self):   # 获取所有顶点的名称
        return self.vertList.keys()

    def __iter__(self):  # 迭代取出
        return iter(self.vertList.values())

    def dfs(self):
        for v in self:
            v.setColor('white')   # 先将所有节点设为白色
            v.setPred(-1)

        for v in self:
            if v.getColor() == 'white':
                self.dfsvisit(v)  # 如果还有未包括的顶点,则建立森林

    def dfsvisit(self, stratVertex):
        stratVertex.setColor('gray')
        self.time += 1   # 记录步数
        stratVertex.setDiscovery(self.time)

        for v in stratVertex.getConnections():
            if v.getColor() == 'white':
                v.setPred(stratVertex)   # 把下一个节点的前驱节点设为当前节点
                self.dfsvisit(v)  # 递归调用自己
        stratVertex.setColor('black')  # 把当前节点设为黑色
        self.time += 1   # 设为黑色表示往回走了，所以步数加一
        stratVertex.setFinish(self.time)

    def kosaraju(self):  # kosaraju划分强连通分支
        self.dfs()  # 第一步调用DFS，得到节点的Finish_time
        # 第二步将图转置
        self.transposformed()  # 将图转置
        # 对转置的图调用DFS,但不能直接调用
        num = self.numVertices
        max_finish = 0
        while num > 0:
            for v in self:
                # 得到最大的Finish_time
                if v.getColor() == 'black' and v.fin >= max_finish:
                    max_finish = v.fin

            for v in self:
                # 按照Finish_time从大到小组成深度优先森林
                if v.fin == max_finish:
                    self.kosaraju_dfsvisit(v)
                    print('其中一个强联通分支是:')
                    for v in self:
                        if v.getColor() == 'gray':  # 将灰色的都返回
                            print(v.getId(), end=' ')
                            v.setColor('white')  # 将颜色设为白色
                            num -= 1   # 记录还剩多少节点
            max_finish = 0

    def transposformed(self):
        Edge_tuples = []  # 里面装 某节点-指向->相邻节点和相邻边
        for v1 in self:  # 把所有节点取出来
            for v2 in self:  # 两两交换边
                if v2 in v1.getConnections():
                    Edge_tuples.append((v1, v2, v1.getWeight(v2)))
            v1.connectedTo = {}  # 把v1的全部变成空
        for v3 in Edge_tuples:
            current_Vertex = v3[1]  # current_Vertex 是原本被指向的节点
            current_Vertex.addNeighbor(v3[0], v3[2])

    def kosaraju_dfsvisit(self, stratVertex):
        # 写一个专门用于kosaraju逆序的dfs
        stratVertex.setColor('gray')  # 把color从黑色转为灰色
        for v in stratVertex.getConnections():
            if v.getColor() == 'black':
                v.setPred(stratVertex)   # 把下一个节点的前驱节点设为当前节点
                self.kosaraju_dfsvisit(v)  # 递归调用自己


if __name__ == '__main__':
    g = DFSGraph()
    g.addEdge('A', 'B')
    g.addEdge('B', 'E')
    g.addEdge('B', 'C')
    g.addEdge('C', 'F')
    g.addEdge('E', 'A')
    g.addEdge('E', 'D')
    g.addEdge('D', 'G')
    g.addEdge('D', 'B')
    g.addEdge('G', 'E')
    g.addEdge('F', 'H')
    g.addEdge('H', 'I')
    g.addEdge('I', 'F')
    g.kosaraju()
