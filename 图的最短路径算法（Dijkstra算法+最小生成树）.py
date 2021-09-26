# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 07:52:14 2021

@author: 潘登
"""

# %%图的最短路径算法（Dijkstra算法）无法处理权值为负的情况

import sys
import numpy as np


class Binheap():
    def __init__(self):
        self.heaplist = [(0, 0)]   # 专门用于Dijkstra算法，第一个是节点第二个是数值
        # 因为要利用完全二叉树的性质，为了方便计算，把第0个位置设成0，不用他
        '''
        完全二叉树的特性  如果某个节点的下标为i
        parent = i//2
        left = 2*i
        right = 2*i +1
        '''
        self.currentSize = 0

    def perUp(self, i):
        while i//2 > 0:
            # 如果子节点比父节点要小，就交换他们的位置
            if self.heaplist[i][1] < self.heaplist[i//2][1]:
                self.heaplist[i], self.heaplist[i//2] =\
                                        self.heaplist[i//2], self.heaplist[i]
            i = i//2

    def insert(self, k):
        self.heaplist.append(k)
        self.currentSize += 1
        self.perUp(self.currentSize)

    def delMin(self):
        # 删掉最小的那个就是删掉了根节点，为了不破坏heaplist
        # 需要把最后一个节点进行下沉，下沉路径的选择，选择子节点中小的那个进行交换
        # 先把最后一个与第一个交换顺序
        self.heaplist[1], self.heaplist[-1] =\
                                    self.heaplist[-1], self.heaplist[1]
        self.currentSize -= 1
        self.perDown(1)
        return self.heaplist.pop()

    def minChild(self, i):
        if i*2+1 > self.currentSize:
            return 2*i
        else:
            if self.heaplist[2*i][1] < self.heaplist[2*i+1][1]:
                return 2*i
            else:
                return 2*i+1

    def perDown(self, i):  # 下沉方法
        while 2*i <= self.currentSize:  # 只有子节点就比较
            min_ind = self.minChild(i)
            if self.heaplist[i][1] > self.heaplist[min_ind][1]:
                # 如果当前节点比子节点中小的要大就交换
                self.heaplist[i], self.heaplist[min_ind] =\
                                self.heaplist[min_ind], self.heaplist[i]
                i = min_ind
            else:
                break  # 如果当前节点是最小的就退出循环

    def findMin(self):
        return self.heaplist[1]

    def isEmpty(self):
        return self.heaplist == [(0, 0)]

    def size(self):
        return self.currentSize

    def buildHeap(self, alist):  # 这个alist里面装的元素是元组
        # 将列表变为二叉堆
        # 采用下沉法 算法复杂度O(N)  如果一个一个插入的话，算法复杂的将会是O(nlgn)
        # 自下而上的下沉（先下沉最底层的父节点）
        i = len(alist)//2
        self.currentSize = len(alist)
        self.heaplist = [(0, 0)] + alist
        while i > 0:
            self.perDown(i)
            i -= 1
        return self.heaplist

    def __iter__(self):
        for item in self.heaplist[1:]:
            yield item

    def __contains__(self, n):    # 判断节点是否在优先队列内（专门为prim写的）
        return n in [v[0] for v in self.heaplist]


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.color = 'white'  # 为了解决词梯问题的
        self.dist = sys.maxsize  # 无穷大
        self.pred = None

    def addNeighbor(self, nbr, weight=0):   # 增加相邻边，
        self.connectedTo[nbr] = weight   # 这个nbr是一个节点对象，不是名称

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


class DIJKSTRAGraph:
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

    def getVertices(self):   # 获取所有顶点的名称
        return self.vertList.keys()

    def __iter__(self):  # 迭代取出
        return iter(self.vertList.values())

    def Dijkstra(self, startVertex):   # 输入的stratVertex是节点的key
        startVertex = self.vertList[startVertex]
        startVertex.setDistance(0)
        startVertex.setPred(None)   # 将起始节点的前驱节点设置为None
        pq = Binheap()
        pq.buildHeap([(v, v.getDistance()) for v in self])
        while not pq.isEmpty():
            current_tuple = pq.delMin()
            for nextVertex in current_tuple[0].getConnections():
                newDistance = current_tuple[0].getDistance() +\
                                current_tuple[0].getWeight(nextVertex)
                # 如果当下一节点的dist属性大于当前节点加上边权值,就更新权值
                if newDistance < nextVertex.getDistance():
                    nextVertex.setDistance(newDistance)
            # 把更新好的值重新建队
            pq.buildHeap([(v[0], v[0].getDistance()) for v in pq])
            if not pq.isEmpty():
                # 把下一节点的前驱节点设置为当前节点
                nextVertex_set_pred = pq.findMin()[0]
                nextVertex_set_pred.setPred(current_tuple[0])

    def minDistance(self, from_key, to_key):
        self.Dijkstra(from_key)
        to_key = self.getVertex(to_key)
        min_distance = to_key.getDistance()
        while to_key.getPred():
            print(to_key.getId()+'<--', end='')
            to_key = to_key.getPred()
        print(from_key+' 最短距离为:', min_distance)

    def matrix(self, mat):    # 这里的mat用numpy传进来
        key = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(len(mat)):    # 邻接矩阵行表示from_key
            for j in range(len(mat)):  # 列表示to_key
                if i != j and mat[i, j] > 0:
                    self.addEdge(key[i], key[j], mat[i, j])

    def prim(self, startVertex):
        pq = Binheap()
        for v in self:
            v.setDistance(sys.maxsize)
            v.setPred(None)

        startVertex = self.vertList[startVertex]
        startVertex.setDistance(0)
        pq.buildHeap([(v, v.getDistance()) for v in self])
        while not pq.isEmpty():
            current_tuple = pq.delMin()
            for nextVertex in current_tuple[0].getConnections():
                # 注意这里是两顶点找最短边（因为是贪心算法）而不是找全局最短
                newWeight = current_tuple[0].getWeight(nextVertex)
                # 当这个节点在图中且新的权重比旧权重小，就更新权重，更新连接
                if nextVertex in pq and newWeight < nextVertex.getDistance():
                    nextVertex.setDistance(newWeight)
                    nextVertex.setPred(current_tuple[0])
                    # 对优先队列从新排列
                    pq.buildHeap([(v[0], v[0].getDistance()) for v in pq])
        for v in self:
            if v.getPred():
                print(f'节点{v.getId()}的前驱节点是{v.getPred().getId()}')


if __name__ == '__main__':
    DijGraph = DIJKSTRAGraph()
    inf = float('inf')
    a = np.array([[0, 1, 12, inf, inf, inf],
                  [inf, 0, 9, 3, inf, inf],
                  [inf, inf, 0, inf, 5, inf],
                  [inf, inf, 4, 0, 13, 15],
                  [inf, inf, inf, inf, 0, 4],
                  [inf, inf, inf, inf, inf, 0]])
    DijGraph.matrix(a)
    DijGraph.minDistance('A', 'F')
    DijGraph.prim('A')   # 输出最小生成树
    