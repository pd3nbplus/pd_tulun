# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 08:53:52 2021

@author: 潘登
"""

# %%图的实现
# Vertex包含了顶点信息，以及顶点连接边的信息
import sys


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.color = 'white'  # 为了解决词梯问题的
        self.dist = sys.maxsize  # 无穷大
        self.pred = None

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

    def getVertices(self):   # 获取所有顶点的名称
        return self.vertList.keys()

    def __iter__(self):  # 迭代取出
        return iter(self.vertList.values())


# %词梯问题:采用字典建立桶（每个桶有三个字母是相同的  比如head,lead,read
# 那么每个词梯桶内部所有单词都组成一个无向且边为1的图


def buildGraph(wordfile):
    d = {}
    g = Graph()
    wfile = open(wordfile, 'r')
    # 创建桶，每个桶中只有一个字母是不同的
    for line in wfile:
        word = line[:-1]
        for i in range(len(word)):   # 每一个单词都可以属于4个桶
            bucket = word[:i] + '_' + word[i+1:]
            if bucket in d:
                d[bucket].append(word)
            else:
                d[bucket] = [word]
    # 在桶内部建立图
    for bucket in d.keys():
        for word1 in d[bucket]:
            for word2 in d[bucket]:
                if word1 != word2:
                    g.addEdge(word1, word2)
    return g


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


def BFS(g, start):   # g是图，start是起始的节点
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


def traverse(y):
    x = y
    while (x.getPred()):
        print(x.getId())
        x = x.getPred()
    # print(x.getPred())


if __name__ == '__main__':
    wordgraph = buildGraph('fourletterwords.txt')
    BFS(wordgraph, wordgraph.getVertex('FOOL'))
    traverse(wordgraph.getVertex('SAGE'))

























