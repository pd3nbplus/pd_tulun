# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:52:32 2021

@author: 潘登
"""

#%% 词梯问题BFS算法
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

# %词梯问题:采用字典建立桶（每个桶有三个字母是相同的  比如head,lead,read
# 那么每个词梯桶内部所有单词都组成一个无向且边为1的图


def buildGraph(wordfile):
    d = {}
    g = New_Graph()
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
    print('FOOL')