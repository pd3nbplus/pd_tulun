# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:58:57 2021

@author: 潘登
"""

# %%图的实现(骑士周游)
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

    def getVertices(self):   # 获取所有顶点的名称
        return self.vertList.keys()

    def __iter__(self):  # 迭代取出
        return iter(self.vertList.values())


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


class Solution(object):
    def knightGraph(self, bdSize):
        ktGraph = Graph()
        for row in range(bdSize):
            for col in range(bdSize):
                index = bdSize * row + col  # 当前节点索引(key)
                newPositions = self.genLegalMoves(row, col, bdSize)  # 下一个节点的集合
                for i in newPositions:
                    index_next = bdSize*i[0]+i[1]
                    ktGraph.addEdge(index, index_next, 1)
        return ktGraph

    def genLegalMoves(self, x, y, bdSize):
        newMoves = []
        # 马的走棋规则
        moveOffsets = [(-1, -2), (-1, 2), (-2, -1), (-2, 1),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        for i in moveOffsets:
            newX = x + i[0]
            newY = y + i[1]
            if self.legalCoord(newX, bdSize) and self.legalCoord(newY, bdSize):
                newMoves.append((newX, newY))
        return newMoves

    def legalCoord(self, x, bdSize):   # 查看落点是否越界
        if x >= 0 and x < bdSize:
            return True
        else:
            return False

    def DFS(self, g, start):
        result = []

        # 深度优先搜索
        def trace(path, g, start):   # path是探索的路径 g是图，start是起始的节点
            start.setColor('gray')  # 将正在探索的节点设置为灰色
            path.append(start)
            if len(path) == 64:   # 64是总的目标（走完棋盘）
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

    # 回溯改进Warnsdorff算法
    # 将strat 的合法移动目标棋盘格排序为：具有最少合法移动目标的格子优先搜索
    # def Warnsdorff(g, start):

    # 深度优先搜索
    def Warnsdorff(self, path, g, start):   # path是探索的路径 g是图，start是起始的节点
        start.setColor('gray')  # 将正在探索的节点设置为灰色
        path.append(start)
        temp_choice = list(self.orderByAvail(start))
        for i in temp_choice:
            if i.getColor() == 'white':
                self.Warnsdorff(path, g, i)
                if len(path) == 64:   # 64是总的目标（走完棋盘）(这与完全遍历有区别)
                    return path  # 完全遍历的判断语句就在函数前端，这个只要找到一个就行
                path.pop()
                i.setColor('white')

    # 这个函数的目的是把要探索的节点按照走的先后次序进行排序（按照下下步选择少的排在前面）
    # 相当于先验知识（启发式算法）
    def orderByAvail(self, start):
        reslist = []
        for i in start.getConnections():
            if i.getColor() == 'white':
                c = 0
                for j in i.getConnections():
                    if j.getColor() == 'white':
                        c += 1
                reslist.append((c, i))
        reslist.sort(key=lambda x: x[0])
        return [y[1] for y in reslist]

        # trace1([], g, start)
        # return result


if __name__ == '__main__':
    s = Solution()
    g = s.knightGraph(8)
    path = s.Warnsdorff([], g, g.getVertex(0))
    for i in path:
        print(i.getId(), end=' ')
