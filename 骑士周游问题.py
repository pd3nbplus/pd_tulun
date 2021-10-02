# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:36:10 2021

@author: 潘登
"""
#%%骑士周游问题
from Vertex import Vertex # 导入Vertex
from Graph import Graph  # 导入之前实现的Graph

class New_Vertex(Vertex):  # 某一个具体问题的数据结构需要继承原有数据结构
    def __init__(self, key):
        super().__init__(key)
        self.color = 'white'  # 新增类属性(用于记录节点是否被走过)

    # 新增类方法, 设置节点颜色
    def setColor(self, color):
        self.color = color

    # 新增类方法, 查看节点颜色
    def getColor(self):
        return self.color

class New_Graph(Graph):  # 继承Graph对象
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

class Solution(object):
    # 构造图
    def knightGraph(self, bdSize):
        ktGraph = New_Graph()
        for row in range(bdSize):
            for col in range(bdSize):
                index = bdSize * row + col  # 当前节点索引(key)
                newPositions = self.genLegalMoves(row, col, bdSize)  # 下一个节点的集合
                for i in newPositions:
                    index_next = bdSize*i[0]+i[1]
                    ktGraph.addEdge(index, index_next, 1)
        return ktGraph

    # 判断马的走法是否合法
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

    # 深度优先搜索
    def DFS(self, g, start):
        result = []

        # 回溯
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

    # Warnsdorff算法
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