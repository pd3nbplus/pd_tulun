# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 08:38:48 2021

@author: 潘登
"""
from Vertex import Vertex

class Graph:
    def __init__(self):
        self.vertList = {}  # 这个虽然叫list但是实质上是字典
        self.numVertices = 0

    def addVertex(self, key):   # 增加节点
        '''
        input: Vertex key (str)
        return: Vertex object
        '''
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)   # 创建新节点
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, key):   # 通过key获取节点信息
        '''
        input: Vertex key (str)
        return: Vertex object
        '''
        if key in self.vertList:
            return self.vertList[key]
        else:
            return None

    def __contains__(self, n):  # 判断节点在不在图中
        '''
        input: Vertex key (str)
        return: bool
        '''
        return n in self.vertList

    def addEdge(self, from_key, to_key, cost=1):    # 新增边
        '''
        input:
            from_key: vertex key (str)
            to_key: vertex key (str)
            cost: int
        return:
            None
        '''
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

    def __len__(self):  # 查看节点个数
        return self.numVertices

if __name__ == '__main__':
    # 邻接表
    g_dict ={'1':[['2', 10], ['3', 10]],
             '2':[['3', 2], ['4', 4], ['5', 8]],
             '3':[['5', 9]],
             '4':[['6', 10]],
             '5':[['4', 6], ['6', 10]]}
    # 创建graph对象
    g = Graph()
    # 在Graph中vertList的数据结构其实就是上面这个
    # 遍历g_dict
    for from_key in g_dict:
                for to_key in g_dict[from_key]:
                    g.addEdge(from_key, to_key[0], [0, to_key[1]])
    # 测试通过节点名称获取节点
    print(g.getVertex('3'))
    # 获得所有节点的名称
    print(g.getVertices())
    # 查看节点个数
    print(len(g))
    # 判断节点在不在图中
    print('7' in g, '3' in g)
















