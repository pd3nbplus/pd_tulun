# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:34:18 2021

@author: 潘登
"""

#%% 科学家排座问题
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

    # 新增方法
    def addUndirectedEdge(self, key1, key2, cost=1):
        self.addEdge(key1, key2, cost)
        self.addEdge(key2, key1, cost=1)

class Solution():   # 解决具体问题的类对象
    # 对科学家们创建图,之前有过相关操作
    def createGraph(self, a_dict):
        '''
        input: a_dict type: dict, key: people_name, value: language
        return: graph
        '''
        # 创建图
        graph = New_Graph()
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

    # 关键函数DFS,有
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
                   if i.getColor() == 'white':  # 如果与他相邻的顶点还是白色,就探索他
                       trace(path, g, i)
                       path.pop()
                       i.setColor('white')    # 将已经撤销的结果的节点设置回白色

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
            print(i.getId(), end='-')
        print('\t')

