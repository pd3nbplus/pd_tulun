# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:14:04 2021

@author: 潘登
"""

#%%最大匹配
from Vertex import Vertex  # 导入Vertex

class New_Vertex(Vertex):
    def __init__(self,key):
        super().__init__(key)
        self.labels = 0   # 新增类属性(节点标记)

    # 新增类方法 设置标记
    def setlabel(self, label):
        self.labels = label

    # 新增类方法 获得标记值
    def getlabel(self):
        return self.labels

class Bin_Graph():
    # 这里因为要划分X与Y的点集，所以重写一个图的数据解构,与原本的Graph区别也不太大
    def __init__(self):
        self.x_vertList = {}  # 储存x的顶点
        self.y_vertList = {}  # 储存y的顶点
        self.numVertices = 0

    def addVertex(self, key, label):   # 增加节点
        self.numVertices = self.numVertices + 1
        newVertex = New_Vertex(key)   # 创建新节点
        if label == 'x':
            self.x_vertList[key] = newVertex
        elif label == 'y':
            self.y_vertList[key] = newVertex
        return newVertex

    def getVertex(self, key):   # 通过key获取节点信息
        if key in self.x_vertList:
            return self.x_vertList[key]
        elif key in self.y_vertList:
            return self.y_vertList[key]
        else:
            return None

    def __contains__(self, n):  # 判断节点在不在图中
        return n in self.x_vertList or n in self.y_vertList

    def addEdge(self, from_key, to_key, label, cost=1):    # 新增边
        if (from_key not in self.x_vertList) and (from_key not in self.y_vertList):
            return print('from_key不在图中！！')

        if (to_key not in self.x_vertList) and (to_key not in self.y_vertList):
            return print('to_key不在图中！！')
        # 调用起始顶点的方法添加邻边
        if label == 'x':
            self.x_vertList[from_key].addNeighbor(self.y_vertList[to_key], cost)
        elif label == 'y':
            self.y_vertList[from_key].addNeighbor(self.x_vertList[to_key], cost)

    def addUndirectedEdge(self, key1, key2, cost=1):
        self.addEdge(key1, key2, 'x', cost)
        self.addEdge(key2, key1, 'y', cost=1)

    def get_x_Vertices(self):   # 获取所有x顶点的名称
        return self.x_vertList.keys()

    def get_y_Vertices(self):   # 获取所有y顶点
        return self.y_vertList.keys()

    def __iter__(self):  # 迭代取出
        return iter(list(self.x_vertList.values()) + list(self.y_vertList.values()))

    def __len__(self):
        return self.numVertices


class Solution():   # 解决具体问题的类对象
    # 创建二部图
    def createBinpartiteGraph(self, a_dict):
        '''
        
        Parameters
        ----------
        a_dict : dict对象,keys用于存放X,Values用于存放Y

        Returns
        -------
        bin_Graph : Bin_Graph二部图对象

        '''
        bin_Graph = Bin_Graph()
        for i in a_dict:
            bin_Graph.addVertex(i, 'x')
            for j in a_dict[i]:
                # 如果y不在图中就先添加
                if j not in bin_Graph.get_y_Vertices():
                    bin_Graph.addVertex(j, 'y')
                bin_Graph.addUndirectedEdge(i, j)
        return bin_Graph

    # 求出邻接顶点集
    def Ng(self, w, g):
        '''
        w:图g的一个顶点集的子集, 其中元素表示的是节点的名称
        return: w的邻接顶点集, 其中的元素表示的是节点的名称
        '''
        neighbors = []
        for i in w:
            temp = list(g.getVertex(i).getConnections())
            neighbors += [i.getId() for i in temp]
        return neighbors

    # 算法实现(最大匹配)
    def MaxMatch(self, g):
        '''
        g: 二部图
        return: 最大匹配m 字典，keys表示y values表示x
        '''
        m = {}
        # 如果x的顶点里面有0标记的
        while (0 in [g.getVertex(i).getlabel() for i in g.get_x_Vertices()]):
            # 选择一个标记为0的x
            for x in g.get_x_Vertices():
                if g.getVertex(x).getlabel()  == 0:
                    # U中装的是节点的keys
                    U = [x]
                    V = []
                    neibor = self.Ng(U, g)
                    while True:  # 这里设置while循环因为下面要回来这里
                        turn130while = False
                        # 如果Ng(U, g)=V, 则x无法作为一条可增广道路的端点，将x标记为2
                        if neibor == V:
                            # 将x标记为2
                            g.getVertex(U[0]).setlabel(2)
                            # 退回到130行while
                            turn130while = True
                            break
                        else:
                            turn130while = False
                            # 选择y属于neibor - V
                            diff = [i for i in neibor if i not in V]
                            for y in diff:
                                if g.getVertex(y).getlabel() == 1:
                                    # 修改U，V
                                    if m[y] not in U:
                                        U.append(m[y])
                                    if y not in V:
                                        V.append(y)
                                    # 退回到138行while循环
                                    break
                                else:
                                    turn130while = True
                                    # 把x，y加到可增广道路
                                    m[y] = x
                                    # 修改x，y的标记
                                    g.getVertex(x).setlabel(1)
                                    g.getVertex(y).setlabel(1)
                                    # 退回到130while循环 需要两步
                                    break
                            if turn130while:
                                break
                    if turn130while:
                        break
        return m

if __name__ == '__main__':
    a_dict = {'x1':['y3', 'y5'],
                            'x2':['y1', 'y2', 'y3'],
                            'x3':['y4'],
                            'x4':['y4', 'y5'],
                            'x5':['y5']}
    s = Solution()
    g = s.createBinpartiteGraph(a_dict)
    print(s.MaxMatch(g))