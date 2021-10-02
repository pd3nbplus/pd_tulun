# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 08:05:25 2021

@author: 潘登
"""

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}     # connectedTo用于储存相邻节点 key是相邻节点(对象而不是名称) value是连接权值

    def addNeighbor(self, nbr, weight=0):   # 增加相邻边，默认weight为0
        '''
        input:
            nbr: Vertex object
            weight: int
        return:
            None
        '''
        self.connectedTo[nbr] = weight

    def getConnections(self):   # 获得相邻节点
        '''
        input: None
        return: Vertex object
        '''
        return self.connectedTo.keys()

    def getId(self):   # 获得节点名称
        '''
        input: None
        return: key(str)
        '''
        return self.id
    
    def getWeight(self, nbr):   # 获得相邻边数据
        '''
        input: Vertex object
        return: weight(int)
        '''
        return self.connectedTo[nbr]
    
    def __str__(self):   # 显示设置(__str__是内置方法，也就是不需要显示的使用这个方法就可以用这个函数)
        return str(self.id) + ' connectTo:' + \
               str([x.id for x in self.connectedTo])


if __name__ == '__main__':
    # 与节点相关的数据, 链接表
    a_info = {'x1':['y3', 'y5']}
    # 新建节点
    x1 = Vertex('x1')
    y3 = Vertex('y3')
    y5 = Vertex('y5')
    # 根据邻接表添加相邻节点
    x1.addNeighbor(y3, 1)
    x1.addNeighbor(y5, 5)
    # 测试定义的函数功能
    # 获取节点id
    print(x1.getId())
    print(y5.getId())
    # 获取节点的相邻节点
    print(x1.getConnections())
    print(x1)
    # 由相邻节点获得与相邻节点的权值
    print(x1.getWeight(y5))
    