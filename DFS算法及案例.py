# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:41:16 2021

@author: 潘登
"""

#%%DFS详解
# 回溯算法
# 排列问题
def template(n):  # 主函数(设定他的目的是保证储存结果的result不会因为递归栈的关闭而丢失)
    result = []  # 这个保存结果

    def trace(path, choices):  # 递归函数
        '''
        path: 用于储存目前排列
        choices: 用于储存所有可选择的数字(1-n)
        '''
        if len(path) == n:  # 当path的长度等于输入的n时，结束算法，把结果加入结果列表
            result.append(list(path))
            return

        for item in choices:
            if item in path:  # 减枝操作(如果path里面有item这个数)
                continue
            path.append(item)  # 如果没有就把他加到目前的path中
            trace(path, choices)  # 递归调用自身
            path.pop()  # 撤销最后一次的选择继续向前找答案

    trace([], range(1, n+1))  # 调用函数trace
    return result

if __name__ == '__main__': 
    print(template(3))
#%%矩阵找零
import copy
def template(matrix):  # 主函数(设定他的目的是保证储存结果的result不会因为递归栈的关闭而丢失)
    result = []  # 这个保存结果

    def trace(path, choices):  # 递归函数
        '''
        path: dict 用于储存找到的零 key表示row values表示column
        choices: 矩阵matrix
        '''
        if len(path) == len(matrix):  # 当path的长度等于输入的n时，结束算法，把结果加入结果列表
            if path not in result:
                result.append(copy.deepcopy(path))
            return

        for row, items in enumerate(choices):
            for column, j in enumerate(items):
                if j == 0:  # 当元素为零时
                    if row in path.keys():
                        continue
                    if column in path.values():  # 减枝操作(如果path里面有column,说名这一列已经有零了)
                        continue
                    path[row] = column  # 如果没有就把他加到目前的path中
                    trace(path, choices)  # 递归调用自身
                    path.popitem()  # 撤销最后一次的选择继续向前找答案

    trace({}, matrix)  # 调用函数trace
    return result
if __name__ == '__main__':
    matrix = [[7, 0, 2, 0, 2],
              [4, 3, 0, 0, 0],
              [0, 8, 3, 5, 3],
              [11, 8, 0, 0, 4],
              [0, 4, 1, 4, 0]]
    print(template(matrix))
#%%将矩阵找零问题转化
'''
在[2,4],[3,4,5],[1],[3,4],[1,5]中找到1-5的一个排列（如2，3，1，4，5）
把所有可能的结果输出成列表
'''
def template(a):
    result = []

    def trace(path,choices):
        if len(path) == len(choices):
            if path not in result:
                result.append(copy.deepcopy(path))
            return

        for row, items in enumerate(choices):
            if len(path) != row:  # path的元素的index对应choice的第几行
                continue  # 要是没有这一个if判断,path中储存的元素会乱
            for j in items:
                if j in path:
                    continue
                path.append(j)
                trace(path,choices)
                path.pop()  #删掉最后一个元素
    trace([],a)
    return result

if __name__ == '__main__':
    a = [[2,4],[3,4,5],[1],[3,4],[1,5]]
    print(template(a))
#%%77. 组合
# 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        result = []
        def trace(path, choices, k):
            if len(path) == k:
                path = sorted(path)
                if path not in result:
                    result.append(path)
                return
            for items in choices:
                if items in path:
                    continue
                path.append(items)
                trace(path,choices, k)
                path.pop()
        
        trace([], range(1, n+1), k)
        return result

if __name__ == '__main__':
    print(Solution().combine(4, 2))
