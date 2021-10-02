# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:39:59 2021

@author: 潘登
"""

#%%最大权匹配(指派问题)
import copy
def template(matrix):  # 主函数(设定他的目的是保证储存结果的result不会因为递归栈的关闭而丢失)
    result = []  # 这个保存结果
    profit = 0
    def trace(path, choices):  # 递归函数
        '''
        path: dict 用于储存找到的零 key表示row values表示column
        choices: 矩阵matrix
        '''
        nonlocal result,profit
        if len(path) == len(matrix):  # 当path的长度等于输入的n时，结束算法，把结果加入结果列表
            # 计算profit
            temp = 0
            for row in path:
                temp += choices[row][path[row]]
            # 比较当前方案与此前最优方案
            if temp > profit:
                result = copy.deepcopy(path)
                profit = temp
            return

        for row in range(len(choices)):
            for column in range(len(choices)):
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
    C = [[12,7,9,7,9],
         [8,9,6,6,6],
         [7,17,12,14,12],
         [15,14,6,6,10],
         [4,10,7,10,6]]
    print(template(C))
#%%最小权匹配(指派问题)
def template(matrix):  # 主函数(设定他的目的是保证储存结果的result不会因为递归栈的关闭而丢失)
    result = []  # 这个保存结果
    profit = 1e10
    def trace(path, choices):  # 递归函数
        '''
        path: dict 用于储存找到的零 key表示row values表示column
        choices: 矩阵matrix
        '''
        nonlocal result,profit
        if len(path) == len(matrix):  # 当path的长度等于输入的n时，结束算法，把结果加入结果列表
            # 计算profit
            temp = 0
            for row in path:
                temp += choices[row][path[row]]
            # 比较当前方案与此前最优方案
            if temp < profit:
                result = copy.deepcopy(path)
                profit = temp
            return

        for row in range(len(choices)):
            for column in range(len(choices)):
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
    C = [[12,7,9,7,9],
         [8,9,6,6,6],
         [7,17,12,14,12],
         [15,14,6,6,10],
         [4,10,7,10,6]]
    print(template(C))