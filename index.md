# 图论--潘登同学的图论笔记(Python)
## 目录
+ 1.<a href="#图的数据结构">图的数据结构</a>
+ 2.<a href="#图的分类">图的分类</a>
    + <a href="#无向图">无向图</a>
    + <a href="#有向图">有向图</a>
    + <a href="#道路、回路与连通性">道路、回路与连通性</a>
    + <a href="#欧拉图">欧拉图</a>
    + <a href="#哈密顿图">哈密顿图</a>
+ 
<p id="图的数据结构"><h2 id="图的数据结构">图的数据结构</h2></p>
  在数据结构中我们可以知道，图由节点和边构成，那么想实现图的数据结构就必然离不开节点Node，为了避免混淆，我的节点同一命名为**Vertex**；
  
  *那么实现Vertex需要一些什么属性和方法呢?*
  
  + 节点的名称
  + 节点的与什么节点连接以及连接的权值
  + 增加相邻节点(或者叫修改相邻节点的边权值)
  + 获得相邻节点(就是通过这个节点知道他的相邻节点是什么)
  + 获得节点名称(由Vertex对象获得自己的名称)
  + 获得相邻边数据(通过相邻的节点的名称知道他们之间的权值)
  + 显示设置(如果在命令行把Vertex输入, 显示的东西, 如果不写的话, 就会显示对象地址)
  
#### 话不多说 上代码！

```python
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
        return str(self.id) + 'connectTo:' + \
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
```
  实现完节点了,接下来就是实现图了;
  
  *那么图的数据结构都包括什么呢?*
  
  + 图的一种储存数据的方式就是邻接表，采取邻接表来储存节点
  + 图的节点个数
  + 在图中增加节点
  + 通过节点的key获得节点对象
  + 判断节点在不在图中(in 内置函数)
  + 新增两节点的边
  + 获取所有节点的名称
  + 迭代器的实现(通过for 循环把vertex对象取出)
  
#### 话不多说 上代码！

```python
import Vertex  # 导入上面写的代码
class Graph:
    def __init__(self):
        self.vertList = {}  # 这个虽然叫list但是实质上是字典
        self.numVertices = 0  # 记录节点个数

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
        return iter(self.vertList.values())  # 采用iter直接转成可迭代对象

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
```

**OK!!, 现在我们已经成功实现了图的数据结构, 有了数据结构我们就可以用图的特性搞点事情了！！！**

细心的同学已经注意到了, 上面刚刚出现了邻接表这个词语, 当然总是跟他相提并论的还有邻接矩阵,我们先把这两个概念说清楚；

### 图的邻接表

图的邻接表存储法。邻接表既适用于存储无向图，也适用于存储有向图；

在具体讲解邻接表存储图的实现方法之前，先普及一个"邻接点"的概念。在图中，如果两个点相互连通，即通过其中一个顶点，可直接找到另一个顶点，则称它们互为邻接点；

	邻接指的是图中顶点之间有边或者弧的存在;

例如，存储图 1a) 所示的有向图，其对应的邻接表如图 1b) 所示：

![图 1 邻接表存储有向图](http://data.biancheng.net/uploads/allimg/190106/2-1Z106140Q33H.gif)

`拿顶点 V1 来说，与其相关的邻接点分别为 V2 和 V3，因此存储 V1 的链表中存储的是 V2 和 V3 在数组中的位置下标 1 和 2;`

#### ` 简化起见 `

我喜欢把邻接表定义为一个字典， 字典的keys储存出发点， values是一个长度为 2 的list, list的第一个元素表示到达点， 第二个元素表示weight
+ 具体代码就是这样
```python
# 邻接表
g_dict ={'1':[['2', 10], ['3', 10]],
     '2':[['3', 2], ['4', 4], ['5', 8]],
     '3':[['5', 9]],
     '4':[['6', 10]],
     '5':[['4', 6], ['6', 10]]}
```
	'1':[['2', 10], ['3', 10]] 就表示顶点'1'出发到达顶点'2'的权值是10， 到达顶点'3'的权值是10;

### 图的邻接矩阵

图的邻接矩阵大家都很熟悉, 就是用一个矩阵来表示点与点之间的关系, 沿用上面邻接表的例子;
+ 具体长这样
```python
import numpy as np
np.array([[0, 10, 10, np.inf, np.inf, np.inf],
          [np.inf, 0, 2, 4, 8, np.inf],
          [np.inf, np.inf, 0, np.inf, 9, np.inf],
          [np.inf, np.inf, np.inf, 0, np.inf, 10],
          [np.inf, np.inf, np.inf, 6, 0, 10],
          [np.inf, np.inf, np.inf, np.inf, np.inf, 0]])
```

![邻接矩阵](https://s3.bmp.ovh/imgs/2021/09/42b90da3c83dd843.png)

好了！我们已经介绍完图的数据结构了, 那现在我们开始系统的学习图论吧！！

<p id="图的分类"><h2 id="图的分类">图的分类</h2></p>

<p id="无向图"><h3 id="无向图">无向图(我们着重讨论简单图)</h3></p>

#### 图的数学语言
根据前面数据结构的引入, 我们可以清楚的知道:图的由顶点, 边组成；

+ 贯彻集合论的思想, 将点写成一个点集V, 边写成一个边集E;

	> V = {v1,v2,v3,...,vn}
	> 
	> E = {e1,e2,e3,...,em}

+ 还有他们的对应关系呢！ 定义一个由E→V^2的映射r；

	> r(e) = {u,v} ⊆ V

**所以我们以后就用G = (V,E,r)来描述无向图了**

` 接下来是用来描述图的术语 `

+ 邻接

	> 假设G = (V,E,r)为无向图, 若G的两条边e1和e2都与同一个顶点相关联, 则称e1和e2是邻接的

+ 自环,重边

	> 假设G = (V,E,r)为无向图,如果存在e,有r(e)={u,v}且u=v, 那么称e为一个`自环`
	> 
	> 假设G = (V,E,r)为无向图,若G中关联同一对顶点的边多于一条,则称这些边为`重边`

+ degree(度)

	> 假设G = (V,E,r)为无向图,v∈V, 顶点v的度数dgree `ged(v)` 是G中与v关联的边的数目(说白了就是有多少个顶点跟v相连)

+ 孤立顶点, 悬挂点,奇度点,偶度点

	> 假设G = (V,E,r)为无向图,G中度数为零的顶点称为`孤立零点`,
	> 
	> 图中度数为1的顶点成为`悬挂点`,与悬挂点相关联的边称为`悬挂边`,
	> 
	> 图中度数为k的点成为`k度点`,度数为奇数的为`奇数点`,度数为偶数的称为`偶数点`

#### 简单图:不存在自环和重边的无向图
+ `图的基本定理/握手定理`
	+ *假设G = (V,E,r)为无向图,则有∑deg(v) = 2|E|; 即所有顶点的度数是边的两倍*
		> 这个的定理是显然的，因为一条边能连接两个顶点
	+ `推论`: *在任何无向图中,奇数度的点必为偶数个*
		> 这个也很好证明，因为总的度数是偶数, 那想所有顶点的度数之和为偶数, 偶数个奇数度的顶点之和才能是偶数

+ 例子:
问题如下: 假设有9个工厂, 求证:
	+ (1)在他们之间不可能每个工厂都只与其他3个工厂有业务联系
	+ (2)在他们之间不可能只有4个工厂与偶数个工厂有业务联系
		> 证明留给读者喔！

#### 在简单图范畴下的其他有特点的图
+ 零图,离散图
	> G中全是孤立顶点, 也称0-正则图
+ 正则图
	> 所有顶点度数都相同的图, 若所有顶点度数均为k, 则称k-正则图
+ 完全图
	> 任意两个顶点都有边, 也称n-正则图
+ n-立方体图
	> 其实长的就跟立方体一样,但由于图是平面的,下面给出严谨的数学定义
	> 如果图的顶点集v由集合{0,1}上的所有长为n的二进制串组成,两个顶点的相邻当且仅当他们的标号序列仅在一位数字上不同
如图绘制了多个立方体图
![n-立方体图](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.it610.com%2Fimage%2Finfo10%2F32f9c552dafb42a69ca878b4168142c1.png&refer=http%3A%2F%2Fimg.it610.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1635411402&t=065c4aed6d7281cfc8b8f3859da34c77)

+ 大家也可以尝试自己构建(推荐一个构建网络的小工具networkx,但是networkx不是本文的主要工具,故不详细讲解,感兴趣的朋友们自己了解哦)
+ 贴上代码

```python
import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
points = {'100':['101', '110', '000'],
		  '110':['111', '100', '010'],
		  '000':['001', '010', '100'],
		  '001':['101', '011', '000'],
		  '011':['001', '111', '010'],
		  '101':['111', '100', '001'],
		  '111':['110', '101', '011'],
		  '010':['110', '011', '000']}
for i in points:
	for j in points[i]:
		G.add_edge(i, j)
 # 设置节点的位置
left = ['100', '110']
middle_left = ['000', '010']
middle_right = ['001', '011']
right = ['101', '111']

options = {
		"font_size": 36,
		"node_size": 100,
		"node_color": "white",
		"edgecolors": "black",
		"edge_color": 'red',
		"linewidths": 5,
		"width": 5,
		'alpha':0.8,
		'with_labels':True
	}
pos1 = {n: (0, 5-3*i) for i, n in enumerate(left)}
pos1.update({n: (1, 4-i) for i, n in enumerate(middle_left)})
pos1.update({n: (2, 4-i) for i, n in enumerate(middle_right)})
pos1.update({n: (3, 5-3*i) for i, n in enumerate(right)})
plt.clf()
nx.draw(G, pos1, **options)
```

+ 一个3-立方体图如下:
![3-立方体图](https://i.bmp.ovh/imgs/2021/09/dc91c290b8395db5.png)

+ 圈图
	>假设V = {1,2,...,n} (n>2) E = {(u,v)| 1 ≤ u, v ≤ n, u-v ≡ 1(mods n)}
    > 则称简单图为轮图

+ 一个圈图如下：

![圈图](https://i.bmp.ovh/imgs/2021/09/eba5c9fbcf215274.png)

+ 轮图
    > 假设V = {0,1,2,...,n} (n>2), E = {(u,v)| 1 1 ≤ u, v ≤ n, u-v ≡ 1(mods n) 或者 u=0, v>0}
    > 则称简单图为轮图, 其实就是在圈图的基础上增加了一个顶点去连接所有点

+ 一个轮图如下:

![轮图](https://i.bmp.ovh/imgs/2021/09/390232a0890181b8.png)

+ 二部图(很经典)
    > 若简单图G = (V,E,r)的顶点V存在一个划分{V1, V2}使得G中任意一条边的两端分别属于V1和V2,则称G是二部图
    > 
    > 如果V1中每个顶点都与V2中每个顶点相邻, 则成为完全二部图

+ 一个二部图如下:
![二部图](https://tse1-mm.cn.bing.net/th/id/R-C.fe163db6d57378f55ef58e28b1f4b006?rik=r8P3aKwvo9VRew&riu=http%3a%2f%2fdata.biancheng.net%2fuploads%2fallimg%2f181111%2f1-1Q111131943a0.gif&ehk=JgxSy7qbYBXRWEvGoERyddXCLSGdVdAQTMdbC78cQAg%3d&risl=&pid=ImgRaw&r=0)
+ 一个完全二部图如下:
![完全二部图](https://tse3-mm.cn.bing.net/th/id/OIP-C.Ja515hHc762tn2Y6Ao7R1wAAAA?w=159&h=131&c=7&r=0&o=5&pid=1.7)

<p id="有向图"><h3 id="有向图">有向图</h3></p>

就是上面的无向图的边有了方向即边e = {u, v} 是一个有序对

+ degree(度)
    > 上面已经介绍过度了, 但在有向图中我们要把他细分一下
    > deg(v) = deg(v)(-) + deg(v)(+)
    > 
    > `出度` : 以v为起始点的有向边的数目, 记为deg(v)(+)
    > 
    > `入度`: 以v为终点的有向边的数目, 记为deg(v)(-)

+ `定理`
    > 对任意有向图G=(V,E,r), 有∑deg(v)(+) = ∑deg(v)(+) = |E|

+ 赋权图
    > 假设G=(V,E,r)为图, W是E到**R**的一个函数,则称G为赋权图,对于e∈E,w(e)称作边e的权重或者简称权
    > 
    > *其实就是之前实现图的数据结构中的cost*

#### 图的同构与子图
+ 同构
> 设G1 = (V1,E1,r1)和G2 = (V2,E2,r2)是两个无向图
> 
> 如果存在V1到v2的双射f和E1到E2的双射g, 满足对任意的e∈E1,若r1(e)={u,v},则`r2(g(e))= {f(u), f(v)}` ,则称G1和G2是同构的,并记为G1≌G2

+ 一对全等图的图像如下:
![一对全等图](https://i.bmp.ovh/imgs/2021/09/e943079ddb65d4c6.png)
![一对全等图](https://i.bmp.ovh/imgs/2021/09/8a364c1721321127.png)
    > 点集之间的双射函数f(v1) = a, f(v2) = b, f(v3) = c, f(v4) = d,
    > 
    > 边集之间的双射函数f(ei) = Ei, i = 1,2,...,6

+ 补图
> 假设G=(V,E)是一个n阶简单图
> 令E' = {(u,v)|u,v∈V, u ≠ v, {u,v}∉E}
> 则称(V, E')为G的补图, 记作 G' = (V,E')
> 
> 若G与G'同构, 则称G为自补图

+ 子图
> 设G1 = (V1,E1,r1)和G2 = (V2,E2,r2)是两个图;
> 若满足V2 ⊆ V1, E1 ⊆ E2, 及r2 = r1|E2, 即对于∀e∈E2 有`r2(e) = r1(e)` ,则称G2是G1的子图
> 
> + 当V1=V2时,称G2时G1的支撑子图(因为当把顶点看成构成这个图的支点,当支点与原本图都一样的时候就可以把他称为支撑子图)
> 
> + 当E1 ⊆ E2 或 V2 ⊆ V1 时, 称G2是G1的真子图
> 
> + 当V2 = V1 且 E2 = E1或E2 = ∅ 时, 称G2是G1的真子图

+ 导出子图
> 设G2 = (V2,E2,r2)是G1 = (V1,E1,r1)的子图,若E2={e|r1(e)={u,v}⊆V2}(即E2包含了图G中V2的所有边)

##### 构造子图的几种方法
+ 在原图中删掉点和与他关联的边,(得到的图也叫删点子图)
+ 在原图中删掉边(得到的是删边子图)

### <p id="道路、回路与连通性">道路、回路与连通性</p>
+ 道路
> 有向图G=(V,E,r)中一条道路Π 是指一个点-边序列V0,e1,v1,...,ek,vk,满足对于所有i=1,2,...,k
> r(ei) = (vi-1,vi) 称Π是从v0到vu的道路

+ 回路
> 如果上面的道路中v1 = vk, 就是回路

+ 简单道路/简单回路
> 如果道路/回路中各边互异,则称Π是简单道路/简单回路
> 
> 如果道路/回路中,除v0,vk外,每个顶点出现一次, 则称Π是初级道路/初级回路(也称圈)

+ **定理**
> 假设简单图G中每个顶点的度数都大于1,则G中存在回路

`证明`: 假设Π:v0, v1, ... ,vk-1,vk是图G中最长的一条初级道路,显然v0和vk都不会与不在道路Π上的任何顶点相邻,否则就会有一条更长的道路;

但是,deg(v0)>1,因此必定存在Π上的顶点vi与v0相邻,于是产生了回路

+ 可达
> u,v是G中两个顶点,如果u=v或G中存在u到v的道路,则称u到v是可达的

+ 连通图
> 假设G是无向图,如果图中任意两相异点之间都存在道路,则称G是连通的或者连通图

+ `连通分支`(重要)
> 假设无向图G=(V,E)的顶点集V在可达关系下的等价类为{v1, v2, ..., vk},则G关于vi的导出子图称作G的一个连通分支(其中i=1,2,...,k)
> 
> 如果形象理解连通分支呢？ 我举一个通俗一点的例子好了
> 
> 就拿国家与国家之间的关系来理解好了, 一个国家可能有很大的疆域也可能只有很小的疆域;
> 但是在与其他的国家进行外交的时候,无论这个国家再小,都是以一个独立的身份与别国外交;
> 而无论某个国家的某个地区的实力再强大,也不能独立的与别国外交(即一个国家只会一个身份去与别国外交);
> 
> 而连通分支就是来描述这样一种情况, 在一个网络中, 有很多的顶点, 但是很多顶点都有共性(比如他们500年前是一家), 那么就可以把他们看成一个顶点,这样就方便分析了
> 
> 而这个共性具体来说就是; `互相可达`(就是这一堆的任何顶点都可以达到另一个顶点(也要在网络内))

> 后面还会讲到怎么求解连通分支的kosaraju算法
+ 一个包含三个连通分支的图如下:
![连通分支](https://tse1-mm.cn.bing.net/th/id/R-C.b83b407a9a05dca4f8ec6be475ea0833?rik=0kSsQRXtBwfiAA&riu=http%3a%2f%2fblog.kongfy.com%2fwp-content%2fuploads%2f2015%2f03%2fB0842D09-98C3-4F8C-B551-315CBAC0874E.jpg&ehk=bB1q8GegpNLKLu%2bFcZrOknJZJ3mq47zCIrzkdMVPeME%3d&risl=&pid=ImgRaw&r=0)

+ 割边(桥)
> 假设G=(V,E,r)是连通图, 若e∈E,且G-e(从G中去掉边e)不连通,则称e是图G中的一条割边或桥
> 
> 如上图的7→9就是桥, 因为去掉之后图就不连通了

+ `定理`
> 连通图中的边e是桥, 当且仅当e不属于图中任意的一条回路;(因为回路一定是有来又回,如果桥在回路里面,那把他去掉图仍然是连通的)

+ 有向图的连通
> 假设G是有向图,如果忽略图中边的方向后得到的无向图是连通的,则称G为连通的;否则就是不连通的

+ 强连通, 单向连通
> 有向连通图G, 对于图中任意两个顶点u和v,u到v 和 v到u都是可达的,则称G为强连通的;
> 如果图中任意两个顶点u和v, u到v 和 v到u至少之一是可达的,则称G为单向连通的

+ 有向无环图(DAG)
> 有向图G,如果图中不存在有向回路,则称G为有向无环图


<p id="欧拉图"><h3 id="欧拉图">欧拉图</h3></p>

+ 欧拉图

其实欧拉图的来源是那个柯尼斯堡七桥问题,这不是本文重点,感兴趣的同学自行百度喔

> 通过图G中每条边一次且仅一次的道路称作该图的一条欧拉道路;(相当于一笔画)
> 
> 通过图G中每条边一次且仅一次的回路称作该图的一条欧拉回路;(相当于头尾相连的一笔画)
> 
> 存在欧拉回路的图称作欧拉图

+ 一些例子如下:

![欧拉图](https://www.pianshen.com/images/806/d8c8d30da77eec0b7c3491b1d0b5a766.png)

+ `欧拉图的充要条件`
> 无向图G是欧拉图当且仅当G是连通的而且所有顶点都是`偶数度`

`证明`: (必要性)假设G是欧拉图,即图中有欧拉回路,即对于任意一个顶点,有入必有出,因此每个顶点都是偶数度;

(充分性)设G的顶点是偶数度, 采用构造法证明欧拉回路的存在性: 

从任意点v0出发,构造一条简单回路C, 因为各顶点的度数都是偶数, 因此一定能够回到v0, 构造简单回路;**(!!!想清楚了再往下看!!!)**

下一步,在上面构造的简单回路中挑一点再剩余的点和边中构造简单函数,重复该过程,直到不能再构造为止.

+ `定理`
> 无向图G存在欧拉道路当且仅当G是连通的而且G中奇数度顶点不超过两个

`证明`: (必要性)G中一条欧拉道路L,在L中除起点和终点之外,其余每个顶点都与偶数条边(一条入,一条出)相关联. 因此, G中至多有两个奇数度的顶点

(充分性) ①若G中没有奇数度的顶点u,v,由上面定理,存在欧拉回路,显然是欧拉道路

②若G中有两个奇数度顶点u,v,那么连接uv,根据上面定理,可知存在一个欧拉回路C,从C中去掉边uv,得到顶点为u,v的一条欧拉道路(当我证到这的时候,应该有妙蛙妙蛙的声音)

#### `(算法)`构造欧拉回路 Fleury(G)

前提:得存在

输入: 至多有两个奇数度顶点的图G=(V,E,r)

输出:以序列的形式呈现欧拉回路/道路Π

算法流程:

    ① 选择图中一个奇数度的顶点v∈V,如果图中不存在奇数度顶点,则任取一个顶点v, 道路序列Π←v

    ② while |E| ≠ 0
        if 与v关联的边多余一条,则任取其中不是桥的一条边e:
        else 选择该边e:
            假设e的两个端点是v,u, Π←Π·e·u, v←u
            删除边e及孤立顶点(如果有)
    ③ 输出序列

+ `定理` 
> 假设连通图G中有k个度为奇数的顶点,则G的边集可以划分成 k/2 条简单道路,而不能分解为(k/2 - 1)或更少条道路

`证明`: (前半句) 由握手定理知k必为偶数.将这k个顶点两两配对,然后增添互不相邻的k/2条边,得到一个无奇数度顶点的连通图G',

那么由前面的定理知,G'存在欧拉回路C,在C中删去增添的k/2条边, 便得到了k/2条简单道路;

(后半句) 假设图G的边集可以分为q条简单道路,则在图G中添加q条边可以得到欧拉图G',因此图中所有顶点都是偶数度,而每添加一条边最多可以把两个奇数度顶点变为偶数度,即2q≥k

+ 无向图的欧拉道路--推广-->有向图
`定理`
> 有向图G中存在欧拉道路当且仅当G是连通的,而且G中每个顶点的入度都等于出度;
> 
> 有向图G中存在欧拉道路,但不存在欧拉回路当且仅当G是连通的;
> 
> 除两个顶点外,每个顶点的入度都等于出度,而且这两个顶点中一个顶点的入度比出度大1,另一个的入度比出度小1;

### <p id="哈密顿图"><h3 id="哈密顿图">哈密顿图</h3></p>

+ 哈密顿图
> 图中含有`哈密顿回路`的图

+ 哈密顿道路
> 通过图G中每个顶点一次且仅一次的道路称作该图的一条哈密顿道路;

+ 哈密顿回路
> 通过图G中每个顶点一次且仅一次的回路称作该图的一条哈密顿回路;

+ `注意`
> 1.图G中存在自环/重边不影响哈密顿回路/道路的存在性
> 
> 2.哈密顿图中一定不存在悬挂边
> 
> 3.存在哈密顿道路的图中不存在孤立顶点

+ 例子: 科学家排座问题
> 由七位科学家参加一个会议,已知A只会讲英语;B会讲英、汉;C会讲英、意、俄;D会日、汉;E会德、意;
> 
> F会法、日、俄;G会讲德、法;安排他们坐在一个圆桌,使得相邻的科学家都可以使用相同的语言交流;

这就是一个很经典的哈密顿图的例子,转换一下,就是把他们之间的关系用图来表示,再在图中找一条哈密顿回路;

将讲相同语言的科学家之间连一条边,构造图,如下:

![科学家排座图](https://i.bmp.ovh/imgs/2021/09/aabd57aee6694326.png)

`构图代码`
```python
# %%科学家排座问题
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
a_dict = {'A':['English'],
          'B':['English', 'Chinese'],
          'C':['English', 'Italian', 'Russian'],
          'D':['Japanese', 'Chinese'],
          'E':['German', 'Italian'],
          'F':['French', 'Japanese', 'Russian'],
          'G':['German', 'French']}

# 用于储存相邻节点, key是语言， value是people_name
neighbors_dict = {}
for people_name in a_dict:
    G.add_node(people_name)
    for language in a_dict[people_name]:
        # 如果这个语言在neighbors_dict, 那么就把会这门语言的其他人
        # 与这个人之间连接一条边， 否则就把这个人会的语言加到dict中
        if language in neighbors_dict:
            for people_name_exist in neighbors_dict[language]:
                G.add_edge(people_name_exist, people_name, language=language)
            neighbors_dict[language].append(people_name)
        else:
            neighbors_dict[language] = [people_name]

options = {
        "font_size": 36,
        "node_size": 100,
        "node_color": "white",
        "edgecolors": "white",
        "edge_color": 'red',
        "linewidths": 5,
        "width": 5,
        'alpha':0.8,
        'with_labels':True
    }
pos = nx.spring_layout(G)
plt.clf()
nx.draw(G, pos, **options)

# 显示edge的labels
edge_labels = nx.get_edge_attributes(G, 'language')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='blue')
```


再寻找一条回路经过所有顶点,这里我们采用DFS(深度优先搜索)

`DFS深度优先搜索`
> DFS的本质是暴力搜索算法,也称回溯算法,就是一种在找不到正确答案的时候回退一步,再继续向前的算法

算法的思想很简单,但是实际操作可能会有点不好理解;下面我们通过几个小案例来深入理解DFS算法

+ 例子1：
> 全排列问题： 给出一个数字n，要求给出1-n的全排列,输出结果为列表

如给定数字3, 那结果就是[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

上代码!!!

```python
# 回溯算法
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
```

观察到输出结果与上面预期一致,(回溯算法是后面很多算法的基础)

+ 例子2:
> 给定一个5*5的*矩阵,里面有零元素,要求在矩阵里面5个0,使得每一行每一列都有零
> 
> [[7, 0, 2, 0, 2],[4, 3, 0, 0, 0],[0, 8, 3, 5, 3],[11, 8, 0, 0, 4],[0, 4, 1, 4, 0]]

上代码!!!

```python
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
```

可以看到最终返回了两个结果:

> [{0: 1, 1: 2, 2: 0, 3: 3, 4: 4}, {0: 1, 1: 3, 2: 0, 3: 2, 4: 4}]
> 
> 表示的就是第0行选择第1列,第1行选择第2列,第2行选择第0列,第3行选择第3列,第4行选择第4列;带回矩阵验证发现符合条件

其实可以把例子2中的问题转换一下,又变成一个排列问题,仍然采用相同的方法求解

+ 例子2(变形):
> 在[2,4],[3,4,5],[1],[4,5],[1,5]中找到1-5的一个排列（如2，3，1，4，5）
> 
> 把所有可能的结果输出成列表

上代码!!!

```python
#%%将矩阵找零问题转化
import copy
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
```

可以看到结果是一致的

+ 例子3:给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合

上代码!!!
```python
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
```

可以看到结果符合题意

经过上面几个例子的引入,同学们对回溯算法应该比较了解了,现在我们来完成科学家的排座问题吧!

#### **科学家排座问题**

+ 修改数据结构,因为
