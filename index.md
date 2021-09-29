# 图论--潘登同学的图论笔记(Python)
## 目录
+ 1.<a href="#图的数据结构">图的数据结构</a>
+ 2.<a href="#图的分类">图的分类</a>
    + <a href="#无向图">无向图</a>
    + <a href="#有向图">有向图</a>
    + <a href="#道路、回路与连通性">道路、回路与连通性</a>
+ 
## 图的数据结构 <p id="图的数据结构"></p>
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

## <p id="图的分类">图的分类</p>

### <p id="无向图">无向图(我们着重讨论简单图)</p>

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

### <p id="有向图">有向图</p>

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
> 如果形象理解连通分支呢？ 我们都学过
