class RBTreeNode:
    def __init__(self,val=None,left=None,right=None,color='black',parent=None):
        self.left = left
        self.right = right
        # color = 'red' or 'black'
        self.color = color
        self.parent = parent
        self.val = val

    def hasleft(self):
        return self.left

    def hasright(self):
        return self.right

    def isleft(self):
        return self.parent and self.parent.left == self

    def isright(self):
        return self.parent and self.parent.right == self

    def isroot(self):
        return not self.parent

    def getcolor(self):
        return self.color

    def getparent(self):
        return self.parent

    def setcolor(self,color):
        self.color = color

    def __str__(self):
        return '数据是:{}'.format(self.val) 
    
    def __iter__(self):
        if self:
            if self.hasleft():
                # 递归调用
                for elem in self.left:
                    yield elem
            yield self.val  # 把根节点放在中间，最终的输出就是从小到大
            if self.hasright():
                # 递归调用
                for elem in self.right:
                    yield elem

    def search(self,val):
        '''
        :type val: number
        :rtype: RBTreeNode
        '''
        if self.val > val:
            return self.left.search(val) if self.left else None
        elif self.val == val:
            return self
        else:
            return self.right.search(val) if self.right else None

class RBTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()
    


     # 左旋操作
    def leftRotate(self,rotRoot):
        '''
        :type rotRoot: RBTreeNode
        '''
        newRoot = rotRoot.right
        rotRoot.right = newRoot.left
        if newRoot.left:
            newRoot.left.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isroot():
            self.root = newRoot
        else:
            if rotRoot.isleft():
                rotRoot.parent.left = newRoot
            else:
                rotRoot.parent.right = newRoot
        newRoot.left = rotRoot
        rotRoot.parent = newRoot

    # 右旋操作
    def rightRotate(self,rotRoot):
        '''
        :type rotRoot: RBTreeNode
        '''
        newRoot = rotRoot.left
        rotRoot.left = newRoot.right
        if newRoot.right:
            newRoot.right.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isroot():
            self.root = newRoot
        else:
            if rotRoot.isleft():
                rotRoot.parent.left = newRoot
            else:
                rotRoot.parent.right = newRoot
        newRoot.right = rotRoot
        rotRoot.parent = newRoot

    def put(self,val):
        # 新增节点，分为两步
        #   1.普通二叉树的插入
        #   2.红黑树的平衡(旋转+变色)
        if self.root:
            self._put(val, self.root)
        else:
            self.root = RBTreeNode(val=val,color='black')
        self.size += 1

    def _put(self,val,currentNode):
        if val < currentNode.val:
            if currentNode.hasleft():
                # 如果有左节点就递归调用自身
                self._put(val, currentNode.left)
            else:
                currentNode.left = RBTreeNode(val=val, parent=currentNode)
                self.fixAfterput(currentNode.left)
        # 如果与某个节点相等那就不管他，直接return
        elif val == currentNode.val:
            return
        else:
            if currentNode.hasright():
                # 如果有右节点就递归调用自身
                self._put(val, currentNode.right)
            else:
                currentNode.right = RBTreeNode(val=val, parent=currentNode)
                self.fixAfterput(currentNode.right)
    
    # 1.2-3-4树 2节点 新增一个元素 直接合并为3-节点 -- 红黑树直接添加一个红色节点无需调整
    # 2.2-3-4树 3节点 新增一个元素 合并为4-节点 -- 红黑树有6种情况，有两种不需要调整
    #              根左左、根右右 旋转一次   根左右 根右左 旋转两次
    # 3.2-3-4树 4节点 新增一个元素 4节点中间元素升级为父节点，新增元素和剩下元素合并
    #       --  红黑树:新增节点为红色 + 爷爷节点是黑色.父节点和叔叔节点为红色-->
    #           爷爷节点变为红色(如果是root则为黑色),父亲和叔叔节点变为黑色,
    def fixAfterput(self,node):
        node.setcolor('red')
        while node and not node.isroot() and node.parent.getcolor() == 'red':
            # 如果插入的节点的父节点是爷爷节点的左侧
            if node.parent == node.parent.parent.left:
                # 对满足条件的4中情况 一分为二进行处理
                # 判断是否有叔叔节点，如果叔叔节点是红色就做变色操作
                if node.parent.parent.hasright() and node.parent.parent.right.getcolor() == 'red':
                    # 将父亲和叔叔节点变为黑色,将爷爷节点变为红色
                    node.parent.setcolor('black')
                    node.parent.parent.right.setcolor('black')
                    node.parent.parent.setcolor('red')
                    # 将修改了颜色的这一组子、父、爷节点视作一个整体，
                    # 看成一个红色节点插入到爷爷的父节点中递归处理
                    node = node.parent.parent
                else:
                    # 如果是根左右，先对‘左’进行一次左旋，就变成根左左
                    if node == node.parent.right:
                        self.leftRotate(node.parent)
                        node = node.left
                    # 如果是根左左
                    # 将父节点变为黑色
                    # 在对爷爷节点变色变为红色，最后对爷爷节点进行右旋
                    node.parent.setcolor('black')
                    node.parent.parent.setcolor('red')
                    self.rightRotate(node.parent.parent)
            else:
                # 与上面刚好相反的4中情况
                if node.parent.parent.hasleft() and node.parent.parent.left.getcolor() == 'red':
                    # 将父亲和叔叔节点变为黑色,将爷爷节点变为红色
                    node.parent.setcolor('black')
                    node.parent.parent.left.setcolor('black')
                    node.parent.parent.setcolor('red')
                    # 将修改了颜色的这一组子、父、爷节点视作一个整体，
                    # 看成一个红色节点插入到爷爷的父节点中递归处理
                    node = node.parent.parent
                else:
                    # 如果是根右左，先对‘右’进行一次右旋，就变成根右右
                    if node == node.parent.left:
                        self.rightRotate(node.parent)
                        node = node.right
                    # 如果是根右右
                    # 将父节点变为黑色
                    # 在对爷爷节点变色变为红色，最后对爷爷节点进行左旋
                    node.parent.setcolor('black')
                    node.parent.parent.setcolor('red')
                    self.leftRotate(node.parent.parent)
        # 最后处理爷爷节点为根节点的情况
        self.root.setcolor('black')

    # 删除操作: 找到要删除元素的前驱节点(或者后继节点)，替换该元素的值，
    # 然后删除前驱节点(或者后继节点), 注意删除时可能前驱和后继节点会带有子节点(但只有可能有一个)
    # 红黑树的删除： 对应回2-3-4树，一定是删除2-3-4树的叶子节点
    def delete(self,val):
        '''
        :type val:number
        :rtype: RBTreeNode
        '''
        # 先找到要删除的节点
        node = self.root.search(val)
        self._delete(node)
        return node
    
    # 删除节点3种情况
    # 1.删除叶子节点，直接删除
    # 2.删除叶节点有一个子节点，由子节点来替代
    # 3.删除叶子节点有两个子节点，那么此时需要获取对应的后继节点来替代
    def _delete(self,node):
        # 有两个子节点
        if node.hasleft() and node.hasright():
            # 找到后继节点
            subsequent = self.find_subsequent(node)   
            node.val = subsequent.val
            # 那么要删除的就是后继节点了
            node = subsequent
        # 只有一个子节点的情况,就把子节点与当前节点替换，然后删掉当前节点
        if node.hasleft() or node.hasright():
            if node.hasleft():
                temp = node.left
                temp.parent = node.parent
                if node.isroot():
                    self.root = temp
                elif node.isleft():
                    node.parent.left = temp
                else:
                    node.parent.right = temp
            else:
                temp = node.right
                temp.parent = node.parent
                if node.isroot():
                    self.root = temp
                elif node.isleft():
                    node.parent.left = temp
                else:
                    node.parent.right = temp
            # 将node节点与红黑树脱离联系等待垃圾回收
            node.left = None
            node.right = None
            node.parent = None
            # 最后调整
            if node.color == 'black':
                self.fixAfterRemove(temp)
        # 如果没有子节点
        else:
            # 先调整
            if node.color == 'black':
                self.fixAfterRemove(node)
            if node.isroot():
                self.root = None
            elif node.isleft():
                node.parent.left = None
                node.parent = None
            else:
                node.parent.right = None
                node.parent = None
    
    #  2-3-4树叶子节点的删除有三种情况：
    #         1.删除的是3节点或4节点中的元素，直接删除 
    #         2.删除的是2节点，如果兄弟节点是3-节点或者4-节点，父节点到删除节点位置，兄弟节点上升一个
    #         3.删除的是2节点，如果兄弟节点也是2-节点，那就不合法，必须重构2-3-4树
    def fixAfterRemove(self, node):
        # 情况2和3
        while not node.isroot() and node.getcolor() == 'black':
            if node.isleft():
                # 查找兄弟节点
                temp = node.parent.right
                # 如果兄弟节点的颜色是红色，说明不是真正的兄弟节点，先将(假的)兄弟节点
                # 变为黑色，再将该节点的父节点变为红色，对父节点进行左旋
                if temp.getcolor() == 'red':
                    temp.setcolor('black')
                    node.parent.setcolor('red')
                    self.leftRotate(node.parent)
                    # 这样就能得到真正的兄弟节点
                    temp = node.parent.right
                # 如果兄弟节点的子节点都是黑色（都是2-节点）情况3
                # 或者是兄弟节点就是2-节点
                if (not temp.hasleft() or temp.left.getcolor() == 'black') and (not temp.hasright() or temp.right.getcolor() == 'black'):
                    # 需要递归地调整整颗树的结构
                    temp.setcolor('red')
                    node = node.parent
                # 情况2
                else:
                    # 如果兄弟节点存在左子节点,就对兄弟节点变为红色，其子节点变为黑色
                    # 对兄弟节点做右旋操作，让其子节点上升，自己下降，变为根右右型
                    if not temp.hasright() or temp.right.getcolor()=='black':
                        temp.setcolor('red')
                        temp.left.setcolor('black')
                        self.rightRotate(temp)
                        # 兄弟节点更新
                        temp = node.parent.right
                    # 现在兄弟节点存在右子节点
                    # 就让兄弟节点去成为父节点，兄弟节点的子节点代替其位置
                    # 父节点下沉为原本node节点所在位置
                    temp.setcolor(node.parent.getcolor())
                    node.parent.setcolor('black')
                    temp.right.setcolor('black')
                    self.leftRotate(node.parent)
                    node = self.root
            else:
                 # 查找兄弟节点
                temp = node.parent.left
                # 如果兄弟节点的颜色是红色，说明不是真正的兄弟节点，先将(假的)兄弟节点
                # 变为黑色，再将该节点的父节点变为红色，对父节点进行右旋
                if temp.getcolor() == 'red':
                    temp.setcolor('black')
                    node.parent.setcolor('red')
                    self.rightRotate(node.parent)
                    # 这样就能得到真正的兄弟节点
                    temp = node.parent.left
                # 如果兄弟节点的子节点都是黑色（都是2-节点）情况3
                if (not temp.hasleft() or temp.left.getcolor() == 'black') and (not temp.hasright() or temp.right.getcolor() == 'black'):
                    # 需要递归地调整整颗树的结构
                    temp.setcolor('red')
                    node = node.parent
                # 情况2
                else:
                    # 如果兄弟节点存在右子节点,就对兄弟节点变为红色，其子节点变为黑色
                    # 对兄弟节点做左旋操作，让其子节点上升，自己下降，变为根左左型
                    if not temp.hasleft() or temp.left.getcolor()=='black':
                        temp.setcolor('red')
                        temp.right.setcolor('black')
                        self.leftRotate(temp)
                        # 兄弟节点更新
                        temp = node.parent.left
                    # 现在兄弟节点存在右子节点
                    # 就让兄弟节点去成为父节点，兄弟节点的子节点代替其位置
                    # 父节点下沉为原本node节点所在位置
                    temp.setcolor(node.parent.getcolor())
                    node.parent.setcolor('black')
                    temp.left.setcolor('black')
                    self.rightRotate(node.parent)
                    node = self.root
        # 情况1: 替代的节点是红色，直接设置为黑色即可
        node.setcolor('black')
        
    # 寻找前驱节点,前驱节点是恰好比该节点小的节点，一定在该节点左节点的最右边
    def find_pred(self, node):
        # 如果有左or右子节点
        if node.hasright() or node.hasleft():
            temp = node.left
            while temp.right:
                temp = temp.right
            return temp
        # 如果要查找叶子节点的前驱节点(这种情况不会发生在删除操作中，只是完善功能罢了)
        else:
            p = node.parent
            # 也有可能没有前驱
            while p and p.val > node.val:
                p = p.parent
            return p
     # 寻找后继节点,后继节点是恰好比该节点大的节点，一定在该节点右节点的最左边
    def find_subsequent(self, node):
        # 如果有左or右子节点
        if node.hasright() or node.hasleft():
            temp = node.right
            while temp.left:
                temp = temp.left
            return temp
        # 如果要查找叶子节点的后继节点(这种情况不会发生在删除操作中，只是完善功能罢了)
        else:
            p = node.parent
            # 也有可能没有后继
            while p and p.val < node.val:
                p = p.parent
            return p