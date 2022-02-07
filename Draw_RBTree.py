import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Draw_RBTree:
    def __init__(self, tree):
        self.tree = tree
    
    def show_node(self, node, ax, height, index, font_size):
        if not node:
            return
        x1, y1 = None, None
        if node.left:
            x1, y1, index = self.show_node(node.left, ax, height-1, index, font_size)
        x = 100 * index - 50
        y = 100 * height - 50
        if x1:
            plt.plot((x1, x), (y1, y), linewidth=2.0,color='b')
        circle_color = "black" if node.getcolor()=='black' else 'r'
        text_color = "beige" if node.getcolor()=='black' else 'black'
        ax.add_artist(plt.Circle((x, y), 50, color=circle_color))
        ax.add_artist(plt.Text(x, y, node.val, color= text_color, fontsize=font_size, horizontalalignment="center",verticalalignment="center"))
        # print(str(node.val), (height, index))

        index += 1
        if node.right:
            x1, y1, index = self.show_node(node.right, ax, height-1, index, font_size)
            plt.plot((x1, x), (y1, y), linewidth=2.0, color='b')

        return x, y, index

    def show_rb_tree(self, title):
        fig, ax = plt.subplots()
        left, right = self.get_left_length(), self.get_right_length(), 
        height = 2 * np.log2(self.tree.size + 1)
        # print(left, right, height)
        plt.ylim(0, height*100 + 50)
        plt.xlim(0, 100 * self.tree.size + 100)
        self.show_node(self.tree.root, ax, height, 1, self.get_fontsize())
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    def get_left_length(self):
        temp = self.tree.root
        len = 1
        while temp:
            temp = temp.left
            len += 1
        return len
    
    def get_right_length(self):
        temp = self.tree.root
        len = 1
        while temp:
            temp = temp.left
            len += 1
        return len
    
    def get_fontsize(self):
        count = self.tree.size
        if count < 10:
            return 30
        if count < 20:
            return 20
        return 16