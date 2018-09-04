1、二维数组的查找
''' 思路
 矩阵是有序的，从左下角来看，向上数字递减，向右数字递增，
 因此从左下角开始查找，当要查找数字比左下角数字大时。右移
 要查找数字比左下角数字小时，上移
'''
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        x = len(array) - 1
        y = len(array[0]) - 1
        i,j = x,0
        while i >= 0 and j <= y:
            if target > array[i][j]:
                j += 1
            elif target < array[i][j]:
                i -= 1
            else:
                return True
        return False
		
2、替换空格
#思路：从前往后遍历 开辟新列表来存储
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        r = ''
        for i in range(len(s)):
            if s[i] == ' ':
                r+='%20'
            else:
                r+=s[i]
        return r
3、从头到尾打印链表
#递归
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        if listNode is None:
            return []
        return self.printListFromTailToHead(listNode.next)+[listNode.val]
		
4、重构二叉树
'''
  先序遍历第一个位置肯定是根节点node，
 
  中序遍历的根节点位置在中间p，在p左边的肯定是node的左子树的中序数组，p右边的肯定是node的右子树的中序数组
 
  另一方面，先序遍历的第二个位置到p，也是node左子树的先序子数组，剩下p右边的就是node的右子树的先序子数组
 
  把四个数组找出来，分左右递归调用即可
'''
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        elif len(pre) == 1:
            return TreeNode(pre[0])
        else:
            ans = TreeNode(pre[0])
            ans.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1], tin[:tin.index(pre[0])])  
            ans.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:], tin[tin.index(pre[0])+1:])
            return ans 

5、两个栈实现队列
#思路：栈A用来作入队列 栈B用来出队列，当栈B为空时，栈A全部出栈到栈B,栈B再出栈（即出队列）
class Solution:
    def __init__(self):
        self.stackA = []
        self.stackB = []
    def push(self, node):
        self.stackA.append(node)
        # write code here
    def pop(self):
        # return xx
        if self.stackB:
            return self.stackB.pop()
        elif not self.stackA:
            return None
        else:
            while self.stackA:
                self.stackB.append(self.stackA.pop())
            return self.stackB.pop()

6、输出旋转数组的最小数字
#旋转之后的数组实际上可以划分成两个有序的子数组：前面子数组的大小都大于后面子数组中的元素
class Solution:
    def minNumberInRotateArray(self,rotateArray):
        left = 0
        right = len(rotateArray) - 1
        mid = (left + right) / 2
        while left < right - 1:  #折半查找
            if (rotateArray[left] == rotateArray[right]) and (rotateArray[right] == rotateArray[mid]):
                return self.minInorder(rotateArray, left, right) #顺序查找
            if (rotateArray[left] <= rotateArray[mid]):
                left = mid
            else:
                right = mid
            mid = (left + right) / 2
        return rotateArray[right]
    #顺序查找数组里的最小值
    def minInorder(self,rotateArray,left,right):
        tmp = rotateArray[0]
        length = len(rotateArray)
        for i in range(1,length):
            if rotateArray[i] < tmp:
                tmp = rotateArray[i]
        return tmp
        
7、斐波那契数列
class Solution:
    def Fibonacci(self, n):
        return self.fib(n, 0, 1)
    def fib(self, n,acc1,acc2):
        if (n == 1):
            return acc2;
        if (n == 0):
            return 0;

        return self.fib(n-1,acc2,acc1 + acc2)  #尾递归

8、跳台阶
#还是斐波那契数列问题
class Solution:
    def jumpFloor(self, number):
        # write code here
        return self.f(number, 1, 2)
    def f(self, number, acc1, acc2):
        # write code here
        if number == 1:
            return acc1
        if number == 2:
            return acc2
        return self.f(number-1, acc2, acc1+acc2)
		
9、变态跳台阶
'''
因为n级台阶，第一步有n种跳法：跳1级、跳2级、到跳n级
跳1级，剩下n-1级，则剩下跳法是f(n-1)
跳2级，剩下n-2级，则剩下跳法是f(n-2)
所以f(n)=f(n-1)+f(n-2)+...+f(1)
因为f(n-1)=f(n-2)+f(n-3)+...+f(1)
所以f(n)=2*f(n-1)
'''
class Solution:
    def jumpFloorII(self, number):
        # write code here
        return self.f(number, 1, 2)
    def f(self, number, acc1, acc2):
        # write code here
        if number == 1:
            return acc1
        if number == 2:
            return acc2
        return self.f(number-1, acc2, 2*acc1+acc2)

10、矩形覆盖
#依旧是斐波那契数列
class Solution:
    def rectCover(self, number):
        # write code here
        if number <= 0:
            return 0
        else:
            return self.fic(number, 1, 2)
    def fic(self, number, acc1, acc2):
        if number == 1:
            return acc1
        elif number == 2:
            return acc2
        else:
            return self.fic(number-1, acc2, acc1+acc2)
			
11、二进制中1的个数
'''
如果一个整数不为0，那么这个整数至少有一位是1。
如果把这个整数减1，那么原来处在整数最右边的1就会变为0，
原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。
然后做n & n-1 消掉后面的1
'''
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n < 0:
            n = n & 0xffffffff #转zhen正数
        while n:
            count += 1
            n = (n - 1) & n
        return count

12、数值的整数次方		
class Solution:
    def Power(self, base, exponent):
        # write code here
        flag = True
        if exponent < 0:
            exponent = -exponent
            flag = False
        if exponent == 0:
            return 1
        print(exponent)
        if base == 0 and exponent < 0:
            return None
        if exponent % 2 == 0:
            result = self.get_Power(base, exponent)
        else:
            result = base * self.get_Power(base, exponent-1)
        if flag == False:
            return 1.0 / result
        else:
            return result
    
    def get_Power(self, base, exponent):
        if exponent == 1:
            return base
        else:
            return base * self.get_Power(base, exponent / 2)
			
13、调整数组顺序使奇数位于偶数前面
#开辟新空间来存
class Solution:
    def reOrderArray(self, array):
        # write code here
        even = []
        odd = []
        for i in range(len(array)):
            if array[i] % 2 == 0:
                even.append(array[i])
            else:
                odd.append(array[i])
        return odd+even

14、表中倒数第k个结点
#两个指针 保持k的距离
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        n = 0
        tmp = head
        while head:
            if n != k:
                n += 1
            else:
                tmp = tmp.next
            head = head.next
        if n == k:
            return tmp
        else:
            return None
			
15、反转链表
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead is None:
            return None    
        tmp = ListNode(pHead.val)
        #tmp.next = None
        while pHead.next != None:
            #print('1')
            pHead = pHead.next
            new = ListNode(pHead.val)
            new.next = tmp
            tmp = new
            
        while tmp != None:
            return tmp
            tmp = tmp.next			

16、合并两个排序的链表
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        p = ListNode(0)
        p.next = None
        q = p
        while pHead1 != None and pHead2 != None:
            if pHead1.val < pHead2.val:
                tmp = ListNode(pHead1.val)
                tmp.next = p.next
                p.next = tmp
                p = p.next
                pHead1 = pHead1.next
            else:
                tmp = ListNode(pHead2.val)
                tmp.next = p.next
                p.next = tmp
                p = p.next
                pHead2 = pHead2.next
        if pHead2 != None:
            p.next = pHead2
        if pHead1 != None:
            p.next = pHead1
        return q.next