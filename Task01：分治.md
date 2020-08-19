# Task01：分治

[TOC]



> MapReduce(分治算法的应用) 是 Google 大数据处理的三驾马车之一，另外两个是 GFS 和 Bigtable。它在倒排索引、PageRank 计算、网页分析等搜索引擎相关的技术中都有大量的应用。
> 尽管开发一个 MapReduce 看起来很高深，感觉遥不可及。实际上，万变不离其宗，它的本质就是分治 算法思想，分治算法。如何理解分治算法?为什么说 MapRedue 的本质就是分治算法呢?

## 1、主要思想

分治算法的主要思想是将原问题递归地分成若干个子问题，直到子问题满足边界条件，停止递归。将子问题逐个击破(一般是同种方法)，将已经解决的子问题合并，最后，算法会层层合并得到原问题的答案。

#### 1.1 分治算法的步骤

- 分:递归地将问题分解为各个的子问题(性质相同的、相互独立的子问题); 
- 治:将这些规模更小的子问题逐个击破; 
- 合:将已解决的子问题逐层合并，最终得出原问题的解。

![image-20200819153911945](/Users/taolong/Library/Application Support/typora-user-images/image-20200819153911945.png)

#### 1.2 分治法适用的情况

- 原问题的计算复杂度随着问题的规模的增加而增加；
-  原问题能够被分解成更小的子问题；
- 子问题的结构和性质与原问题一样，并且相互独立，子问题之间不包含公共的子子问题；
- 原问题分解出的子问题的解可以合并为该问题的解。

#### 1.3 伪代码

```python
def divide_conquer(problem, paraml, param2,...): 
  	# 不断切分的终止条件
		if problem is None:
        print_result
				return
    # 准备数据
		data=prepare_data(problem)
		# 将大问题拆分为小问题
    subproblems=split_problem(problem, data)
    # 处理小问题，得到子结果 
    subresult1=self.divide_conquer(subproblems[0],p1,.....)	
    subresult2=self.divide_conquer(subproblems[1],p1,...) 	
    subresult3=self.divide_conquer(subproblems[2],p1,....)
    # 对子结果进行合并 得到最终结果
    result=process_result(subresult1, subresult2, subresult3,...)
```

#### 1.4 举例

分治算法应用在排序问题。

> 相关概念: 
>
> - 有序度:表示一组数据的有序程度
> - 逆序度:表示一组数据的无序程度 

一般**<u>通过计算有序对或者逆序对的个数</u>**，来表示数据的有序度或逆序度。
假设我们有 n 个数据，我们期望数据从小到大排列，那完全有序的数据的有序度就是`n*(n-1)/2`，逆序度等于 0;相反，倒序排列的数据的有序度就是 0，逆序度是`n*(n-1)/2`。

**Q:如何编程求出一组数据的有序对个数或者逆序对个数呢?** 

因为有序对个数和逆序对个数的求解方式是类似的，所以这里可以只思考逆序对(常接触的)个数的求解方法。 

**方法1**

- 拿数组里的每个数字跟它后面的数字比较，看有几个比它小的。

- 把比它小的数字个数记作 k ，通过这样的方式，把每个数字都考察一遍之后，然后对每个数字对应的 k 值求和

- 最后得到的总和就是逆序对个数。

  **<u>注意：这样操作的时间复杂度是O(n2)(需要两层循环过滤)。</u>**

那有没有更加高效的处理方法呢? 这里尝试套用分治的思想来求数组 A 的逆序对个数。

**方法2**

- 首先将数组分成前后两半 A1 和 A2，分别计算 A1 和 A2 的逆序对个数 K1 和 K2

- 然后再计算 A1 与 A2 之间的逆序对个数 K3。那数组 A 的逆序对个数就等于 **<u>K1+K2+K3</u>**

  <u>**注意：使用分治算法其中一个要求是，子问题合并的代价不能太大，否则就起不了降低时间复杂度的效果了**</u>

  

  如何快速计算出两个子问题 A1 与 A2 之间的逆序对个数呢?这里就要借助**<u>归并排序算法</u>**了。 

#### 1.5 补充：归并排序

归并排序中有一个非常关键的操作，就是将两个有序的小数组，合并成一个有序的数组。实际上，在这个合并的过程中，可以计算这两个小数组的逆序对个数了。每次合并操作，我们都计算逆序对个数，把这些计算出来的逆序对个数求和，就是这个数组的逆序对个数了。

归并算法的理解比较难，是一种区别于插入算法，选择算法和交换算法的一种独特算法，需要逐步理解。
**核心思想**：归并排序（MERGE-SORT）是利用归并的思想实现的排序方法，该算法采用经典的分治（divide-and-conquer）策略（分治法将问题分(divide)成一些小的问题然后递归求解，而治(conquer)的阶段则将分的阶段得到的各答案”修补”在一起，即分而治之)。


**（1）“分”的步骤：**

![这里写图片描述](https://img-blog.csdn.net/20180815112729271?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NxeDEzNzYzMDU1MjY0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**（2）“治”的步骤：**

![这里写图片描述](https://img-blog.csdn.net/20180815112753617?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NxeDEzNzYzMDU1MjY0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**（3）“治”的步骤详解：**

步骤一

![这里写图片描述](https://img-blog.csdn.net/20180815112907211?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NxeDEzNzYzMDU1MjY0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

步骤二

![这里写图片描述](https://img-blog.csdn.net/20180815112915428?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NxeDEzNzYzMDU1MjY0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

步骤三

![这里写图片描述](https://img-blog.csdn.net/20180815112921220?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NxeDEzNzYzMDU1MjY0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)





> 特点：
>
> （1）稳定性
> 　归并排序是一种稳定的排序。
> （2）存储结构要求
> 　可用顺序存储结构。也易于在链表上实现。
> （3）时间复杂度
> 　对长度为n的文件，需进行趟二路归并，每趟归并的时间为O(n)，故其时间复杂度无论是在最好情况下还是在最坏情况下均是O(nlgn)。
> （4）空间复杂度
> 　 需要一个辅助向量来暂存两有序子文件归并的结果，故其辅助空间复杂度为O(n)，显然它不是就地排序。

注意：若用单链表做存储结构，很容易给出就地的归并排序
归并排序是稳定排序，它也是一种十分高效的排序，能利用完全二叉树特性的排序一般性能都不会太差。java中Arrays.sort()采用了一种名为TimSort的排序算法，就是归并排序的优化版本。从上文的图中可看出，每次合并操作的平均时间复杂度为O(n)，而完全二叉树的深度为|log2n|。总的平均时间复杂度为O(nlogn)。而且，归并排序的最好，最坏，平均时间复杂度均为O(nlogn)。

**参考链接**：[归并排序](https://blog.csdn.net/cqx13763055264/article/details/81701419?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159783097119724835833868%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159783097119724835833868&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-81701419.first_rank_ecpm_v3_pc_rank_v4&utm_term=%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F&spm=1018.2118.3001.4187)

## 2、算法应用

#### 2.1 [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

> 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 `n/2` 的元素。
>
> 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
>
> 示例 1:
>
> 输入: [3,2,3]
> 输出: 3
> 示例 2:
>
> 输入: [2,2,1,1,1,2,2]
> 输出: 2



##### **官方思路：分治法**

1、**确定切分的终止条件**

直到所有的子问题都是长度为 1 的数组，停止切分。 

2、**准备数据**，**将大问题切分为小问题** 

递归地将原数组二分为左区间与右区间，直到最终的数组只剩下一个元素，将其返回

3、**处理子问题得到子结果**，**并合并**

> - 长度为 1 的子数组中唯一的数显然是众数，直接返回即可。 
> - 如果它们的众数相同，那么显然这一段区间的众数是它们相同的值。 
> - 如果他们的众数不同，比较两个众数在整个区间内出现的次数来决定该区间的众数

```python
class Solution(object):
    def majorityElement2(self, nums):
				"""
				:type nums: List[int] :rtype: int
				"""
				# 【不断切分的终止条件】 
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]
				# 【准备数据，并将大问题拆分为小问题】
        left = self.majorityElement(nums[:len(nums)//2]) 
        right = self.majorityElement(nums[len(nums)//2:]) 
        # 【处理子问题，得到子结果】
        # 【对子结果进行合并 得到最终结果】
				if left == right:
            return left
        if nums.count(left) > nums.count(right):
            return left
        else:
						return right
```

![image-20200819165055636](/Users/taolong/Library/Application Support/typora-user-images/image-20200819165055636.png)



##### **个人思路1：计数法**

- 对数组里的元素计数，按照出现数值进行倒序排列，然后获取第一个元素的key值即可

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        from collections import Counter
        return sorted(Counter(nums).items(),key=lambda x:x[1],reverse=True)[0][0]
```

![image-20200819162243653](/Users/taolong/Library/Application Support/typora-user-images/image-20200819162243653.png)



##### **个人思路2：分治法**

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        if not nums:
            return None
        if len(nums)==1:
            return nums[0]
        left=self.majorityElement(nums[:len(nums)//2])
        right=self.majorityElement(nums[len(nums)//2:])

        if nums.count(left)>=nums.count(right):
            return left
        else:
            return right
```

![image-20200819165329297](/Users/taolong/Library/Application Support/typora-user-images/image-20200819165329297.png)

#### 2.2 [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例:**

```python
输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**进阶:**

如果你已经实现复杂度为 O(*n*) 的解法，尝试使用更为精妙的分治法求解。

##### **官方思路**：

**1、确定切分的终止条件**
直到所有的子问题都是长度为 1 的数组，停止切分。 

**2、准备数据，将大问题切分为小问题** 

递归地将原数组二分为左区间与右区间，直到最终的数组只剩下一个元素，将其返回 

**3、处理子问题得到子结果，并合并**

> - 将数组切分为左右区间
>          对与左区间:从右到左计算左边的最大子序和
>          对与右区间:从左到右计算右边的最大子序和
> - 由于左右区间计算累加和的方向不一致，因此，左右区间直接合并相加之后就是整个区
>   间的和
> - 最终返回左区间的元素、右区间的元素、以及整个区间(相对子问题)和的最大值

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int] :rtype: int
        """
        # 【确定不断切分的终止条件】 
        n = len(nums)
        if n == 1:
            return nums[0]
          
        # 【准备数据，并将大问题拆分为小的问题】
        left = self.maxSubArray(nums[:len(nums)//2]) 
        right = self.maxSubArray(nums[len(nums)//2:])
        # 【处理小问题，得到子结果】
        # 从右到左计算左边的最大子序和
        max_l = nums[len(nums)//2 -1] # max_l为该数组的最右边的元素 
        tmp = 0 # tmp用来记录连续子数组的和
        for i in range( len(nums)//2-1 , -1 , -1 ):# 从右到左遍历数组的元素 
            tmp += nums[i]
            max_l = max(tmp ,max_l)
        # 从左到右计算右边的最大子序和
        max_r = nums[len(nums)//2]
        tmp = 0
        for i in range(len(nums)//2,len(nums)):
            tmp += nums[i]
            max_r = max(tmp,max_r)
        # 【对子结果进行合并 得到最终结果】
        # 返回三个中的最大值
        return max(left,right,max_l+ max_r)
```

![image-20200819214901052](/Users/taolong/Library/Application Support/typora-user-images/image-20200819214901052.png)



##### **个人思路**1：

（不成熟的初级想法，时间复杂度过高！）

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        sum1=[]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)+1):
                sum1.append(sum(nums[i:j]))
        return max(sum1)
```



##### **个人思路2：**

**<u>最大子序列的第一个元素不可能是负值！</u>**

因此若某一个元素x加上下一个元素y的和小于y，说明x小于0，那么最大子序列不可能从x开始。因此应选择下一个起始点进行讨论。

> 基本思路就是遍历一遍，用两个变量，一个记录最大的和，一个记录当前的和。
>
> 时间复杂度 O(n)，空间复杂度 O(l)

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int] :rtype: int
        """
        tmp=nums[0]
        sum1=tmp
        for i in range(1,len(nums)):
          #当当前序列加上此时的元素的值大于tmp的值，说明最大序列和可能出现在后续序列中，记录此时的最大值
            if nums[i]+tmp>nums[i]:
                sum1=max(sum1,tmp+nums[i])
                tmp=tmp+nums[i]
            else:
            #当tmp(当前和)小于下一个元素时，当前最长序列到此为止。以该元素为起点继续找最大子序列,
            # 并记录此时的最大值
                sum1=max(sum1,tmp,tmp+nums[i],nums[i]) #注意[-2,1]的情况
                tmp=nums[i]
        return sum1
```

![image-20200820014449869](/Users/taolong/Library/Application Support/typora-user-images/image-20200820014449869.png)



扩展：**动态规划试一试**？



#### 2.3 [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

难度中等474收藏分享切换为英文关注反馈

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数。

**示例 1:**

```python
输入: 2.00000, 10
输出: 1024.00000
```

**示例 2:**

```python
输入: 2.10000, 3
输出: 9.26100
```

**示例 3:**

```python
输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```

**说明:**

- -100.0 < *x* < 100.0
- *n* 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。



##### **官方思路1**：

**1、确定切分的终止条件**
对 n 不断除以2，并更新 n ，直到为0，终止切分 

**2、准备数据，将大问题切分为小问题**
对 n 不断除以2，更新 

**3、处理子问题得到子结果，并合并**
x 与自身相乘更新 x 

如果 n%2 ==1：将 p 乘以 x 之后赋值给 p (初始值为1)，返回 p 

**4、最终返回 p**

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float 
        :type n: int 
        :rtype: float 
        """
        # 处理n为负的情况 
        if n < 0 :
            x = 1/x 
            n = -n
        # 【确定不断切分的终止条件】 
        if n == 0 :
            return 1
        
        # 【准备数据，并将大问题拆分为小的问题】 
        if n%2 ==1:
            # 【处理小问题，得到子结果】
            p = x * self.myPow(x,n-1)# 【对子结果进行合并 得到最终结果】 
            return p
        return self.myPow(x*x,n/2)
```

![image-20200819222656910](/Users/taolong/Library/Application Support/typora-user-images/image-20200819222656910.png)

##### **官方思路2：**

![image-20200820001224411](/Users/taolong/Library/Application Support/typora-user-images/image-20200820001224411.png)

**复杂度分析**

> 时间复杂度：O(logn)，即为递归的层数。
>
> 空间复杂度：O(logn)，即为递归的层数。这是由于递归的函数调用会使用栈空间。
>

```python
class Solution(object):
    def myPow(self, x, n):
        def quick(m):
            if m==0:
                return 1
            y=quick(m//2)
            return y*y if m%2==0 else y*y*x
        return quick(n) if n>=0 else 1/quick(-n)
```

![image-20200820000424564](/Users/taolong/Library/Application Support/typora-user-images/image-20200820000424564.png)

##### **个人思路**：

在官方思路2的基础上，省略了定义函数的过程，直接在讨论特殊情况之后进行递归，然后根据奇偶性决定最终结果表达式。

```python
class Solution(object):
    def myPow(self, x, n):
        if n==0:
            return 1
        if n<0:
            x=1/x
            n=-n
        res=self.myPow(x,n//2)
        return res*res if n%2==0 else res*res*x
```

![image-20200820001100717](/Users/taolong/Library/Application Support/typora-user-images/image-20200820001100717.png)

