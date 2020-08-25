# Task03：查找--查找表

### 考虑的基本数据结构

#### 第一类: 查找有无--set 

元素'a'是否存在，通常用set:集合 set只存储键，而不需要对应其相应的值。 set中的键不允许重复

#### 第二类: 查找对应关系(键值对应)--dict 

元素'a'出现了几次:dict-->字典 dict中的键不允许重复

#### 第三类: 改变映射关系--map 

通过将原有序列的关系映射统一表示为其他

### 算法应用

LeetCode 349 Intersection Of Two Arrays 1
**题目描述**
给定两个数组nums,求两个数组的公共元素。

```python
如nums1 = [1,2,2,1],nums2 = [2,2]
结果为[2] 
结果中每个元素只能出现一次 
出现的顺序可以是任意的
```

**分析实现**
由于每个元素只出现一次，因此不需要关注每个元素出现的次数，用set的数据结构就可以了。记录元 素的有和无。
把nums1记录为set，判断nums2的元素是否在set中，是的话，就放在一个公共的set中，最后公共的 set就是我们要的结果。
代码如下:

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = set(nums1)
        return set([i for i in nums2 if i in nums1])
```



也可以通过set的内置方法来实现，直接求set的交集:

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1 = set(nums1)
        set2 = set(nums2)
        return set2 & set1
```

