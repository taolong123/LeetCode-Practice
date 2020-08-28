# Task04：查找-对撞指针

**算法应用**

**LeetCode 1 Two Sum**

**题目描述**

> 给出一个整型数组nums，返回这个数组中两个数字的索引值i和j，使得nums[i] + nums[j]等于一个 给定的target值，两个索引不能相等。
>
> 如:nums= [2,7,11,15],target=9 返回[0,1]

**解题思路**

需要考虑:

> 1. 开始数组是否有序;
> 2. 索引从0开始计算还是1开始计算? 
> 3. 没有解该怎么办?
> 4. 有多个解怎么办?保证有唯一解。

**分析实现**
 暴力法 O（n2）
时间复杂度为  O（n2）,第一遍遍历数组，第二遍遍历当前遍历值之后的元素，其和等于target则return

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        len_nums = len(nums)
        for i in range(len_nums):
            for j in range(i+1,len_nums):
                if nums[i] + nums[j] == target:
return [i,j]
```



