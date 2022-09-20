[TOC]

## Word Lists

enumerate 枚举

duplicate 重复

recursion 

recursively 递归

Iterate 迭代

increase

increment 增加

brackets 括号

traverse 遍历











------

# Grind 75

## (1) Two Sum

### Content

**Example 1:**

```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
```

You may assume that each input would have ***exactly\* one solution**, and you may not use the *same* element twice.

**Constraints:**

- **Only one valid answer exists.**

### Try1

Boudaries:

lo < hi

lo < target // 2

hi > target // 2



Sort the List first

​		for i in range(0,len)

​				lo = i

​				While lo < hi						

​				if  num[lo] + num[hi] == target

​						res = 						

​						return res

​				else

​						hi--	

### Try2

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        lo = 0
        while(1):
            time = 1
            hi = len(nums) - time
            while (lo < hi):
                if nums[lo] + nums[hi] == target:
                    res = []
                    res.append(lo)
                    res.append(hi)
                    return res
                else:
                    hi = hi - 1
            lo = lo + 1
```

### Solutions

#### Brute Force

Two loop,  $O(n^2)$

```
for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[j] == target - nums[i]:
                    return [i, j]
```

#### Two-pass Hash Table

Have a map

check $target - nums[i]$

must not be nums[i] itself!

> A simple implementation uses two iterations. In the first iteration, we add each element's value as a key and its index as a value to the hash table. Then, in the second iteration, we check if each element's complement (target - nums[i]*t**a**r**g**e**t*−*n**u**m**s*[*i*]) exists in the hash table. If it does exist, we return current element's index and its complement's index. Beware that the complement must not be nums[i]*n**u**m**s*[*i*] itself!

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            hashmap[nums[i]] = i
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap and hashmap[complement] != i:
                return [i, hashmap[complement]] 
```

#### One-pass Hash Table

Once you iterat and insert elements into the hash table

<u>Look back to check!</u>

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i
```

### Review

One pass Hash Table, once you insert once you check

## (20) Valid Parentheses

### Content

string s containing just ( ) { } [ ]

input is valid if:

- Same type
- Correct order
- Pair

```
Input: s = "()[]{}"
Output: true
```

```
Input: s = "(]"
Output: false
```

- `s` consists of parentheses only `'()[]{}'`.

### Try1

str to list

for

compare every 2 digit

3 if

```python
class Solution:
    def isValid(self, s: str) -> bool:
        S = list(s)
        l = len(S)
        if not (l % 2):
            return False
        
        for i in range(l-2):
            if not (i%2):
                if (S[i] == '('):
                    if not (S[i+1] == ')'):
                        return False
                elif (S[i] == '['):
                    if not (S[i+1] == ']'):
                        return False
                elif (S[i] == '{'):
                    if not (S[i+1] == '}'):
                        return False
                else:
                    return False
        
        return True
```

### Try2

build a stack

use if X in []

if correspodding, pop()

if len(stack) == 0, True

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        i = 0
        
        if (len(s)%2):
            return False
        
        for c in s:
            i += 1
            if (i%2) :
                if c in ['(','[','{']:
                    stack.append(c)
                else:
                    return False
            else:
                if c in [')',']','}']:
                    if c == ')':
                        if not (stack[-1] == '('):
                            return False
                        else:
                            stack.pop()
                    elif c == ']':
                        if not (stack[-1] == '['):
                            return False
                        else:
                            stack.pop()
                    elif c == '}':
                        if not (stack[-1] == '{'):
                            return False
                        else:
                            stack.pop()
                else:
                    return False 
                
        if not (len(stack)):        
            return True
```

Wrong understanding: "{[]}" is True too

### Solutions

Traverse first

Stack: its `Last In First Out (LIFO)` property

Put left in first, search for right to compare

At last, it will be a empty Stack if it's right

Time: Traverse, $O(n^2)$ 

Space: Stack, $O(n^2)$ 

```python
def isValid(s: str) -> bool:
    # Stack for left symbols
    leftSymbols = []
    # Loop for each character of the string
    for c in s:
        # If left symbol is encountered
        if c in ['(', '{', '[']:
            leftSymbols.append(c)
        # If right symbol is encountered
        elif c == ')' and len(leftSymbols) != 0 and leftSymbols[-1] == '(':
            leftSymbols.pop()
        elif c == '}' and len(leftSymbols) != 0 and leftSymbols[-1] == '{':
            leftSymbols.pop()
        elif c == ']' and len(leftSymbols) != 0 and leftSymbols[-1] == '[':
            leftSymbols.pop()
        # If none of the valid symbols is encountered
        else:
            return False
    return leftSymbols == []
```

### Review



# LeetCode Everyday

## **(18)** 4Sum

### Content

- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, and `d` are **distinct**.
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

You may return the answer in **any order**.

 

**Example 1:**

```
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```

**Constraints:**

- `1 <= nums.length <= 200`
- `-109 <= nums[i] <= 109`
- `-109 <= target <= 109`

### Try1

NaN

### Solutions

#### Two pointers

Sort array first

Call kSum with start = 0, k = 4 and target, return at last

​		At the beginning of the kSum function, we will check three conditions:

​				How we run out of numbers

​				Lowest number of numbers: <= target / k

​				Largest number of numbers: >= target / k

​		

​		If k == 2, call twoSum

​		Iterate i through array

​				If value == the one before, skip it

​				Recursively call kSum with start = i + 1, k = k - 1, and target - num[i]

​				For each return a subset

​						Add this subset to Res

TwoSum function:

​		Set the low pointer form lo to start, and high pointer to the last index

​		While lo < hi:

​				if the sum of num[lo] and num[hi] < than target, lo++

​				if num[lo] == num[lo - 1], lo++

​				if the sum > target, hi--

​				if num[hi] == num[hi + 1], hi--

​				found a pair: Add it to the Res, hi-- and lo++

​		Return Res

### Review

- Boundaries 
  - Number of num
  - Most special: target // 2 + target // 2 = target