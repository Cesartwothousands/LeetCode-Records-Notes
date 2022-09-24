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



## (21) Merge Two Sorted Lists

### Content

Two sorted lists *list1* and *list2*

The list should be made by splicing together the nodes of the first two lists.



### Try1

Use Stack

Try list first, use trans def

```python
def LtoL(linkedNode):
l=[]
while linkedNode:
l.append(linkedNode.val)  
linkedNode= linkedNode.next
return l                   

def list2link(List):
if List == []:
return None

head = ListNode(List[0])
p = head
for i in range(1, len(List)):
p.next = ListNode(List[i])
p = p.next
return head
```

while(len(l1) >0 and len(l2) >0):

​	if l1[i] <= l2[j]:

​		res.append()

​		pop(0)

​	else

​		......

if len(l1)==0:

​	res.append(l2)



return res

### Solutions

Check if any of the lists is empty.

Use LinkedList

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        l1 = list1
        l2 = list2
        
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        # Choose head which is smaller of the two lists
        if l1.val < l2.val:
            temp = head = ListNode(l1.val)
            l1 = l1.next
        else:
            temp = head = ListNode(l2.val)
            l2 = l2.next
        # Loop until any of the list becomes null
        while l1 is not     None and l2 is not None:
            if l1.val < l2.val:
                temp.next = ListNode(l1.val)
                l1 = l1.next
            else:
                temp.next = ListNode(l2.val)
                l2 = l2.next
            temp = temp.next
        # Add all the nodes in l1, if remaining
        while l1 is not None:
            temp.next = ListNode(l1.val)
            l1 = l1.next
            temp = temp.next
        # Add all the nodes in l2, if remaining
        while l2 is not None:
            temp.next = ListNode(l2.val)
            l2 = l2.next
            temp = temp.next
        return head
```

Time: For travesing and assigning value both lists, $O(m+n)$

Space: $O(1)$

### Review

```
class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
```

## (121) Best Time to Buy and Sell Stock

### Content

**Example 1:**

```python
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
```

If you can't have a profit, return 0

### Try1

Brute: (n-1 + 1) n / 2            $O(n^2)$

for i in range(len())

​	temp = p[0]

​	p.pop(0)	

​	res = max(p) - temp

``` python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        temp = 0
        res = 0
        Res = 0
        p = []
        p = p + prices
        
        for i in range(len(prices)-1) :
            temp = prices[i]
            p.pop(0)
            if temp <= max(p):
                res = max(p) - temp
            if res > Res:
                Res = res
        
        return Res
```

Time Limit Exceeded

### Try2

int 3 variables, actually 2

num: real time num

Min: lowest in traverse, it will only decrase



res = num - Min

if res > Res, Res = res

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0
        Res = 0
        Min = prices[0]
        
        for i in range(len(prices)):
            
            if i > 0:
                if prices[i] < prices[i-1] and prices[i] < Min:
                    Min = prices[i]
                
            res = prices[i] - Min  
            
            Res =max(Res,res)
            
        return Res
```

 ### Solutions

### Review

```
if res > Res:
                Res = res
#### Can be optimized as:
                
Res =max(Res,res)
```

### 





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

## (985) Sum of Even Numbers After Queries

### Content

```
Input: nums = [1,2,3,4], queries = [[1,0],[-3,1],[-4,0],[2,3]]
Output: [8,6,2,4]
Explanation: At the beginning, the array is [1,2,3,4].
After adding 1 to nums[0], the array is [2,2,3,4], and the sum of even values is 2 + 2 + 4 = 8.
After adding -3 to nums[1], the array is [2,-1,3,4], and the sum of even values is 2 + 4 = 6.
After adding -4 to nums[0], the array is [-2,-1,3,4], and the sum of even values is -2 + 4 = 2.
After adding 2 to nums[3], the array is [-2,-1,3,6], and the sum of even values is -2 + 6 = 4.
```

### Try1

def A:

if integer is even, append(1), [[nums[i], bool], XX, XX, XX]

def B:

if  ( $A[i][2]$ ):

sum



for i in range(len())

​	if odd and A[] [] == 0 

​			change A[] []

​	elif even

......

Change bool first

do the sum later



```python
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        res = []
        Nums = []
        
        for i in range(len(nums)):
            temp = []
            temp.append(nums[i])
            temp.append(nums[i] % 2)
            # even for bool 0
            Nums.append(temp)
  
        for i in range(len(nums)):
            sum = 0

            Nums[queries[i][1]][0] = Nums[queries[i][1]][0] + queries[i][0]
         
            if queries[i][0] % 2 :
                Nums[queries[i][1]][1] = int(not Nums[queries[i][1]][1])
               
            for i in range(len(nums)):
                if not Nums[i][1] :
                    sum += Nums[i][0]
            res.append(sum)   
        
        return res
```

**Time Limit Exceeded**

### Solutions

#### Approach 1: Maintain Array Sum

```python
class Solution(object):
    def sumEvenAfterQueries(self, A, queries):
        S = sum(x for x in A if x % 2 == 0)
        ans = []

        for x, k in queries:
            if A[k] % 2 == 0: S -= A[k]
            A[k] += x
            if A[k] % 2 == 0: S += A[k]
            ans.append(S)

        return ans
```

Brute Force

faster than 8.64% of Python3 online submissions

#### Better speed

```python
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        lst=[]
        su=0
        for k in nums:
            if k%2==0:
                su+=k
        for x,y in queries:
            if nums[y]%2==0:
                su-=nums[y]
            nums[y]+=x
            if nums[y]%2==0:
                su+=nums[y]
            lst.append(su)
        return lst
```

faster than 96.44%

### Review



## (557) Reverse Words in a String III

### Content

**Example 1:**

```
Input: s = "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
```

**Example 2:**

```
Input: s = "God Ding"
Output: "doG gniD"
```

### Try1

res = []

stack = []

s.append(' ')



for len()

​	stack.appned

​	if s[i] == [' ']

​		for len(stack)

​			res.append(stack[-1])

​			stack.pop()



res.pop(1)

return res

> Runtime: 306 ms, faster than 5.04% of Python3 online submissions for Reverse Words in a String III.

### Try2

``` python
class Solution:
    def reverseWords(self, s: str) -> str:
        res = []
        List = []
        s = list(s)
        s.append(' ')

        stack = []
        for i in range(len(s)):
            stack.append(s[i])

            if s[i] == ' ':
                List.append(stack)
                stack = []
                
        for l in List:
            for j in range(len(l)):
                res.append(l[-1])
                l.pop()
                    
        res.pop(0)
        res = ''.join(res)
        return res
```

> Runtime: 116 ms, faster than 24.29% of Python3 online submissions for Reverse Words in a String III.

### Solutions

#### Using Two Pointers

However, there is another optimal approach to reverse the string in \mathcal{O}(N/2)O(*N*/2) time in place using two pointer approach.

### Review



## (1680) Concatenation of Consecutive Binary Numbers

### Content

```
Input: n = 12
Output: 505379714
Explanation: The concatenation results in "1101110010111011110001001101010111100".
The decimal value of that is 118505380540.
After modulo 109 + 7, the result is 505379714.
```

### Try1

```python
class Solution:
    def concatenatedBinary(self, n: int) -> int:
        Len = []
        Num = []
        LL = 0
        res = 0
    
        for i in range(n):
            num = i + 1
            Num.append(num)
            b = ''
            s = 1
            while(num):
                s = num // 2
                y = num % 2
                b = b + str(y)
                num = s
            
            Len.append(len(b))

        L = sum(Len)
        
        for i in range(len(Num)):
            LL += Len[i]
            res += Num[i] * pow(2, L-LL)
        
        res = res % 1000000007
        return res
```

> Time Limit Exceeded

### Solutions

```python
class Solution:
    def concatenatedBinary(self, n: int) -> int:
        bits, res, MOD = 1, 0, 10**9 + 7
        for x in range(1, n + 1):
            res = ((res << bits) + x) % MOD
            if x == (1 << bits) - 1:
                bits += 1    
        return res
```

The bitwise left shift operator (`<<`) 

> ((res << bits) + x)
>
> x for value


```python
class Solution:
    def concatenatedBinary(self, n: int) -> int:
        s = 0
        for i in range(1, n+1):
            s = (s << i.bit_length() | i) % 1000000007    
        return s
```

### Review

n & (n-1) == 0 to indentify is it a power of 2?

100000000000

011111111111

## (409) Longest Palindrome

### Content

```python
Input: s = "abccccdd"
Output: 7
Explanation: One longest palindrome that can be built is "dccaccd", whose length is 7.
```

### Try1

Build a Hash Table

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        Hashmap = {}
        l = []
        res = 0
        d = 0
        
        for i in range(len(s)):
            if s[i] not in Hashmap:
                Hashmap[s[i]] = 0
            Hashmap[s[i]] += 1
            
        for i in Hashmap:
            res += 2*(Hashmap[i]//2)
            if Hashmap[i] % 2 :
                d = 1
        
        if d:
            return res + 1
        return res
```

### Solutions

## (150) Evaluate Reverse Polish Notation

### Content

### Try1

Stack

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        voc = []
        temp = 0
        res = 0
        
        for i in range(len(tokens)):  
            if tokens[i] == "+":
                temp = voc[-1] + voc[-2]
                voc.pop()
                voc.pop()
                voc.append(temp)
                
            elif tokens[i] == '-':
                temp = voc[-2] - voc[-1]
                voc.pop()
                voc.pop()
                voc.append(temp)
                
            elif tokens[i] == '*':
                temp = voc[-1] * voc[-2]
                voc.pop()
                voc.pop()
                voc.append(temp)
                
            elif tokens[i] == '/':
                temp = int(voc[-2] / voc[-1])
                voc.pop()
                voc.pop()
                voc.append(temp)
            
            else:
                tokens[i] = int(tokens[i])
                voc.append(tokens[i])
                
        res = int(voc[0])
        return res
```

## (226) Invert Binary Tree

### Content

Given the root of a binary tree, and return its root.

```python
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
```

### Solutions

```python
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        if not root or (not root.left and not root.right):
            return root
##########################################################调换左右节点
        tmp = root.left
        root.left = root.right
        root.right = tmp
#########################################################
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

