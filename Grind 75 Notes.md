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

ascend 向上

prefix 字首

suffix 字尾

consecutive 连续

concatenate 使连接，把...连成一串

intuitive 直观的、直觉的

------

# Grind 75: Array

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

### Review: Array

```
if res > Res:
                Res = res
#### Can be optimized as:
                
Res =max(Res,res)
```

-----

## (238) Product of Array Except Self

### Content

Given an integer array `nums`, return *an array* `answer` *such that* `answer[i]` *is equal to the product of all the elements of* `nums` *except* `nums[i]`.

The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in `O(n)` time and without using the division operation.

### Try1(Failed)

### Solutions

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = []
        
        acc = 1
        for n in nums:
            res.append(acc)
            acc *= n

        acc = 1
        for i in reversed(range(len(nums))):
            res[i] *= acc
            acc *= nums[i]
            
        return res
```

### Review: Prefix Sum

When we meet Prefix Sum, we often use Inclusion–exclusion principle (容斥原理)

-----

## (169) Majority Element

### Content

### Try 1 & 2

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        for i in set(nums):
            if nums.count(i) > len(nums)//2:
                return i
            
class Solution:
    def majorityElement(self, nums) -> int:
        H_2 = {}

        for mm in nums:
            H_2[mm] = H_2.get(mm, 0) + 1

        for key in H_2:
            if H_2[key] > len(nums)//2:
```

### Solutions

LeetCode give 7 solutions:

>https://leetcode.com/problems/majority-element/solutions/127412/majority-element/

### Review: Array & Hash Table & Sort & Divide and Conquer

-----

## (217) Contains Duplicate

### Content

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: true
```

**Example 2:**

```
Input: nums = [1,2,3,4]
Output: false
```

### Try1(Success)

#### Sort

```python
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        nums.sort()
        for i in range(len(nums)):
            if i>0:
                if nums[i] == nums[i-1]:
                    return True

        return False 
```

### Solutions

#### Set

```python
class Solution(object):
    def containsDuplicate(self, nums):
        seen = set()
        for n in nums:
            if n in seen:
                return True
            seen.add(n)
        return False
```

#### Hash table

### Review: Array & Sort & Hash Table

set.add(n)
numsSet =  set(nums)

-----

## (219) Contains Duplicate II

### Content

Given an integer array `nums` and an integer `k`, return `true` if there are two **distinct indices** `i` and `j` in the array such that `nums[i] == nums[j]` and `abs(i - j) <= k`.

### Try1(Success)

Use a Hash table to store the location of first number

every time meet another same number, compare the abs(i - j)

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        H = {}
        for i in range(len(nums)):
            if not nums[i] in H:
                H[nums[i]] = i 
            else:
                if abs(i - H[nums[i]]) <= k:
                    return True
                H[nums[i]] = i

        return False
```

### Solutions

### Review: Hash Table

-----

## (220) Contains Duplicate III

### Content

You are given an integer array `nums` and two integers `indexDiff` and `valueDiff`.

Find a pair of indices `(i, j)` such that:

- `i != j`,
- `abs(i - j) <= indexDiff`.
- `abs(nums[i] - nums[j]) <= valueDiff`, and

Return `true` *if such pair exists or* `false` *otherwise*.

### Try1(Failed)

### Solutions


### Review: 

-----

## (252) Meeting Room

### Content

Given an array of meeting time intervals consisting of start and end times`[[s1,e1],[s2,e2],...]`(si< ei), determine if a person could attend all meetings.

### Try1()

use si to sort them first

use a theory to decide whether is right

if si > ei-1 :

elif ei < si-1 :

else: 

return false

### Solutions

**Time complexity** : O(nlogn). The time complexity is dominated by sorting. Once the array has been sorted, only O(n) time is taken to go through the array and determine if there is any overlap.

**Space complexity** : O(1). Since no additional space is allocated.

```python
def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        new_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(1,len(new_intervals)):
            if new_intervals[i-1][1] > new_intervals[i][0]:return False
        return True
```

```c++
class Solution {
    public boolean canAttendMeetings(Interval[] intervals) {
        Arrays.sort(intervals, new Comparator<Interval>() {
           public int compare(Interval i1, Interval i2) {
               return i1.start - i2.start;
           } 
        });
        Interval last = null;
        for (Interval i: intervals) {
            if (last != null && i.start < last.end) {
                return false;
            }
            last = i;
        }
        return true;
    }
}
```

### Review: Array & Sort 

-----

## (283) Move Zeroes

### Content

Given an integer array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Note** that you must do this in-place without making a copy of the array.

### Try1(Success)

while l

​	append in the end

​	pop()

​	i--

i++

l--

```Python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        i = 0
        count = 0
        while(l):
            if nums[i] == 0:
                nums.append(0)
                nums.pop(i)
                i -= 1
            i += 1
            l -= 1
```

### Solutions

#### Two Pointers

```Python
def moveZeroes(self, nums):
    zero = 0  # records the position of "0"
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[i], nums[zero] = nums[zero], nums[i]
            zero += 1
```

### Review: Array & Two Pointers

-----

## (977) Square of a Sorted Array

### Content

Given an integer array `nums` sorted in **non-decreasing** order, return *an array of **the squares of each number** sorted in non-decreasing order*.



> **Follow up:** Squaring each element and sorting the new array is very trivial, could you find an `O(n)` solution using a different approach?

### Try1(Success)

Two pointers

from end and start

campare each other

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        l = len(nums)
        end = l - 1
        start = 0
        res = []

        while start <= end:
            if nums[end] * nums[end] >= nums[start] * nums[start]:
                res.append(nums[end] * nums[end])
                end -= 1    
            else:
                res.append(nums[start] * nums[start])
                start += 1

        res.reverse()

        return res
    
        ##or return res[::-1]
```

### Solutions

### Review: Two Pointers & Array & Sort

-----

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

```
Boudaries:

lo < hi

lo < target // 2

hi > target // 2



Sort the List first

		for i in range(0,len)
				lo = i
				While lo < hi						
				if  num[lo] + num[hi] == target
						res = 						
						return res
				else
						hi--	
```



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

-----

# Grind 75: String

## (125) Valid Palindrome

### Content

Given a string `s`, return `true` *if it is a **palindrome**, or* `false` *otherwise*.

**Example 1:**

```python
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
```



### Try1

Use List to store string, use a pop(0),pop(-1)stack to make decesion

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        S = []

        for ss in s:
            if ss.isalnum():
                S.append(ss) 

        for i in range(len(S)//2):
            if S[0] == S[-1]:
                S.pop(0)
                S.pop(-1)
            else:
                return False

        if S != []:
            S.pop()

        if S == []:
            return True
```

### Solutions

#### Shortest and funny:

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
            newS= [i.lower() for i in s if i.isalnum()]
            return newS == newS[::-1]
```

#### Two Pointer!

```python
class Solution:
    def isPalindrome(self, s):
        l, r = 0, len(s)-1
        # First and Last
        while l < r:
        # Guarantee the order
            while l < r and not s[l].isalnum():
                l += 1
            while l <r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l +=1; r -= 1
        return True
```

### Review: String & Two Pointers

.lower() function

```
s.isalnum所有字符都是数字或者字母，为真返回Ture，否则返回False。（重点，这是字母数字一起判断的！！）

s.isalpha所有字符都是字母，为真返回Ture，否则返回False。（只判断字母）

s.isdigit所有字符都是数字，为真返回Ture，否则返回False。（只判断数字）

s.islower所有字符都是小写，为真返回Ture，否则返回False。

s.isupper所有字符都是大写，为真返回Ture，否则返回False。

s.istitle所有单词都是首字母大写，为真返回Ture，否则返回False。

s.isspace所有字符都是空白字符，为真返回Ture，否则返回False。
```

while l < r and not s[l].isalnum():

-----

## (242) Valid Anagram

### Content

Given two strings `s` and `t`, return `true` *if* `t` *is an anagram of* `s`*, and* `false` *otherwise*.

**Example 1:**

```
Input: s = "anagram", t = "nagaram"
Output: true
```

### Try1

First For: Hashmap 

Use Hashmap to count

Second For: if == : 

Third For review

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        Hashmap = {}
        if len(s) != len(t):
            return False
        
        for i in range(len(s)):
            if s[i] not in Hashmap:
                Hashmap[s[i]] = 0
            Hashmap[s[i]] += 1

        for tt in t:
            if tt in Hashmap:
                Hashmap[tt] -= 1

        for key in Hashmap:
            if Hashmap[key] != 0:
                return False

        return True
```

### Solutions

#### Traditional Dictionary

```python
def isAnagram1(self, s, t):
    dic1, dic2 = {}, {}
    for item in s:
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2
```

#### Pre set 0 Dictionary

```python
def isAnagram2(self, s, t):
    dic1, dic2 = [0]*26, [0]*26
    for item in s:
        dic1[ord(item)-ord('a')] += 1
    for item in t:
        dic2[ord(item)-ord('a')] += 1
    return dic1 == dic2
```

#### Pre set 0 Dictionary

Use Sorted()
```python
def isAnagram3(self, s, t):
    return sorted(s) == sorted(t)
```

### Reciew: String & Hash Table & Sort

-----

## (3) Longest Substring Without Repeating Characters

### Content

Given a string `s`, find the length of the **longest substring** without repeating characters.

> substring:**A substring is a contiguous non-empty sequence of characters within a string.**

### Try1(Success)

Use a Hashtable to see if any words to repeat

Use a varaiable to store the begin of our substring, we call it mark

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l = 0
        mark = 0
        H = {}
        
        for i in range(len(s)):
            if not s[i] in H:
                H[s[i]] = i
            else:
                if mark < H[s[i]] + 1:
                    mark = H[s[i]] + 1
                H[s[i]] = i

            if i - mark + 1 > l:
                l = i - mark + 1
        
        return l
```

### Solutions

#### Brute

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        def check(start, end):
            chars = set()
            for i in range(start, end + 1):
                c = s[i]
                if c in chars:
                    return False
                chars.add(c)
            return True

        n = len(s)

        res = 0
        for i in range(n):
            for j in range(i, n):
                if check(i, j):
                    res = max(res, j - i + 1)
        return res
```

### Review: Hashtable & Sliding Window

Tips

All previous implementations have no assumption on the charset of the string `s`.

If we know that the charset is rather small, we can mimic what a HashSet/HashMap does with a boolean/integer array as direct access table. Though the time complexity of query or insertion is still O(1)O(1)*O*(1), the constant factor is smaller in an array than in a HashMap/HashSet. Thus, we can achieve a shorter runtime by the replacement here.

Commonly used tables are:

- `int[26]` for Letters 'a' - 'z' or 'A' - 'Z'
- `int[128]` for ASCII
- `int[256]` for Extended ASCII

-----

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

### Review: Hash Table & String

-----

## (14) Longest Common Prefix

### Content

```
Input: strs = ["flower","flow","flight"]
Output: "fl"
```

### Try1(Failed)

Use a list to collect all letters(in special orders) in first word

Use a count to count how long the common prefix is

```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        count = len(strs[0])
        Map = []
        res = ""

        for i in range(len(strs[0])):
            Map.append(strs[0][i])
        
        for i in range(1,len(strs)):
            c = 0
            for ii in range(len(strs[i])):
                if strs[i][ii] == Map[ii]:
                    c += 1
                else:
                    if c < count:
                        count = c
                    break

        if count == 0 or strs == [""]:
            return ""
        else:    
            if len(strs) < 2:
                return strs[0]
            else:
                return strs[0][0:count]
```

### Try2(Success)

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        l = 200
        exm = ""
        string = ""

        for strss in strs:
            if len(strss) < l:
                l = len(strss)
                exm = strss
                if l == 0:
                    return ""

        for i in range(len(strs)):
            j = 0
            while j < len(exm):
                if strs[i][j] != exm[j]:
                    exm = exm[0:j]
                    break
                j += 1
        
        for e in exm:
            string += str(e)
        return string
```

### Solutions

https://leetcode.com/problems/longest-common-prefix/solutions/127449/longest-common-prefix/


### Review: String

Swap in python:

nums[i], nums[zero] = nums[zero], nums[i]

-----

# Grind 75: Matrix

# Grind 75: Binary Search

# Grind 75: Graph

# Grind 75: Binary Search Tree

## (704) Binary Search

### Content

Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

**You must write an algorithm with `O(log n)` runtime complexity.**

### Try1

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if nums[len(nums)//2-1] >= target:
            nums = nums[:len(nums)//2-1]
        else:
            nums = nums[len(nums)//2-1:]

        if len(nums) == 1:
            if nums[0] == target:
                return nums[0]
            else: 
                return -1

        self.search(nums, target)    
```

### Try2

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        while(len(nums) > 1):
            if nums[len(nums)//2-1] >= target:
                nums = nums[:len(nums)//2-1]
            else:
                nums = nums[len(nums)//2-1:]

        if len(nums) == 1:
            if nums[0] == target:
                return nums[0]
            else: 
                return -1
```

### Solutions

#### Two Pointers Loop

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while(left<= right):
            mid=(left+right)//2
            if nums[mid]==target:
                return mid
            elif nums[mid] < target:
                left=mid+1
            else:
                right=mid-1
        return -1
```

#### Recursive Solution

```python
class Solution(object):
    def search(self, nums, target,count=0):
        length=len(nums)
        half_len=length//2
        test=nums[half_len]
        
        if test == target:
            return half_len+count
        
        if length == 1:
            return -1
        
        if test > target:
            return self.search(nums[0:half_len],target,count+0)
        
        else:
            return self.search(nums[half_len:],target,count+half_len)
```

### Review: Binary Search

-----


## (278) First Bad Version

### Content

**Example 1:**

```
Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.
```

### Try1(Failed)

Use a half to store n/2

Use half to decide whether bigger or less

### Solutions

#### Binary Search

Binary search ==> initialize **low, high and mid** for binary search

while low pointer crosses high run the loop

calculate for the mid

if isBadVersion true then set high = mid;
else set low = mid+1;

Finally return low;

```c++
class Solution {
public:
    int firstBadVersion(int n) {

        int low = 1;
        int high = n;
        int mid;
        
//////////////////////////////////////////////////////////////////////////////learn
        while(low < high){

            mid = low + (high-low) / 2;

            if(isBadVersion(mid)){
                high = mid;
            }
            else{
                low = mid + 1;
            }
        }
///////////////////////////////////////////////////////////////////////////////////
        
        return low;
    }
};
```

### Review: Binary Search

mid = low + (high-low) / 2;

-----

# Grind 75: Binary Tree

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

-----

# Grind 75: Hash Table

## (383) Ransom Note

### Content

Given two strings `ransomNote` and `magazine`, return `true` *if* `ransomNote` *can be constructed by using the letters from* `magazine` *and* `false` *otherwise*.

**Example 3:**

```
Input: ransomNote = "aa", magazine = "aab"
Output: true
```

### Try1

Two Hashmap

if ==, Hashmap[key] -= 1

```Python
class Solution:
    def canConstruct(self, r: str, m: str) -> bool:
        H_2 = {}

        for mm in m:
            H_2[mm] = H_2.get(mm, 0) + 1

        for rr in r:            
            if rr in H_2 and H_2[rr] > 0:
                H_1[rr] -= 1
                H_2[rr] -= 1
            else :
                return False
        
        return True
```

### Solutions

#### Hashmap

Try1

#### Set

Use set and .count function 

```python
class Solution:
    def canConstruct(self, ransomNote, magazine):
        for i in set(ransomNote):
            if magazine.count(i) < ransomNote.count(i):
                return False
        return True
```

### Review: String & Hash Table

-----

# Grind 75: Recursion

# Grind 75: Linked List

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

-----

# Grind 75: Stack

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

### Review: Stack

-----

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

### Review: Stack

-----

## (844) Backspace String Compare

### Content

Given two strings `s` and `t`, return `true` *if they are equal when both are typed into empty text editors*. `'#'` means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

### Try1(Success)

easy stack problem

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        ss = []
        tt = []

        for sss in s:
            if sss != "#":
                ss.append(sss)
            elif ss:
                ss.pop()
        for ttt in t:
            if ttt != "#":
                tt.append(ttt)
            elif tt:
                tt.pop()
        S = "".join(ss)
        T = "".join(tt)

        return S==T
```

Use elif ss: to prevent the [] situation

### Solutions

Another one: Two pointers

Iterate through the string in reverse. If we see a backspace character, the next non-backspace character is skipped. If a character isn't skipped, it is part of the final answer.

See the comments in the code for more details.

### Review: Stack & String & all()

#### all() function

The **Python all() function** returns true if all the elements of a given iterable (List, Dictionary, Tuple, set, etc.) are True otherwise it returns False. It also returns True if the iterable object is empty. Sometimes while working on some code if we want to ensure that user has not entered a False value then we use the all() function.

```
**Syntax:** all( iterable )
- **Iterable:** It is an iterable object such as a dictionary,tuple,list,set,etc.
**Returns:** boolean
```

-----

# Grind 75: Queue

# Grind 75: Heap

# Grind 75: Trie

# Grind 75: Dynamic Programming

## (70) Climbing Stairs

### Content

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

```python
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```

### Try1(Success)

Two steps are more important than one step

Just count two steps

if i in range(n//2):

​	i is the number of 2 steps

​	how to calculate the numbers of each possible situation

​	$C^i_(n-i)$

​	to escape the n^2 section, calculate $C^i_(n-i)$ first, calculate once a time

write two list to use their value of factorial 

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        p = n // 2 
        Sum, ssm = 0, 0
        
        up,ilist = [1],[1]
        for i in range(1, n+1):
            up.append(i*up[i-1])   
            ilist.append(i*up[i-1]) 
        up.reverse()

        for i in range(p+1):
            ssm = up[i] / ilist[i] / up[2 * i]
            print(up, ilist,up[i] ,ilist[i] ,up[2 * i])
            Sum += ssm
        
        return int(Sum)
```

### Solutions

#### See it as a Fibonacci

```
# Top down - TLE
def climbStairs1(self, n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    return self.climbStairs(n-1)+self.climbStairs(n-2)
```

```python
class Solution:

# Bottom up, O(n) space
    def climbStairs(self, n):
        if n == 1:
            return 1
        res = [0 for i in range(n)]
        res[0], res[1] = 1, 2
        for i in range(2, n):
            res[i] = res[i-1] + res[i-2]
        return res[-1]
```

### Review: Math & Fibonacci & Recursive/Iteration

-----

# Grind 75: Binary

# Grind 75: Math

## (13) Roman to Integer

### Content

Trans Roman to Integer

```
Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

### Try1(Failed)

Use a L1 = [] to sotre the Roman appears

for i in range:

​	if i in L1:

​		if i -1 == i:

​			use count to do action next loop

​		else:

​			L1.pop()

​	else:

​		L1.append()	 

Failed

### Try2(Success)

Bruce

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        L = []
        S = 0
        count = 0

        def cal(self, n):
            if n == 'I':
                return 1
            elif n == 'V':
                return 5
            elif n == 'X':
                return 10
            elif n == 'L':
                return 50
            elif n == 'C':            
                return 100
            elif n == 'D':            
                return 500
            elif n == 'M':            
                return 1000

        for i in range(1,len(s)):
            if cal(self, n = s[i-1])>= cal(self, n = s[i]):
                S+= cal(self, n = s[i-1] )
            else:
                S-= cal(self, n = s[i-1]) 
        S += cal(self, n = s[-1])

        return S
```

better than 5.1%

### Solutions

#### Dictionary to replace

```java
class Solution:
    def romanToInt(self, s: str) -> int:
        translations = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }
        number = 0
        s = s.replace("IV", "IIII").replace("IX", "VIIII")
        s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
        s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
        for char in s:
            number += translations[char]
        return number
```

### Review: String & Hash Table 

-----

## (9) Palindrome Number

### Content

**Example 1:**

```
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.
```

**Example 2:**

```
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
```

### Try1(Success)

<0 false

<10 true

Two pointers,i and -i

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        elif x<10:
            return True
        else:
            X = []
            while x :
                X.append( x % 10)
                x = x//10
                
            for i in range(len(X)//2):
                if X[i] != X[-i-1]:
                    return False

            return True
```

### Solutions

> Second idea would be reverting the number itself, and then compare the number with original number, if they are the same, then the number is a palindrome. However, if the reversed number is larger than int.MAX\text{int.MAX}int.MAX, we will hit integer overflow problem.
>
> Following the thoughts based on the second idea, to avoid the overflow issue of the reverted number, what if we only revert half of the int\text{int}int number? After all, the reverse of the last half of the palindrome should be the same as the first half of the number, if the number is a palindrome.



```
if(x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
```

```C++
int revertedNumber = 0;
        while(x > revertedNumber) {
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }

        return x == revertedNumber || x == revertedNumber/10;
```

Just revert half of x

### Review: Math & Array & String

-----
