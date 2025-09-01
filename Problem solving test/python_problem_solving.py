# Python Problem Solving Examples
# A comprehensive collection of common problem-solving patterns

# =============================================================================
# 1. ARRAY/LIST PROBLEMS
# =============================================================================

def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def max_subarray_sum(arr):
    """Kadane's algorithm for maximum subarray sum"""
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

def rotate_array(nums, k):
    """Rotate array to the right by k steps"""
    n = len(nums)
    k = k % n
    nums[:] = nums[-k:] + nums[:-k]
    return nums

# =============================================================================
# 2. STRING PROBLEMS
# =============================================================================

def is_palindrome(s):
    """Check if string is palindrome (ignoring spaces/case)"""
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

def longest_common_prefix(strs):
    """Find longest common prefix among strings"""
    if not strs:
        return ""
    
    prefix = strs[0]
    for string in strs[1:]:
        while string[:len(prefix)] != prefix:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

def group_anagrams(strs):
    """Group strings that are anagrams"""
    from collections import defaultdict
    groups = defaultdict(list)
    
    for s in strs:
        # Sort characters as key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

# =============================================================================
# 3. MATHEMATICAL PROBLEMS
# =============================================================================

def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def fibonacci(n):
    """Generate nth Fibonacci number (optimized)"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def gcd(a, b):
    """Greatest Common Divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a

def power_mod(base, exp, mod):
    """Fast modular exponentiation"""
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    
    return result

# =============================================================================
# 4. SEARCHING AND SORTING
# =============================================================================

def binary_search(arr, target):
    """Binary search in sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def quick_sort(arr):
    """Quick sort implementation"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    """Merge sort implementation"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# =============================================================================
# 5. DYNAMIC PROGRAMMING
# =============================================================================

def coin_change(coins, amount):
    """Minimum coins needed to make amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_increasing_subsequence(nums):
    """Length of longest increasing subsequence"""
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def knapsack_01(weights, values, capacity):
    """0-1 Knapsack problem"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],
                    dp[i-1][w]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# =============================================================================
# 6. GRAPH PROBLEMS
# =============================================================================

def dfs_recursive(graph, node, visited=None):
    """Depth-First Search (recursive)"""
    if visited is None:
        visited = set()
    
    visited.add(node)
    print(node, end=' ')
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

def bfs(graph, start):
    """Breadth-First Search"""
    from collections import deque
    
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node, end=' ')
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def has_cycle(graph):
    """Detect cycle in directed graph using DFS"""
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {node: WHITE for node in graph}
    
    def dfs(node):
        if colors[node] == GRAY:
            return True  # Back edge found
        if colors[node] == BLACK:
            return False
        
        colors[node] = GRAY
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        colors[node] = BLACK
        return False
    
    for node in graph:
        if colors[node] == WHITE:
            if dfs(node):
                return True
    return False

# =============================================================================
# 7. TREE PROBLEMS
# =============================================================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """Inorder traversal (left, root, right)"""
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

def max_depth(root):
    """Maximum depth of binary tree"""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def is_valid_bst(root):
    """Check if binary tree is valid BST"""
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and 
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

# =============================================================================
# 8. BACKTRACKING PROBLEMS
# =============================================================================

def generate_parentheses(n):
    """Generate all valid parentheses combinations"""
    result = []
    
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

def solve_n_queens(n):
    """N-Queens problem solution"""
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonals
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def solve(board, row):
        if row == n:
            return [[''.join(row) for row in board]]
        
        solutions = []
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                solutions.extend(solve(board, row + 1))
                board[row][col] = '.'
        
        return solutions
    
    board = [['.' for _ in range(n)] for _ in range(n)]
    return solve(board, 0)

# =============================================================================
# 9. GREEDY ALGORITHMS
# =============================================================================

def activity_selection(start, finish):
    """Select maximum number of non-overlapping activities"""
    n = len(start)
    activities = list(zip(start, finish, range(n)))
    activities.sort(key=lambda x: x[1])  # Sort by finish time
    
    selected = [activities[0][2]]
    last_finish = activities[0][1]
    
    for i in range(1, n):
        if activities[i][0] >= last_finish:
            selected.append(activities[i][2])
            last_finish = activities[i][1]
    
    return selected

def huffman_encoding(text):
    """Huffman encoding for text compression"""
    from collections import Counter
    import heapq
    
    if not text:
        return {}, ""
    
    # Count frequencies
    freq = Counter(text)
    
    # Create heap
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract codes
    codes = {symbol: code for symbol, code in heap[0][1:]}
    encoded = ''.join(codes[char] for char in text)
    
    return codes, encoded

# =============================================================================
# 10. OPTIMIZATION PROBLEMS
# =============================================================================

def stock_max_profit(prices):
    """Maximum profit from buying and selling stock once"""
    if len(prices) < 2:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    
    return max_profit

def container_with_most_water(height):
    """Find container that holds most water"""
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        area = min(height[left], height[right]) * width
        max_area = max(max_area, area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# =============================================================================
# 11. DATA STRUCTURE IMPLEMENTATIONS
# =============================================================================

class Stack:
    """Stack implementation using list"""
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        return len(self.items) == 0

class Queue:
    """Queue implementation using collections.deque"""
    def __init__(self):
        from collections import deque
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")
    
    def is_empty(self):
        return len(self.items) == 0

class LRUCache:
    """LRU Cache implementation"""
    def __init__(self, capacity):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        self.cache[key] = value

# =============================================================================
# 12. PATTERN MATCHING AND REGEX
# =============================================================================

def pattern_matching_kmp(text, pattern):
    """KMP algorithm for pattern matching"""
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    if not pattern:
        return []
    
    lps = compute_lps(pattern)
    matches = []
    i = j = 0
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

def wildcard_matching(s, p):
    """Wildcard pattern matching with * and ?"""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    dp[0][0] = True
    
    # Handle patterns like a* or *a*
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]

# =============================================================================
# 13. COMBINATORICS AND PERMUTATIONS
# =============================================================================

def permutations(nums):
    """Generate all permutations of array"""
    if len(nums) <= 1:
        return [nums]
    
    result = []
    for i in range(len(nums)):
        rest = nums[:i] + nums[i+1:]
        for p in permutations(rest):
            result.append([nums[i]] + p)
    
    return result

def combinations(n, k):
    """Generate all combinations of k elements from 1 to n"""
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result

def subsets(nums):
    """Generate all possible subsets"""
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# =============================================================================
# 14. NUMBER THEORY
# =============================================================================

def sieve_of_eratosthenes(n):
    """Find all prime numbers up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]

def factorize(n):
    """Prime factorization of number"""
    factors = []
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factors.append(n)
    
    return factors

# =============================================================================
# 15. EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Test examples
    print("=== ARRAY PROBLEMS ===")
    print("Two Sum:", two_sum([2, 7, 11, 15], 9))
    print("Max Subarray:", max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    
    print("\n=== STRING PROBLEMS ===")
    print("Is Palindrome:", is_palindrome("A man a plan a canal Panama"))
    print("Anagrams:", group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    
    print("\n=== MATH PROBLEMS ===")
    print("Is Prime 17:", is_prime(17))
    print("Fibonacci 10:", fibonacci(10))
    print("GCD(48, 18):", gcd(48, 18))
    
    print("\n=== SEARCHING ===")
    arr = [1, 3, 5, 7, 9, 11]
    print("Binary Search for 7:", binary_search(arr, 7))
    
    print("\n=== DYNAMIC PROGRAMMING ===")
    print("Coin Change:", coin_change([1, 3, 4], 6))
    print("LIS:", longest_increasing_subsequence([10, 22, 9, 33, 21, 50, 41, 60]))
    
    print("\n=== COMBINATORICS ===")
    print("Combinations C(4,2):", combinations(4, 2))
    print("Subsets of [1,2]:", subsets([1, 2]))
    
    print("\n=== NUMBER THEORY ===")
    print("Primes up to 20:", sieve_of_eratosthenes(20))
    print("Factorize 60:", factorize(60))