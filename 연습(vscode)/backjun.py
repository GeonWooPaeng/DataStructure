# 1.
# case_test = int(input())
# for _ in range(case_test):
#     L = list(input())
#     stack_l = []
#     stack_r = []

#     for i in range(len(L)):
#         stack_l.append(L[i])
#         if stack_l[-1] == '<':
#             stack_l.pop()
#             if stack_l:
#                 stack_r.append(stack_l.pop())

#         elif stack_l[-1] == '>':
#             stack_l.pop()
#             if stack_r:
#                 stack_l.append(stack_r.pop())
        
#         elif stack_l[-1] == '-':
#             stack_l.pop()
#             if stack_l:
#                 stack_l.pop()

#     stack_l.extend(reversed(stack_r))
#     print(''.join(stack_l))


# 2.
# test_case = int(input())

# for _ in range(test_case):
#     stack_l = []
#     stack_r = []
#     data = input() 
#     for i in data:
#         if i == '-':
#             if stack_l:
#                 stack_l.pop()
        
#         elif i == '<':
#             if stack_l:
#                 stack_r.append(stack_l.pop())
                
#         elif i == '>':
#             if stack_r:
#                 stack_l.append(stack_r.pop())
        
#         else:
#             stack_l.append(i)

#     stack_l.extend(reversed(stack_r))
#     print(''.join(stack_l))


# 1.
# import hashlib

# data = str(input())
# encoded_data = data.encode()
# result = hashlib.sha256(encoded_data).hexdigest()
# print(result)


# 1.
# N = int(input())
# num1 = set(map(int,input().split()))
# N2 = int(input())
# num2 = list(map(int,input().split()))

# for index2 in num2:
#     if index2 not in num1:
#         print(0)    
#     else:
#         print(1)


# 4195.

# def find(x):
#     if x == parent[x]:
#         return x
    
#     else:
#         p = find(parent[x])
#         parent[x] = p 
#         return parent[x]

# def union(x,y):
#     x = find(x)
#     y = find(y)

#     if x != y:
#         parent[y] = x
#         number[x] += number[y]

# case_task = int(input())

# for _ in range(case_task):
#     parent = dict() 
#     number = dict()

#     relation = int(input())

#     for _ in range(relation):
#         x,y = input().split(' ')
        
#         if x not in parent:
#             parent[x] = x
#             number[x] = 1
#         if y not in parent:
#             parent[y] = y
#             number[y] = 1 
        
#         union(x,y)
#         print(number[find(x)])
    
# 2750.
# n = int(input())
# data = list() 

# for _ in range(n):
#     data.append(int(input()))

# for stand in range(len(data)):
#     lowest = stand 
#     for index in range(stand+1,len(data)):
#         if data[lowest] > data[index]:
#             lowest = index
#         data[stand], data[lowest] = data[lowest], data[stand]

# for i in data:
#     print(i)

      
# 1427.
# 1.
# data = list()
# num = input()
# for i in range(len(num)):
#     data.append(int(num[i]))

# for stand in range(len(data)):
#     highest = stand
#     for index in range(stand, len(data)):
#         if data[index] > data[highest]:
#             highest = index 
#     data[stand], data[highest] = data[highest],data[stand]

# data = list(map(str,data))
# print(('').join(data))

# 2.

# array = input() 

# for i in range(9,-1,-1):
#     for j in array:
#         if int(j) == i:
#             print(i, end='')


# 10814.
# 1.
# case_task = int(input())
# mem_l = list() 

# for _ in range(case_task):
#     mem = input().split()
#     mem_l.append((int(mem[0]),mem[1]))

# for stand in range(len(mem_l)):
#     lowest = stand 
#     for i in range(stand,len(mem_l)):
#         if mem_l[lowest][0] > mem_l[i][0]:
#             lowest = i
#     mem_l[stand] , mem_l[lowest] = mem_l[lowest],mem_l[stand]

# for mem in mem_l:
#     print(mem[0],mem[1])

# 2.
# n = int(input())
# array = []

# for _ in range(n):
#     input_data = input().split(' ')
#     array.append((int(input_data[0]),input_data[1]))

# array = sorted(array, key = lambda x:x[0])

# for i in array:
#     print(i[0],i[1])


# 11650.
# n = int(input())
# location = []

# for _ in range(n):
#     x,y = map(int,input().split(' '))
#     location.append((x,y))

# location = sorted(location)

# for i in location:
#     print(i[0],i[1])


# 10989.
# import sys

# n = int(sys.stdin.readline())
# array = [0] * 10001 

# for _ in range(n):
#     data = int(sys.stdin.readline())
#     array[data] += 1

# for i in range(10001):
#     if array[i] != 0:
#         for j in range(array[i]):
#             print(i)


# 2747.
# 1.
# def pibo(n):
#     if n == 0:
#         return 0
#     elif n == 1:
#         return 1

#     return pibo(n-1) + pibo(n-2)

# n = int(input())
# print(pibo(n))

# 2.
# n = int(input())

# a,b = 0,1
# while n > 0:
#     a,b = b, a+b 
#     n -= 1

# print(a)

# 1074.
# def Z(n,x,y):
#     global result 
#     if n == 2:
#         if x == X and y == Y:
#             print(result)
#             return 
#         result += 1

#         if x == X and y+1 == Y:
#             print(result)
#             return 
#         result += 1

#         if x+1 == X and y == Y:
#             print(result)
#             return 
#         result += 1

#         if x+1 == X and y+1 == Y:
#             print(result)``
#             return 
#         result += 1
#         return

#     Z(n/2,x,y)
#     Z(n/2,x,y+n/2)
#     Z(n/2,x+n/2,y)
#     Z(n/2,x+n/2,y+n/2)

# result = 0
# N,X,Y = map(int,input().split(' '))
# Z(2**N,0,0)

# 7490.
# import copy

# def recursive(array,n):
#     if len(array) == n:
#         operator_list.append(copy.deepcopy(array))
#         return 

#     array.append(' ')
#     recursive(array,n)
#     array.pop()

#     array.append('+')
#     recursive(array,n)
#     array.pop()

#     array.append('-')
#     recursive(array,n)
#     array.pop()

# case_test = int(input())

# for _ in range(case_test):
#     operator_list = []
#     n = int(input())
#     recursive([],n-1)

#     integer = [i for i in range(1,n+1)]

#     for operator in operator_list:
#         string = " "
#         for i in range(n-1):
#             string += str(integer[i]) + operator[i]
#         string += str(integer[-1])

#         if eval(string.replace(" ","")) == 0:
#             print(string)
    
# 2751.

# def qsort(data):
#     if len(data) <= 1:
#         return data
    
#     pivot = data[0]
#     left,right = list(),list()

#     for index in range(1,len(data)):
#         if data[index] < pivot:
#             left.append(data[index])
        
#         else:
#             right.append(data[index])

#     return qsort(left) + [pivot] + qsort(right)

# def split(data):
#     if len(data) <= 1:
#         return data

#     med = len(data) // 2

#     left = split(data[:med])
#     right = split(data[med:])

#     return merge(left,right)

# def merge(left,right):
#     merged = list()

#     lp,rp = 0,0
#     while len(left) > lp and len(right) > rp:
#         if left[lp] > right[rp]:
#             merged.append(right[rp])
#             rp += 1
#         else:
#             merged.append(left[lp])
#             lp += 1 

#     while len(left) > lp:
#         merged.append(left[lp])
#         lp += 1

#     while len(right) > rp:
#         merged.append(right[rp])
#         rp += 1 

#     return merged

# test_case = int(input())
# num = list()

# for _ in range(test_case):
#     n = int(input())
#     num.append(n)

# sorted_num = split(num)
# # sorted_num = qsort(num)

# for i in sorted_num:
#     print(i)


# 1300.

# N = int(input())
# k = int(input())

# B = set() 
# for i in range(N+1):
#     for j in range(N+1):
#         B.add(i*j)

# list(B).sort() 
# print(list(B)[k-2])

# 2110.
# # 사이 간격(gap)을 중점으로 생각 해야 한다.
# n, c = list(map(int,input().split(' '))) # n(집수), c(공유기수)
# array = []
# for _ in range(n):
#     array.append(int(input()))
# array = sorted(array)

# start = array[1] - array[0] #start(min => 최소 gap)
# end = array[-1] - array[0] #end(max => 최대 gap)
# result = 0 #최적의 gap (start랑 end가 같을 때)

# while (start <= end): # 최대 최소가 바뀔때 
#     mid = (start+end)//2 #이분법 위한 평균 gap 
#     value = array[0] #초기 공유기가 설치 되있는 집
#     count = 1 #공유기 count (array[0]에 설치 되어 있으므로 1)

#     for i in range(1,len(array)):
#         if array[i] >= value + mid: #공유기가 있는 곳에서 + gap 한 값에 공유기를 놓을 수 있다.
#             value = array[i] # 다음 공유기 놓을 수 있는 곳
#             count += 1 #공유기수 += 1
    
#     if count >= c: #c개 이상의 공유기수 설치 
#         start = mid + 1 #gap 간격 넓혀보기
#         result = mid #모든 조건 다 됬으므로 저장
#     else: #c개 미만의 공유기수 설치
#         end = mid - 1 # gap 간격 줄이기

# print(result) 

# 1939.
# from collections import deque

# n,m = map(int,input().split())
# adj = [[] for _ in range(n+1)] #특정 섬에 (연결되어 있는 섬, weight) 저장

# def bfs(c): # 경로를 이동하였는지 확인하기 위한 함수
#     queue = deque([start_node]) 
#     visited = [False] * (n + 1) # 방문하는 곳 처음에 False로 해놈
#     visited[start_node] = True  # 처음 시작을 넣는다
#     while queue:
#         x = queue.popleft() 
#         for y, weight in adj[x]: #y - x랑 이어진 섬, weight - 중량
#             if not visited[y] and weight >= c: # visited[y]가 있고 weight(for에서 돌아가는 x랑 y 섬의 다리 중량)가 현재 중량보다 큰 경우
#                 visited[y] = True 
#                 queue.append(y)
#     return visited[end_node]

# start = 1000000000
# end = 1

# for _ in range(m):
#     x, y, weight = map(int,input().split())
#     adj[x].append((y,weight)) #특정 섬에 (연결되어 있는 섬, weight) 추가
#     adj[y].append((x,weight)) #특정 섬에 (연결되어 있는 섬, weight) 추가
#     start = min(start, weight) #weight이 가장 작은 것
#     end = max(end, weight)     #weight이 가장 큰 것

# start_node, end_node = map(int, input().split()) #공장이 있는 섬들

# result = start 
# while(start <= end):
#     mid = (start + end)//2 #현재의 중량값
#     if bfs(mid): # 이동이 가능하므로, 중량을 증가시킨다.
#         result = mid 
#         start = mid + 1
#     else: # 이동이 불가능하므로, 중량을 감소 시킨다.
#         end = mid -1 
# print(result)

# 1543.

# doc = input()
# word = input()

# index = 0 
# count = 0

# while len(doc) - index >= len(word):
#     if doc[index:index+len(word)] == word:
#         count += 1
#         index += len(word)
#     else:
#         index += 1
# print(count)

# 1568.
# n = int(input())

# flied_b = 1
# sec = 0

# while n != 0:
#     if flied_b > n:
#         flied_b = 1
    
#     n -= flied_b 
#     flied_b += 1
#     sec += 1

# print(sec)

# 1302.

# n = int(input())
# chart = dict()

# for _ in range(n):
#     book = input()
#     if book not in chart :
#         chart[book] = 1
#     else:
#         chart[book] += 1 
        
# target = max(chart.values())
# array = []

# for book, number in chart.items():
#     if target == number:
#         array.append(book)

# print(sorted(array)[0])

# 1668.

# 1.
# n = int(input())

# lengths = []
# leng_l = 0
# leng_r = 0
# count_l = 0
# count_r = 0

# for _ in range(n):
#     length = int(input())
#     lengths.append(length)

# for i in range(n):
#     if leng_l < lengths[i]:
#         leng_l = lengths[i]
#         count_l += 1
# print(count_l)

# for j in range(n-1,-1,-1):
#     if leng_r < lengths[j]:
#         leng_r = lengths[j]
#         count_r += 1
# print(count_r)

# 2.
# def ascending(array):
#     now = array[0]
#     result = 1
#     for i in range(1,len(array)):
#         if now < array[i]:
#             result += 1
#             now = array[i]
#     return result 

# n = int(input())
# array = []

# for _ in range(n):
#     array.append(int(input()))

# print(ascending(array))
# array.reverse()
# print(ascending(array))

# 1236.

# n,m = map(int,input().split())
# conditions = []

# for _ in range(n):
#     condition = input()
#     conditions.append(condition)

# row = [0] * n
# column = [0] * m

# for i in range(n):
#     for j in range(m):
#         if conditions[i][j] == 'X':
#             row[i] = 1
#             column[j] = 1

# count_row = 0
# for i in range(n):
#     if row[i] == 0:
#         count_row += 1

# count_column = 0 
# for j in range(m):
#     if column[j] == 0:
#         count_column += 1

# print(max(count_row,count_column))

# 10870.

# def pibo(n):
#     if n == 0:
#         return 0 
#     if n == 1:
#         return 1
#     return pibo(n-1) + pibo(n-2)
# n = int(input())
# print(pibo(n))

# 1991.
# class Node:
#     def __init__(self,data,left_node,right_node):
#         self.data = data 
#         self.left_node = left_node 
#         self.right_node = right_node 

# def pre_order(node):
#     print(node.data, end='')
#     if node.left_node != '.':
#         pre_order(tree[node.left_node])
#     if node.right_node != '.':
#         pre_order(tree[node.right_node])

# def in_order(node):
#     if node.left_node != '.':
#         in_order(tree[node.left_node])
#     print(node.data,end='')
#     if node.right_node != '.':
#         in_order(tree[node.right_node])

# def post_order(node):
#     if node.left_node != '.':
#         post_order(tree[node.left_node])
#     if node.right_node != '.':
#         post_order(tree[node.right_node])
#     print(node.data,end='')

# n = int(input())
# tree = {}

# for i in range(n):
#     data, left_node, right_node = input().split() 
#     tree[data] = Node(data,left_node,right_node)

# pre_order(tree['A'])
# print()
# in_order(tree['A'])
# print()
# post_order(tree['A'])

# 2250.
# class Node:
#     def __init__(self,number,left_node,right_node):
#         self.parent = -1 
#         self.number = number 
#         self.left_node = left_node 
#         self.right_node = right_node 

# def in_order(node,level):  #중위 순회로 좌측부터 검사 / level은 행 
#     global level_depth,x # level_depth는 행 개수
#     level_depth = max(level_depth, level)
#     if node.left_node != -1:
#         in_order(tree[node.left_node], level + 1) # 자식 node인 경우에는 level이 다르기 때문에 level + 1을 해준다
    
#     level_min[level] = min(level_min[level], x) # 중위 순회로 해서 x+=1 해서 저장하는 이유는 비어있는 열이 없기 때문입니다.
#     level_max[level] = max(level_max[level], x)
#     x += 1
    
#     if node.right_node != -1:
#         in_order(tree[node.right_node], level + 1) # 자식 node인 경우에는 level이 다르기 때문에 level + 1을 해준다

# n = int(input()) 
# tree = {}
# level_min = [n] #같은 level에서 최소의 열 저장
# level_max = [0] #같은 level에서 최대의 열 저장
# root = -1 
# x = 1
# level_depth = 1 

# for i in range(1, n+1):
#     tree[i] = Node(i,-1,-1)
#     level_min.append(n)
#     level_max.append(0)

# for _ in range(n): #tree에 node 저장 하기
#     number,left_node,right_node = map(int,input().split())
#     tree[number].left_node = left_node 
#     tree[number].right_node = right_node 
#     if left_node != -1:
#         tree[left_node].parent = number 
#     if right_node != -1:
#         tree[right_node].parent = number 

# for i in range(1, n+1): 
#     if tree[i].parent == -1: #부모 node가 없으면 root이다.
#         root = i

# in_order(tree[root], 1)

# result_level = 1
# result_width = level_max[1] - level_min[1] + 1  #결과 넓이
# for i in range(2, level_depth + 1): # 1은 이미 계산 했음
#     width = level_max[i] - level_min[i] + 1
#     if result_width < width: # 결과 넓이보다 넓을 경우
#         result_level = i
#         result_width = width 

# print(result_level, result_width)

# 1927.
# 1.
# array = []
# n = int(input())
# for _ in range(n):
#     num = int(input())

#     if num == 0:
#         if len(array) > 0:
#             array.sort()
#             print(array.pop(0))
#         else:
#             print(0)
#     else:
#         array.append(num)

# 2.
# import heapq 

# n = int(input())
# heap = []
# result = []

# for _ in range(n):
#     data = int(input())
#     if data == 0:
#         if heap:
#             result.append(heapq.heappop(heap))
#         else:
#             result.append(0)
#     else:
#         heapq.heappush(heap,data)

# for data in result:
#     print(data)

# 1715.
# 1.
# import heapq 

# result = []
# heap = []

# n = int(input())
# for _ in range(n):
#     data = int(input())
#     heapq.heappush(heap,data)

# while len(heap) != 1:
#     data2 = heapq.heappop(heap) + heapq.heappop(heap)
#     heapq.heappush(heap,data2)
#     result.append(data2)

# result_n = 0
# for i in result:
#     result_n += i 
# print(result_n)

# 2.
# import heapq 

# n = int(input())
# heap = []

# for i in range(n):
#     data = int(input())
#     heapq.heappush(heap,data)

# result = 0 

# while len(heap) != 1:
#     one = heapq.heappop(heap)
#     two = heapq.heappop(heap)
#     sum_value = one + two
#     result += sum_value 
#     heapq.heappush(heap, sum_value)
# print(result)

# 1766.

# import heapq 
# n,m = map(int,input().split())
# array = [[] for i in range(n+1)]
# indegree = [0] * (n+1)

# heap = []
# result = []

# for _ in range(m):
#     x,y = map(int,input().split())
#     array[x].append(y)
#     indegree[y] += 1

# for i in range(1, n+1):
#     if indegree[i] == 0:
#         heapq.heappush(heap,i)

# while heap:
#     data = heapq.heappop(heap)
#     result.append(data)
#     for y in array[data]:
#         indegree[y] -= 1
#         if indegree[y] == 0:
#             heapq.heappush(heap,y)

# for i in result:
#     print(i, end=' ')

# 1904.
# n = int(input())

# array = [0] * 10001
# array[1] = 1
# array[2] = 2

# for i in range(3,n+1):
#     array[i] = (array[i-1]+array[i-2]) % 15746

# print(array[n])

# 12865.
# n, k = map(int,input().split())
# d_p = [[0] * (k+1) for _ in range(n+1)]

# print(d_p)
# for i in range(1, n+1):
#     weight, value = map(int,input().split())
#     for j in range(1,k+1):
#         if j < weight:
#             d_p[i][j] = d_p[i-1][j]
#         else:
#             d_p[i][j] = max(d_p[i-1][j], d_p[i-1][j-weight] + value)

# print(d_p[n][k]) 

# 11053.
# n = int(input())
# num = list(map(int,input().split()))

# dp = [1] * n 

# for i in range(1,n):
#     for j in range(i):
#         if num[i] > num[j]:
#             dp[i] = max(dp[i],dp[j]+1)

# print(max(dp))

# 9251.
# A = input() 
# B = input() 

# dp = [[0]*(len(B)+1) for _ in range(len(A)+1)]

# for y in range(1,len(A)+1):
#     for x in range(1,len(B)+1):
#         if A[y-1] == B[x-1]:
#             dp[y][x] = dp[y-1][x-1] + 1

#         else:
#             dp[y][x] = max(dp[y-1][x],dp[y][x-1])

# print(dp[len(A)][len(B)])

# 1495.
# n,s,m = map(int,input().split())

# volume = list(map(int,input().split()))

# data = [[0]*(m+1) for _ in range(n+1)]
# data[0][s] = 1

# for y in range(1,n+1):
#     for x in range(m+1):
#         if data[y-1][x] == 0:
#             continue # 0일 시 밑의 코드 건너 뛴다.
#         if x - volume[y-1] >= 0:
#             data[y][x-volume[y-1]] = 1
#         if x + volume[y-1] <= m:
#             data[y][x+volume[y-1]] = 1 

# result = -1

# for x in range(m,-1,-1):
#     if data[n][x] == 1:
#         result = x 
#         break

# print(result)

# 2655.

# n = int(input())

# value = []
# dp = [0] * (n+1)

# value.append((0,0,0,0)) #(번호,밑면,높이,무게)

# for i in range(1,n+1):
#     width,height,weight = map(int,input().split())
#     value.append((i,width,height,weight))

# value.sort(key = lambda x : x[3])

# for i in range(1, n+1):
#     for j in range(0,i):
#         if value[i][1] > value[j][1]:
#             dp[i] = max(dp[i],dp[j]+value[i][2])

# max_value = max(dp)

# result = [] 

# for i in range(n,-1,-1):
#     if max_value == dp[i]:
#         result.append(value[i][0])
#         max_value -= value[i][2]
#         if max_value == 0: 
#             break 

# print(len(result))
# result.reverse() 
# for i in result:
#     print(i)

# 1260.
# 1.
# def bfs(graph,start_node):
#     visited = []
#     need_visit = []
#     need_visit.append(start_node)

#     while need_visit:
#         node = need_visit.pop(0)
#         if node not in visited:
#             visited.append(node)
#             need_visit.extend(graph[node])


#     return visited

# def dfs(graph, start_node):
#     visited = []
#     need_visit = [] 
#     need_visit.append(start_node)

#     while need_visit:
#         node = need_visit.pop() 
#         if node not in visited: 
#             visited.append(node)
#             need_visit.extend(graph[node])

#     return visited

# n,m,start_node = map(int,input().split())

# graph = [[] for _ in range(n+1)]

# for _ in range(m):
#     x,y = map(int,input().split())
#     graph[x].append(y)
#     graph[y].append(x)

# for i in graph:
#     i.sort() 
# bfs_result = bfs(graph,start_node)

# for i in graph:
#     i.reverse() 
# dfs_result = dfs(graph,start_node)

# print(' '.join(map(str,dfs_result)))
# print(' '.join(map(str,bfs_result)))

# 2.
# from collections import deque 

# def dfs(v):
#     print(v,end=' ')
#     visited[v] = True 
#     for e in adj[v]:
#         if not (visited[e]):
#             dfs(e)

# def bfs(v):
#     q = deque([v])
#     while q:
#         v = q.popleft()
#         if not(visited[v]):
#             visited[v] = True 
#             print(v,end=' ')
#             for e in adj[v]:
#                 if not visited[e]:
#                     q.append(e)

# n,m,v = map(int,input().split())
# adj = [[] for _ in range(n+1)]

# for _ in range(m):
#     x, y = map(int,input().split())
#     adj[x].append(y)
#     adj[y].append(x)

# for e in adj:
#     e.sort() 

# visited = [False] * (n+1)
# dfs(v)
# visited = [False] * (n+1)
# bfs(v)

# 1697.
# 1.
# from collections import deque 

# MAX = 100001 
# n, k = map(int,input().split())
# array = [0] * MAX 

# def bfs():
#     q = deque([n])
#     while q:
#         now_pos = q.popleft()
#         if now_pos == k:
#             return array[now_pos]
#         for next_pos in (now_pos -1, now_pos +1, now_pos *2):
#             if 0 <= next_pos < MAX and not array[next_pos]:
#                 array[next_pos] = array[now_pos] + 1
#                 q.append(next_pos)
# print(bfs())

# 2.
# n, k = map(int,input().split())
# array = [0] * 100001 

# def bfs():

#     need_visit = []
#     need_visit.append(n)
#     while need_visit:
#         node = need_visit.pop(0)
#         if node == k:
#             return array[node]

#         for next_num in (node-1, node+1, node*2):
#             if array[next_num] == 0:
#                 array[next_num] = array[node] + 1
#                 need_visit.append(next_num)

# print(bfs())

# 2606.
# 1.
# def bfs(graph,start_node):
#     visited = []
#     need_visit=[]
#     need_visit.append(start_node)
    
#     while need_visit:
#         node = need_visit.pop(0)
#         if node not in visited:
#             visited.append(node)
#             need_visit.extend(graph[node])

#     return (len(visited)-1) 

# def dfs(graph,start_node):
#     visited = []
#     need_visit = []
#     need_visit.append(start_node)

#     while need_visit:
#         node = need_visit.pop(0)
#         if node not in visited:
#             visited.append(node)
#             need_visit.extend(graph[node])
    
#     return (len(visited)-1)

# com_num = int(input())

# graph = [[] for _ in range(com_num+1)]

# link_num = int(input())

# for i in range(link_num):
#     x,y = map(int,input().split())
#     graph[x].append(y)
#     graph[y].append(x)

# print(bfs(graph,1))
## print(dfs(graph,1))

# 2.
# n = int(input())
# m = int(input())
# adj = [[] for _ in range(n+1)]
# visited = [False] * (n+1)
# count = 0

# for _ in range(m):
#     x,y = map(int,input().split())
#     adj[x].append(y)
#     adj[y].append(x)

# def dfs(now_pos):
#     global count 
#     count += 1
#     visited[now_pos] = True 
#     for next_pos in adj[now_pos]:
#         if not visited[next_pos]:
#             dfs(next_pos)

# dfs(1)
# print(count - 1)

# 10872.

# n = int(input())
# def fectorial(n): 
#     if n <= 1:
#         return 1 

#     return n*fectorial(n-1) 

# print(fectorial(n))

# 1012.

# import sys
# sys.setrecursionlimit(100000)

# def dfs(x,y):
#     visited[x][y] = True 
#     directions = [(-1,0),(1,0),(0,-1),(0,1)]
#     for dx,dy in directions:
#         nx,ny = x+dx, y+dy
#         if nx < 0 or nx >= n or ny < 0 or ny >= m:
#             continue
#         if array[nx][ny] and not visited[nx][ny]:
#             dfs(nx,ny)


# for _ in range(int(input())):
#     m,n,k = map(int,input().split()) 
#     array = [[0] * m for _ in range(n)]
#     visited = [[False] * m for _ in range(n)]
#     for _ in range(k):
#         y,x = map(int,input().split())
#         array[x][y] = 1
#     result = 0
#     for i in range(n):
#         for j in range(m):
#             if array[i][j] and not visited[i][j]:
#                 dfs(i,j)
#                 result += 1
#     print(result)

# 2.
# import sys
# sys.setrecursionlimit(100000)

# def dfs(x,y):
#     field[x][y] = 0
#     directions=[(-1,0),(1,0),(0,-1),(0,1)]
#     for dx,dy in directions:
#         nx,ny = x+dx, y+dy
#         if nx < 0 or nx >= n or ny < 0 or ny >= m:
#             continue
#         if field[nx][ny]:
#             dfs(nx,ny)

# for _ in range(int(input())):
#     m,n,k = map(int,input().split())
#     field = [[0]*m for _ in range(n)]

#     for _ in range(k):
#         y,x = map(int,input().split())
#         field[x][y] = 1
#     result = 0
#     for i in range(n):
#         for j in range(m):
#             if field[i][j]:
#                 dfs(i,j)
#                 result += 1
#     print(result)

# 10951.
# import sys

# while True:
#     try:
#         x,y = map(int,sys.stdin.readline().split()) 
#     except:    
#         break
#     print(x+y)

# 15969.
# N = int(input())
# score = list(map(int,input().split()))

# print(max(score)- min(score))

# 1325.
# 1.
# import sys 

# def bfs(graph,start_node):
#     visited = []
#     need_visit = []
#     need_visit.append(start_node)

#     while need_visit:
#         node = need_visit.pop(0)
#         if node not in visited:
#             visited.append(node)
#             need_visit.extend(graph[node])

#     return len(visited)

# n,m = map(int,sys.stdin.readline().split())

# adj_link = [[] for _ in range(n+1)]  
# num = [[] for _ in range(n+1)]

# for _ in range(m):
#     x,y = map(int,sys.stdin.readline().split())
#     adj_link[y].append(x)

# for i in range(1,m+1):
#     num[i].append(bfs(adj_link,i))

# result = []
# for i in range(1,m+1):
#     if num[i] == max(num):
#         result.append(i)

# result.sort()
# print(' '.join(map(str,result)))

# 2.
# from collections import deque 

# n,m = map(int,input().split())
# adj = [[] for _ in range(n+1)]

# for _ in range(m):
#     x,y = map(int,input().split())
#     adj[y].append(x)

# def bfs(v):
#     q = deque([v])
#     visited = [False] * (n+1)
#     visited[v] = True 
#     count = 1
    
#     while q:
#         v = q.popleft()
#         for e in adj[v]:
#             if not visited[e]:
#                 q.append(e)
#                 visited[e] = True 
#                 count += 1 
#     return count 

# result = []
# max_value = -1 

# for i in range(1,n+1):
#     c = bfs(i)
#     if c> max_value:
#         result = [i]
#         max_value = c 
#     elif c == max_value:
#         result.append(i)
#         max_value = c 

# for e in result:
#     print(e, end=' ')

# 1330.
# A,B = map(int,input().split())

# if A > B:
#     print('>')
# elif A < B:
#     print('<')
# elif A == B:
#     print('==')

# 1753.
# import sys
# import heapq

# input = sys.stdin.readline
# INF = sys.maxsize 

# V, E = map(int,input().split())
# start = int(input())

# weight = [INF]*(V+1)
# graph = [[] for _ in range(V+1)]

# for _ in range(E):
#     u,v,w = map(int,input().split())
#     graph[u].append((w,v))

# def dijkstra(start):
#     heap = []
#     heapq.heappush(heap,(0,start))
#     weight[start] = 0

#     while heap:
#         wei, now = heapq.heappop(heap)

#         if weight[now] < wei:
#             continue 
        
#         for w, next_node in graph[now]:
#             next_wei = w + wei 
#             if next_wei < weight[next_node]:
#                 weight[next_node] = next_wei 
#                 heapq.heappush(heap,(next_wei,next_node))
    
# dijkstra(start) 

# for i in range(1, V+1):
#     if weight[i] == INF:
#         print('INF')
#     else:
#         print(weight[i])

# 10282.
# import sys 
# import heapq 

# input= sys.stdin.readline 
# INF = sys.maxsize 

# def dijkstra(start):
#     heap = []
#     heapq.heappush(heap,(0,start))
#     sec[start] = 0

#     while heap:
#         wei, now = heapq.heappop(heap)
#         if sec[now] < wei:
#             continue 
#         for w, next_node in graph[now]:
#             next_wei = w + wei 
#             if sec[next_node] > next_wei:
#                 sec[next_node] = next_wei 
#                 heapq.heappush(heap,(next_wei,next_node)) 

# case = int(input())
# for _ in range(case):
#     n,d,c = map(int,input().split()) #컴퓨터 개수, 의존성, 해킹당한 컴퓨터번호(start)
#     graph = [[] for _ in range(n + 1)]
#     sec = [INF] * (n + 1)
    
#     for _ in range(d):
#         a,b,s = map(int,input().split()) # 컴퓨터, 컴퓨터, 초(weight)
#         graph[b].append((s,a))
#     dijkstra(c)
#     count = 0
#     max_sec = 0

#     for i in sec:
#         if i != INF:
#             count += 1
#             if i > max_sec:
#                 max_sec = i
#     print(count, max_sec)
        
# 5719.

# from collections import deque
# import sys 
# import heapq 
# input = sys.stdin.readline


# def Djikstra():
#     heap = []
#     heapq.heappush(heap, (0,S))
#     length[S] = 0
#     while heap:
#         l, now = heapq.heappop(heap)
        
#         if length[now] < l:
#             continue 

#         for i in graph[now]:
#             total_leng = l + i[1] 
            
#             if length[i[0]] > total_leng and not drop[now][i[0]]:
#                 length[i[0]] = total_leng
#                 heapq.heappush(heap,(total_leng,i[0]))

# def bfs_back():
#     need_visit = deque()
#     need_visit.append(D)
    
#     while need_visit:
#         now = need_visit.popleft()
#         if now == S:
#             continue 
#         for prev, cost in rev_graph[now]:
#             if length[now] == length[prev] + cost:
#                 drop[prev][now] = True 
#                 need_visit.append(prev)


# while True:
#     N,M = map(int,input().split()) 
#     if N == 0:
#         break 
#     S,D = map(int,input().split()) #시작점, 도착점
#     graph = [[] for _ in range(N+1)]
#     rev_graph = [[] for _ in range(N+1)]

#     for _ in range(M):
#         u,v,p = map(int,input().split()) 
#         graph[u].append((v,p))
#         rev_graph[v].append((u,p))

#     drop = [[False]* (N+1) for _ in range(N+1)]
#     length = [1e9] * (N+1)
#     Djikstra()
#     bfs_back()
#     length = [1e9] * (N+1)
#     Djikstra()

#     if length[D] != 1e9:
#         print(length[D])
#     else:
#         print(-1)

# 1774.
# import math 
# import sys
# input = sys.stdin.readline 

# def get_distance(p1,p2):
#     a = p1[0] - p2[0]
#     b = p1[1] - p2[1] 
#     return math.sqrt((a * a) + (b * b))

# def get_parent(parent, n):
#     if parent[n] == n:
#         return n 
#     return get_parent(parent, parent[n])

# def union_parent(parent, a, b):
#     a = get_parent(parent, a)
#     b = get_parent(parent, b) 
#     if a < b:
#         parent[b] = a 
#     else:
#         parent[a] = b 

# def find_parent(parent, a, b):
#     a = get_parent(parent, a)
#     b = get_parent(parent, b)
#     if a == b:
#         return True 
#     else:
#         return False

# edges = []
# parent = {}
# locations = []

# n,m = map(int,input().split())

# for _ in range(n):
#     x, y = map(int,input().split())
#     locations.append((x,y))

# length = len(locations)

# for i in range(length-1):
#     for j in range(i+1,length):
#         edges.append((i + 1, j + 1, get_distance(locations[i],locations[j])))

# for i in range(1, n+1):
#     parent[i] = i 

# for i in range(m):
#     a, b = map(int,input().split())
#     union_parent(parent, a, b)

# edges.sort(key=lambda data: data[2])

# result = 0 
# for a, b, cost in edges:
#     if not find_parent(parent, a, b):
#         union_parent(parent, a, b)
#         result += cost 

# print("%0.2f" %result)

# 5585.
# wallet = [500,100,50,10,5,1]
# cost = int(input())
# rem_cost = 1000 - cost 
# result_num = 0 

# for money in wallet:
#     if rem_cost >= money:
#         mo_num = rem_cost // money 
#         rem_cost -= money * mo_num 
#         result_num += mo_num 

# print(result_num)

# 1439.
# s = input() 

# count_1 = 0
# count_0 = 0

# if s[0] == '1':
#     count_0 += 1
# else:
#     count_1 += 1

# for i in range(len(s)-1):
#     if s[i] != s[i+1]:
#         if s[i+1] == '1':
#             count_0 += 1

#         else:
#             count_1 += 1

# print(min(count_1,count_0))

# 2012.
# 1.
# stu_num = int(input())

# Rank = [False] * (stu_num + 1)
# Rank[0] = True 
# result = 0
# over_rank = []

# for i in range(1,stu_num+1):
#     pre_rank = int(input())
    
#     if Rank[pre_rank] == False:
#         Rank[pre_rank] = True 
#         continue 

#     over_rank.append(pre_rank)
        
# for rank in over_rank:
#     for j in range(1,stu_num+1):
#         if Rank[j] == False:
#             Rank[j] = True 
#             result += abs(rank - j)

# print(result)

# 2.
# n = int(input())
# array = []

# for _ in range(n):
#     array.append(int(input()))

# array.sort()

# result = 0
# for i in range(1,len(array)+1):
#     result += abs(i-array[i-1])
# print(result)

# 1092.
# import sys 

# n = int(input())  #크레인 개수입력
# cranes = list(map(int,input().split())) #크레인 무게제한입력  

# m = int(input()) #박스입력
# boxes = list(map(int,input().split())) #박스 무게입력

# #모든 박스를 옮길 수 없는 경우
# if max(cranes) < max(boxes):
#     print(-1)
#     sys.exit()

# #각 크레인이 현재 옮겨야 하는 박스의 번호(0부터 시작)
# # 현재 크레인이 몇번째 박스를 움직이고 있다. 
# positions = [0] * n 

# #각 박스를 옮겼는지의 여부
# checked = [False] * m 

# #최적의 해를 구해야 하므로 , 내림차순 정렬 
# cranes.sort(reverse=True)
# boxes.sort(reverse=True)

# result = 0
# count = 0

# while True:
#     if count == len(boxes): #다 옮겼으면 종료
#         break 
#     for i in range(n): #모든 크레인에 대하여 각각 처리
#         while positions[i] < len(boxes): #해당 크레인에서 옮긴 박스 수가 전체 박스 수보다 작을 때까지만 실행
#             #아직 안 옮긴 박스 중에서, 옮길 수 있는 박스를 만날때 까지 반복
#             if not checked[positions[i]] and cranes[i] >= boxes[positions[i]]: #박스가 안옮겨지고 무게조건이 성립할 때
#                 checked[positions[i]] = True #박스가 옮겨졌다고 만들어 준다.
#                 positions[i] += 1 
#                 count += 1 
#                 break 
#             positions[i] += 1 
#     result += 1 

# print(result)

# 2212.
# import sys

# n = int(input())
# k = int(input())

# sensor = list(map(int,input().split()))

# dist_dif = []
# if k >= n:
#     print(0)
#     sys.exit()

# sensor.sort()

# for i in range(n-1):
#     dist_dif.append(sensor[i+1]-sensor[i])

# dist_dif.sort()

# for i in range(k-1):
#     dist_dif.pop()

# print(sum(dist_dif))

# 1461. 
# import heapq

# n,m = map(int,input().split()) #책수,들수있는 책수
# posi = list(map(int,input().split()))

# positive = []
# negative = [] 

# posi.sort()
# largest = max(-posi[0], posi[-1]) 

# for i in posi:
#     if i > 0: 
#         heapq.heappush(positive,-i)
#     else:
#         heapq.heappush(negative,i)

# result = 0
# while positive:
#     result += heapq.heappop(positive)
#     for _ in range(m-1):
#         if positive:
#             heapq.heappop(positive)

# while negative:
#     result += heapq.heappop(negative)
#     for _ in range(m-1):
#         if negative:
#             heapq.heappop(negative)

# print(-result*2 - largest)

# 11047.

# n,k = map(int,input().split())

# coins = []
# for _ in range(n):
#     coin = int(input())
#     coins.append(coin)

# coins.sort(reverse = True)
# count = 0

# for i in coins:
#     c_num = k // i 
#     k -= c_num * i 
#     count += c_num 

# print(count)

# 1781. 
# import heapq
# n = int(input())

# array = []
# for i in range(n):
#     d, c = map(int,input().split())
#     array.append((d,c))

# array.sort()
# q = []
# for dead,cup in array:
#     heapq.heappush(q,cup)
#     # 데드라인을 초과하는 경우에는 최소 원소 제거하기
#     if dead < len(q):
#         heapq.heappop(q)

# print(sum(q))

# 9663.
# 1. 
# N = int(input())

# def available(column,board_now):
#     current_row = len(board_now)

#     for r in range(current_row):
#         if board_now[r] == column and abs(board_now[r]-column) == current_row - r:
#             return False 
#     return True

# def DFS(N,row,board_now,result):
#     if row == N:
#         result.append(board_now[:])
#         return 
    
#     for column in range(N):
#         if available(column,board_now):
#             board_now.append(column)
#             DFS(N,row+1,board_now,result)
            
#             board_now.pop()

# def N_Queen(N):
#     result = []
#     DFS(N,0,[],result)
    
#     return result

# 2.
# def check(x):
#     for i in range(x): #현재 queen이 있는 행만 확인 
#         if board[x] == board[i]: #같은 열 확인
#             return False 
#         if abs(board[x]-board[i]) == abs(x-i): #대각선 확인
#             return False 
    
#     return True 

# def dfs(x): # x - row
#     global result 
#     if x == n: #최종 결과 나오는 것
#         result += 1
    
#     else: 
#         for col in range(n): # 열 확인
#             board[x] = col #board 에다가 일일이 넣어 준다.
#             if check(x): # 해당 상황 check하기 
#                 dfs(x+1) #다음 단계 실행


# n = int(input())
# result = 0
# board = [0] * n
# dfs(0)
# print(result)

# 3.
# def check(x):
#     for i in range(x):
#         if board[x] == board[i]:
#             return False 
#         if abs(board[x]-board[i]) == abs(x-i):
#             return False 
#     return True 

# def dfs(x,result):
#     if x == n:
#         result.append(board[:])
#         return 

#     else:
#         for col in range(n):
#             board[x] = col 
#             if check(x):
#                 dfs(x+1,result)


# n = int(input())
# board = [0] *n
# result = []
# dfs(0,result)
# print(result)

# 10539.
# 1.
# n = int(input())
# B = list(map(int,input().split()))

# A_sum = 0
# result = []

# for i in range(n):
#     A = B[i] * (i+1) - A_sum 
#     result.append(A)
#     A_sum += A 

# # for i in result:
# #     print(i,end=' ')

# print(' '.join(map(str,result)))

# 2. 
# n, b= int(input()), list(map(int,input().split()))
# a = [b[0]]

# for i in range(1,n):
#     a.append((b[i]*(i+1)-sum(a)))

# for i in a:
#     print(i, end=' ')

# 1987.
# 1.
# def dfs(board):
#     global result

#     visit = []
#     need_visit = []
#     need_visit.append((0,0))
#     direction = [(-1,0),(0,-1),(0,1),(1,0)]

#     while need_visit:
#         print(visit)
#         print(need_visit)
#         c_row, c_col = need_visit.pop(0)
    
#         alpha = board[c_row][c_col] 
#         if alpha not in visit:
#             visit.append(alpha)
#             result += 1
                
#             for n_row,n_col in direction:
#                 if (0 <= c_row+n_row and c_row+n_row < r and 0 <= c_col+n_col and c_col+n_col < c):
#                     need_visit.append((c_row+n_row,c_col+n_col))

# r,c = map(int,input().split())

# board=[]
# result = 0

# for i in range(r):
#     word = list(input()) 
#     board.append(word)
# dfs(board)
# print(result)

# 2.
# #이동좌표(상,하,좌,우)
# dx = [-1,1,0,0]
# dy = [0,0,-1,1]

# def bfs(x,y):
#     global result 
#     #동일한 경우는 한번만 계산하기 위해서 집합(set)사용
#     q = set()
#     q.add((x,y,array[x][y]))

#     while q:
#         x,y,step = q.pop() 
#         result = max(result,len(step))

#         #네 방향으로 이동하는 경우를 각각 확인
#         for i in range(4):
#             nx = x + dx[i]
#             ny = y + dy[i]

#             #이동할 수 있는 위치이면서, 새로운 알파벳인 경우
#             if (0 <= nx and nx < r and 0 <= ny and ny < c and array[nx][ny] not in step):
#                 q.add((nx,ny,step+array[nx][ny]))

# # 전체 보드 데이터를 입력 받습니다.
# r,c = map(int,input().split())
# array = []
# for _ in range(r):
#     array.append(input())

# #백 트래킹 수행 결과를 출력합니다.
# result = 0
# bfs(0,0)
# print(result)

# 1759.
# 1.
# from itertools import combinations 

# vowels = ('a','e','i','o','u')
# l, c = map(int,input().split())

# #가능한 암호를 사전식으로 출력해야 하므로 정렬 수행 
# array = input().split() 
# array.sort() 

# #길이가 l인 모든 암호 조합을 확인 
# for password in combinations(array,l):
#     #모음의 개수를 세기
#     count = 0
#     for i in password:
#         if i in vowels:
#             count += 1 
    
#     # 최소 한 개의 모음과 최소 두개의 자음이 있는 경우 출력
#     if count >= 1 and count <= l-2:
#         print(''.join(password))


# 2.
# import copy

# result = []
# string = []
# visited = [] 

# #조함(combination) 함수 구현 
# def combination(array, length, index):
#     #길이가 length인 모든 조합 찾기 
#     if len(string) == length:
#         result.append(copy.deepcopy(string))
#         return 
#     #각 원소를 한번씩만 뽑도록 구성
#     for i in range(index,len(array)):
#         if i in visited:
#             continue 
#         string.append(array[i])
#         visited.append(i)
#         combination(array,length,i+1)
#         string.pop() 
#         visited.pop() 

# vowels = ('a','e','i','o','u')
# l,c = map(int,input().split())

# #가능한 암호를 사전식으로 출력해야 하므로 정렬 수행
# array = input().split()
# array.sort()

# combination(array,1,0)

# #길이가 1인 모든 암호 조합을 확인 
# for password in result:
#     #모음의 개수를 세기
#     count = 0
#     for i in password:
#         if i in vowels:
#             count += 1

#     # 최소 한개의 모음과 최소 두개의 자음이 있는 경우
#     if count >= 1 and count <= l-2:
#         print('',join(password))

# 17269.

# board = {'A':3,'B':2,'C':1,'D':2,'E':4,'F':3,'G':1,'H':3,'I':1,'J':1,'K':3,'L':1,'M':3,'N':2,'O':1,'P':2,'Q':2,'R':2,'S':1,'T':2,'U':1,'V':1,'W':1,'X':2,'Y':2,'Z':1}

# n,m = map(int,input().split()) #이름 길이 
# name1, name2 = list(input().split())

# min_num = min(n,m)
# string = ''

# for i in range(min_num):
#     string += name1[i] + name2[i]

# string += name1[min_num:] + name2[min_num:]
# string = string.upper() 

# num = [board[i] for i in string]

# for i in range(n+m-2):
#     for j in range(n+m-1-i):
#         num[j] += num[j+1]

# print('{}%'.format((num[0] % 10)*10 + num[1]%10))

# 17389.
# 1.
# n = int(input())

# s = input() #정답확인

# score = [0] * (n)
# bonus = 0

# for i in range(n):
#     if s[i] == 'O':
#         score[i] = (i+1) + bonus
#         bonus += 1
#     else:
#         bonus = 0

# print(sum(score))

# 2.
# n,s = input(),input() 

# score,bonus = 0,0 

# for idx,ox in enumerate(s):
#     if ox == 'O':
#         score,bonus = score + idx +1 + bonus , bonus + 1
#     else:
#         bonus = 0
# print(score)

# n = int(input())
# a = {i : 1 for i in map(int,input().split())}
# m = input() 
# for i in list(map(int,input().split())):
#     print(a.get(i,0)) #i(key)가 dict함수 a에 있으면 해당 value출력하고 없으면 0 출력한다.

# 16165.
# 1.
# n, m = map(int,input().split()) # 걸그룹수, 문제수

# team = dict() 

# for _ in range(n):
#     team_name = input() 
#     team_num = int(input())

#     for _ in range(team_num):
#         member = input() 
#         team[member] = team_name #멤버이름 : 걸그룹이름 으로 dict에 저장

# #퀴즈 질문
# for _ in range(m):
#     quiz = input() 
#     number = int(input()) #1 - team이름 찾기 2 - 멤버찾기

#     if number == 1:                
#         print(team[quiz])
    
#     elif number == 0:
#         result = []
#         for m_name, t_name in team.items():
#             if t_name == quiz:
#                 result.append(m_name)
#         result.sort()
#         for i in result:
#             print(i)

# 2.
# n, m = map(int,input().split())
# team_mem, mem_team = {}, {} 

# for i in range(n):
#     team_name, mem_num = input(), int(input())
#     team_mem[team_name] = []
#     for j in range(mem_num):
#         name = input() 
#         team_mem[team_name].append(name)
#         mem_team[name] = team_name 

# for i in range(m):
#     name, q = input(), int(input())
#     if q:
#         print(mem_team[name])
#     else:
#         for mem in sorted(team_mem[name]):
#             print(mem)

# 17224.
# 1.
# n,l,k = map(int,input().split()) #문제수,역량,풀 수 있는 문제 개수

# problem = []
# for _ in range(n):
#     easy,hard = map(int,input().split())
#     problem.append((easy,hard))

# problem.sort() 

# score = 0

# for easy_p, hard_p in problem:
#     if k != 0:
#         if easy_p <= l:
#             score += 100 
#             k -= 1
#             if hard_p <= l:
#                 score += 40 
#     else:
#         break

# print(score)

# 2.

# n,l,k = map(int,input().split())

# easy, hard = 0, 0 

# for i in range(n):
#     sub1, sub2 = map(int,input().split())
#     if sub2 <= l:
#         hard += 1
#     elif sub1 <= l:
#         easy += 1 

# #hard 
# result = min(hard,k) * 140 

# #easy 
# if hard < k:
#     result += min(k-hard, easy) * 100

# print(result)

# 9037.
# def candy_check(n,candy):
#     for i in range(n):
#         if candy[i] % 2 == 1:
#             candy[i] += 1 
    
#     return len(set(candy)) == 1 

# def share_candy(n,candy):
#     tmp = [0 for _ in range(n)]
    
#     for i in range(n):
#         candy[i] //= 2 
#         tmp[(i+1)%n] = candy[i]
    
#     for j in range(n):
#         candy[j] += tmp[j]

#     return candy

# t = int(input())
# for _ in range(t):
#     n = int(input())
#     candy = list(map(int,input().split()))

#     count = 0 

#     while not candy_check(n,candy):
#         count += 1 
#         share_candy(n,candy)
#     print(count)

# 16769.
# 1.
# size = []
# milk = []

# for _ in range(3):
#     c,m = map(int,input().split()) #size와 우유양 
#     size.append(c)
#     milk.append(m)

# cnt = 0
# while cnt < 100:
#     for i in range(3):
#         if milk[(i+1)%3] + milk[i] > size[(i+1)%3]:
#             milk[i] = milk[(i+1)%3] + milk[i] - size[(i+1)%3]
#             milk[(i+1)%3] = size[(i+1)%3]
#             cnt += 1
#             if cnt == 100:
#                 break
#         else:
#             milk[(i+1)%3] += milk[i] 
#             milk[i] = 0
#             cnt += 1
#             if cnt == 100:
#                 break

# for i in milk:
#     print(i)

# 2.
# c,m = list(),list() 

# for i in range(3):
#     a,b = map(int,input().split())
#     c.append(a)
#     m.append(b)

# for i in range(100):
#     idx = i % 3
#     nxt = (i+1) % 3
#     m[idx], m[nxt] = max((m[idx]+m[nxt])-c[nxt],0), min(c[nxt], m[idx]+m[nxt])
# for i in m:
#     print(i)

# 1074.
# n,r,c = map(int,input().split())

# #z: 0,0을 기준으로 x,y의 숫자
# def z(sz,x,y):
#     if sz == 1:
#         return 0 

#     sz //= 2
#     #sz가 2일때 z움직이는 for 구문
#     for i in range(2):
#         for j in range(2):
#             #z끝까지 다움직였는지 확인하는 if 구문(0부터 시작)
#             #i = 1, j = 1이면 해당 4분야에 값없는것
#             if x < sz * (i+1) and y < sz * (j+1):
#                 # 다음 작은 정사각형으로 이동 
#                 # return 일때는 큰 정사각형 원하는 값 유무 파악하여 3->15->... 더해준다.
#                 return (i*2+j) * sz * sz + z(sz, x-sz*i, y-sz*j)

# print(z(2**n,r ,c ))

# 2480.
# 1.
# dice = list(map(int,input().split()))

# dice.sort() 

# if len(set(dice)) == 1:
#     print(10000+dice[-1]*1000)
# if len(set(dice)) == 2:
#     num = dict() 
#     for i in dice:
#         num[i] = 0
#     for i in dice:
#         num[i] += 1
    
#     for i in dice:
#         if num[i] == 2:
#             print(1000+i*100)
#             break

# if len(set(dice)) == 3:
#     print(dice[-1]*100)

# 2.
# lst = sorted(list(map(int,input().split())))

# if len(set(lst)) == 1:
#     print(10000+lst[0]*1000)

# elif len(set(lst)) == 2: #3개중에서 2개가 같은 값이기 때문에 항상 lst[1]의 값은 중복된 값이다.
#     print(1000 + lst[1]*100)

# else:
#     print(lst[2]*100)

# 2484.
# 1.
# def count_dice_num(dice):
#     num = dict() 
#     for i in dice:
#         num[i] = 0
#     for j in dice:
#         num[j]+= 1
#     return num 

# money = []

# for _ in range(int(input())):
#     dice = sorted(list(map(int,input().split())))

#     if len(set(dice)) == 1:
#         money.append(50000+dice[0]*5000)

        
#     elif len(set(dice)) == 2:
#         num = count_dice_num(dice)
#         for k in dice:
#             if num[k] == 3:
#                 money.append(10000+k*1000)
#                 break 
#             elif num[k] == 2:
#                 money.append(2000+dice[0]*500+dice[-1]*500)
#                 break
                
#     elif len(set(dice)) == 3:
#         num = count_dice_num(dice)
#         for i in dice:
#             if num[i] == 2:
#                 money.append(1000+i*100)
#                 break

#     elif len(set(dice)) == 4:
#         money.append(dice[-1]*100)

# print(max(money))

# 2.
# def money():
#     lst = sorted(list(map(int,input().split())))
#     if len(set(lst)) == 1:
#         return lst[0] * 5000 + 50000

#     if len(set(lst)) == 2:
#         if lst[1] == lst[2]:
#             return 10000 + lst[1] * 1000 
#         else:
#             return 2000 + (lst[1]+lst[2])*500 

#     for i in range(3):
#         if lst[i] == lst[i+1]:
#             return 1000 + lst[i] * 100
#     return lst[-1] * 100

# n = int(input())

# print(max(money() for i in range(n)))

# 16675.
# 1.
# ms = []
# tk = []
# r_p_s = input().split() 

# ms.extend(r_p_s[:2])
# tk.extend(r_p_s[2:])

# if len(set(ms)) == 2 and len(set(tk)) == 2:
#     print('?')

# elif len(set(ms)) == 1 and len(set(tk)) == 1:
#     if ms[0] == 'R':
#         if tk[0] == 'S':
#             print('MS')
#         elif tk[0] == 'P':
#             print('TK')
#         else:
#             print('?')
    
#     if ms[0] == 'P':
#         if tk[0] == 'R':
#             print('MS')
#         elif tk[0] == 'S':
#             print('TK')
#         else:
#             print('?')
    
#     if ms[0] == 'S':
#         if tk[0] == 'P':
#             print('MS')
#         elif tk[0] == 'R':
#             print('TK')
#         else:
#             print('?')

        
# else:
#     if len(set(ms)) == 1:
#         if ms[0] == 'P':
#             if 'S' in tk:
#                 print('TK')
#             else:
#                 print('?')
#         elif ms[0] == 'R':
#             if 'P' in tk:
#                 print('TK')
#             else:
#                 print('?')
#         elif ms[0] == 'S':
#             if 'R' in tk:
#                 print('TK')
#             else:
#                 print('?')
    
#     else: 
#         if tk[0] == 'P':
#             if 'S' in ms:
#                 print('MS')
#             else:
#                 print('?')

#         elif tk[0] == 'R':
#             if 'P' in ms:
#                 print('MS')
#             else:
#                 print('?')

#         elif tk[0] == 'S':
#             if 'R' in ms:
#                 print('MS')
#             else:
#                 print('?')

# 2.
# ML, MR, TL, TR = ('RSP'.index(i) for i in input().split())

# if ML == MR and (ML+2) % 3 in [TL, TR]:
#     print('TK') 

# elif TL == TR and (TL+2) % 3 in [ML,MR]:
#     print('MS')

# else:
#     print('?')


# 17413.
# s,tmp = input(), "" 

# ck = False
# result = ""

# for text in s:
#     if text == ' ':
#         if ck: 
#             result += " "
#         else:
#             result += tmp[::-1] + " "
#             tmp = ""
    
#     elif text == '<':
#         ck = True 
#         result += tmp[::-1] + '<'
#         tmp = ""
    
#     elif text == '>':
#         ck = False
#         result += '>'
    
#     else:
#         if ck:
#             result += text
#         else:
#             tmp += text
    
# result += tmp[::-1]
# print(result)

# 2293. 동전1
# n,k = map(int,input().split())

# coin = []

# for _ in range(n):
#     coin.append(int(input()))

# dp = [0 for i in range(10001)]
# dp[0] = 1
# for c in coin:
#     for i in range(c,k+1):
#         dp[i] += dp[i-c]

# print(dp[k])



# 16956.

# r,c = map(int,input().split())

# board = [list(input()) for _ in range(r)]

# dx,dy = [0,-1,0,1],[1,0,-1,0]
# ck = False 

# for x in range(r):
#     for y in range(c):
#         if board[x][y] == 'W':
#             for i in range(4):
#                 xx,yy = x + dx[i], y + dy[i] 
#                 if xx == r or xx < 0 or yy == c or yy < 0:
#                     continue
#                 if board[xx][yy] == 'S':
#                     ck = True
# if ck:
#     print(0)
# else:
#     print(1)
#     for x in range(r):
#         for y in range(c):
#             if board[x][y] not in 'SW':
#                 board[x][y] = 'D'
            
#     for x in board:
#         print(''.join(x))
                    


# 14620.
# 1.
# def check(x,y):
    
#     count = 0
#     for i in range(4):
#         xx,yy = x+dx[i],y+dy[i]
#         if condition[xx][yy] == False:
#             count+= 1
#             continue
    
#     if count == 4:
#         return True 
#     else:
#         return False


# n = int(input()) #화단의 한변의 길이

# m = [list(map(int,input().split())) for _ in range(n)]
# condition = [[False]*n for _ in range(n)]

# board = []

# dx, dy = [0,-1,0,1], [1,0,-1,0] #반시계방향

# for x in range(1,n-1):
#     for y in range(1,n-1):
#         cost = m[x][y]
#         for i in range(4):
#             cost += m[x+dx[i]][y+dy[i]]
        
#         board.append((cost,x,y))
         
# board= sorted(board, key = lambda x: x[0])

# result = 0 
# flow_num = 0
# for co,x,y in board:

#     if flow_num < 3:
#         if condition[x][y] == False and check(x,y):
#             result += co
#             flow_num +=1
#             condition[x][y] = True
#             for i in range(4):
#                 condition[x+dx[i]][y+dy[i]] = True
#     else:
#         break
# print(result)
                    
# 2.
# n = int(input())
# g = [list(map(int,input().split()))for _ in range(n)]
# ans = 10000

# dx,dy = [0,0,0,1,-1],[0,1,-1,0,0]

# def ck(lst): #a, b, c 
#     ret = 0
#     flow = []
#     for flower in lst:
#         # for 문 2개를 써야할 2차 배열을 1나로 표현한 것
#         x = flower // n # 행 나타내는 것  
#         y = flower % n # 열 나타내는 것
        
#         # 표를 넘어가는 곳 없애기
#         if x == 0 or x == n-1 or y == 0 or y == n-1:
#             return 10000 
        
#         for w in range(5):
#             flow.append((x+dx[w],y+dy[w]))
#             ret += g[x+dx[w]][y+dy[w]]
#     #겹치는 것이 있는 지 파악하기 위한 것
#     if len(set(flow)) != 15:
#         return 10000
#     return ret 

# #변수 조사 
# # 0,0 은 0 * n + 0 으로 표현할 수 있다.
# #위에서 넘버링을 하여 0~n*n-1로 표현시키는 것
# #꽃이 3개이기 때문에 for 3개(모든 상황파악)
# for i in range(n*n):
#     for j in range(i+1,n*n):
#         for k in range(j+1,n*n):
#             ans = min(ans,ck([i,j,k]))
# print(ans)

# 1012.
# def ck(x,y):
#     check[x][y] = True
    
#     dx,dy = [0,-1,0,1],[1,0,-1,0]
#     for i in range(4):
#         xx,yy = x+dx[i],y+dy[i]

#         if xx < 0 or xx == n or yy <0 or yy == m:
#             continue 

#         if board[xx][yy] and not check[xx][yy]:   
#             ck(xx,yy)

# for _ in range(int(input())):
#     m,n,k = map(int,input().split()) #가로길이 , 세로길이

#     board = [[0]*m for _ in range(n)]
#     check = [[False]*m for _ in range(n)]

#     for _ in range(k):
#         y,x = map(int,input().split()) 
#         board[x][y] = 1 
    
#     ans = 0
#     for x_b in range(n):
#         for y_b in range(m):
#             if board[x_b][y_b] and not check[x_b][y_b]:
#                 ck(x_b,y_b)
#                 ans += 1

#     print(ans)


# 1932.

# n = int(input())
# #dp[i][j] : i,j 도착했을 때 최댓값 
# #dp[i][j] = max(dp[i-1][j-1],dp[i-1][j]) + a[i][j]

# a = [[0 for _ in range(n+1)] for i in range(n+1)]
# dp = [[0 for _ in range(n+1)] for i in range(n+1)]


# for i in range(1,n+1):
#     tmp = list(map(int,input().split()))
#     for j in range(1,i+1):
#         a[i][j] = tmp[j-1]

# for i in range(1,n+1):
#     for j in range(1,i+1):
#         dp[i][j] = max(dp[i-1][j-1],dp[i-1][j]) + a[i][j]

# print(max(dp[-1]))

# 16768.

# n,k = map(int,input().split()) 
# m = [list(input()) for _ in range(n)]
# ck = [[False] * 10 for _ in range(n)]
# ck2 = [[False] * 10 for _ in range(n)]

# dx,dy = [0,1,0,-1],[1,0,-1,0]


# def dfs(x,y):
#     ck[x][y] = True 
#     ret = 1 

#     for i in range(4): 
#         xx,yy = x+dx[i],y+dy[i]
#         if xx < 0 or xx >= n or yy < 0 or yy >= 10:
#             continue 
#         if ck[xx][yy] or m[x][y] != m[xx][yy]:
#             continue 
        
#         ret += dfs(xx,yy)

#     return ret 

# def dfs2(x,y,val):
#     ck2[x][y] = True 
#     m[x][y] = '0'
#     for i in range(4):
#         xx,yy = x+dx[i],y+dy[i]
#         if xx < 0 or xx >= n or yy < 0 or yy >= 10:
#             continue 
#         if ck2[xx][yy] or m[xx][yy] != val:
#             continue 
#         dfs2(xx,yy,val)

# def down():
#     for i in range(10):
#         tmp = [] 
#         for j in range(n):
#             if m[j][i] != '0':
#                 tmp.append(m[j][i])

#         for j in range(n-len(tmp)):
#             m[j][i] = '0'

#         for j in range(n-len(tmp),n):
#             m[j][i] = tmp[j-(n-len(tmp))]


# while True:
#     exist = False 
#     ck = [[False]* 10 for _ in range(n)]
#     ck2 = [[False]* 10 for _ in range(n)] 
    
#     for i in range(n):
#         for j in range(10):
#             if m[i][j] =='0' or ck[i][j]:
#                 continue 
#             res = dfs(i, j)

#             if res >= k:
#                 result += 1
#                 dfs2(i,j,m[i][j])
#                 exist = True 
#     if not exist:
#         break 
#     down() 

# for i in m:
#     print(''.join(i))

# 11559. pass문제

# m = [list(input()) for _ in range(12)]
# ck = [[False]*6 for _ in range(12)]
# ck2 = [[False]*6 for _ in range(12)]

# dx,dy = [0,-1,0,1],[1,0,-1,0]
# result = 0


# def dfs(x,y): #개수 찾기
#     ck[x][y] = True 
#     ret = 1 

#     for i in range(4):
#         xx,yy = x+dx[i],y+dy[i]

#         if xx < 0 or xx >= 12 or yy < 0 or yy >= 6:
#             continue 

#         if ck[xx][yy] or m[x][y] != m[xx][yy]:
#             continue 
        
#         ret += dfs(xx,yy)  

#     return ret

# def dfs2(x,y,val): # 터트리는 곳

#     ck2[x][y] = True 
#     m[x][y] = '.'

#     for i in range(4):
#         xx,yy = x+dx[i], y+dy[i]

#         if xx < 0 or xx >= 12 or yy < 0 or yy >= 6:
#             continue 

#         if ck2[xx][yy] or m[xx][yy] != val:
#             continue
#         dfs2(xx,yy,val)

     
# def down(): # 밑으로 다 보내버리기

#     for i in range(6):
#         tmp = []
#         for j in range(12):
#             if m[j][i] != '.':
#                 tmp.append(m[j][i])

#         for j in range(12-len(tmp)):
#             m[j][i] = '.'

#         for j in range(12-len(tmp),12):
#             m[j][i] = tmp[j-(12-len(tmp))]
    

# while True: #실행부분
#     exist = False
#     ck = [[False]*6 for _ in range(12)]
#     ck2 = [[False]*6 for _ in range(12)]

#     for i in range(12):
#         for j in range(6):
#             if m[i][j] == '.' or ck[i][j]:
#                 continue 

#             count = dfs(i,j)

#             if count >= 4:
#                 result += 1
#                 dfs2(i,j,m[i][j])
#                 exist = True 
#     if not exist:
#         break 
#     down()

# print(result)

# 17406.
# from copy import deepcopy 

# n,m,k = map(int,input().split())
# a = [list(map(int,input().split())) for _ in range(n)]
# q = [tuple(map(int,input().split())) for _ in range(k)]
# dx,dy = [1,0,-1,0],[0,-1,0,1]  #남서북동 순서

# ans = 10000

# def value(arr):
#     return min(sum(i) for i in arr)

# def convert(arr,qry):
#     (r,c,s) = qry 
#     r,c = r-1,c-1 
#     new_arr = deepcopy(arr)
#     for i in range(1,s+1):
#         rr,cc = r-i,c+i 
#         for w in range(4):
#             for d in range(i*2):
#                 rrr,ccc = rr+dx[w], cc+dy[w]
#                 new_arr[rrr][ccc] = arr[rr][cc]
#                 rr,cc = rrr,ccc 
#     return new_arr 

# def dfs(arr,qry):
#     global ans 
#     if sum(qry) == k:
#         ans = min(ans, value(arr))
#         return 
    
#     for i in range(k):
#         if qry[i]: #qry를 처리했다면 continue
#             continue 
#         new_arr = convert(arr,q[i])
#         qry[i] = 1
#         dfs(new_arr,qry)
#         qry[i] = 0 #처리를 안했다고 한다 (백트랙킹 기법)


# dfs(a,[0 for i in range(k)])
# print(ans)

# 1919.

# a = list(input())
# b = list(input())
# ck_a = [False for _ in range(len(a))]
# ck_b = [False for _ in range(len(b))]

# for i in range(len(a)):
#     for j in range(len(b)):
#         if ck_b[j] == True or ck_a[i] == True:
#             continue 
#         if a[i] == b[j]:
#             ck_a[i] = True 
#             ck_b[j] = True

# result = 0

# for i in range(len(a)):
#     if ck_a[i] == False:
#         result += 1

# for j in range(len(b)):
#     if ck_b[j] == False:
#         result += 1

# print(result)

# 10818.
# n = int(input())
# num = list(map(int,input().split()))

# num.sort() 
# print(num[0],num[-1])

# 10950.
# for _ in range(int(input())):
#     num1, num2 = map(int,input().split())
#     print(num1+num2)

# 2675.
# for _ in range(int(input())):
#     result = ""
#     num, string = map(str,input().split())
#     for i in string:
#         result += i*int(num) 
#     print(result)

# 25262.
# nums = []

# for _ in range(9):
#     num = int(input())
#     nums.append(num)

# new_num = sorted(nums)
# print(new_num[-1])
# print(nums.index(new_num[-1])+1)

# 10952.

# while True:
#     num1,num2 = map(int,input().split())
    
#     if num1 == 0:
#         break 

#     print(num1+num2)


# 12100.
# from copy import deepcopy 

# n = int(input()) 
# board = [list(map(int,input().split())) for i in range(n)]

# #90도 회전하기 (암기하기)
# def rotate90(b,n):
#     # nb = [0 for i in range(n)]
#     nb = deepcopy(b)
#     for i in range(n):
#         for j in range(n):
#             nb[j][n-i-1] = b[i][j]

#     return nb

# #한줄로 다음 배열을 만드는 것
# #좌측으로 옮기기 
# def convert(lst,n):
#     new_list = [i for i in lst if i] #0이 아닌수만 남긴다.
#     for i in range(1,len(new_list)):
#         if new_list[i-1] == new_list[i]:
#             new_list[i-1] *= 2
#             new_list[i] = 0
#     new_list = [i for i in new_list if i]
#     return new_list + [0] * (n-len(new_list))


# def dfs(n,b, count):
#     ret = max([max(i) for i in b]) #board의 최대값
#     if count == 0:
#         return ret 
    
#     for _ in range(4): #동서남북 4방향이므로 4
        # x = [convert(i,n) for i in board] # 행 마다 convert실행 해서 새로운 행을 만들어 주는 것
#         if x!= b: #변화가 있는지 없는지 파악(변화가 있는 경우)
#             ret = max(ret, dfs(n, x, count-1)) 
#         b = rotate90(b,n) #board 90도 돌려주기(변화가 없는경우)

#     return ret

# print(dfs(n, board, 5))


# 11055.

# 1.
# import copy 

# n = int(input()) #수열의 크기 
# a = list(map(int,input().split())) #수열을 이루고 있는 값

# #dp[i] : i 까지 왔을 때, 합의 최대 
# dp = copy.deepcopy(a)

# for i in range(1,n):
#     for j in range(i):
#         if a[i] > a[j]:
#             dp[i] = max(dp[i], a[i]+dp[j])

# print(max(dp))

# 2.
# import copy 

# n = int(input()) #수열의 크기 
# a = list(map(int,input().split())) #수열을 이루고 있는 값

# #dp[i] : i 까지 왔을 때, 합의 최대 
# dp = copy.deepcopy(a)
# rev = [i for i in range(n)]
# idx = 0

# for i in range(1,n):
#     for j in range(i):
#         if a[i] > a[j] and dp[i] < dp[j] + a[i]:
#             dp[i] = a[i] + dp[j]
#             rev[i] = j 

#     if dp[idx] < dp[i]:
#         idx = i 

# print(dp[idx])
# while rev[idx] != idx:
#     print(a[idx],sep = " ")
#     idx = rev[idx]

# print(a[idx])

# 2167.
# import sys
# input = sys.stdin.readline

# n,m = map(int,input().split()) # 행,열
# a = [list(map(int,input().split())) for _ in range(n)]

# k = int(input()) #합을 구할 부분의 개수

# dp = [[0] * (m+1) for _ in range(n+1)]

# for i in range(1,n+1):
#     for j in range(1,m+1):
#         dp[i][j] =  a[i-1][j-1] + dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] # (1,1)의 값부터 i,j 까지의 사각형의 합을 만든 것

# for _ in range(k):
#     i,j,x,y = map(int,input().split())
#     print(dp[x][y] -dp[i-1][y]-dp[x][j-1]+dp[i-1][j-1])

# 1915.
# n,m = map(int,input().split())
# a = [[0 for _ in range(m+1)] for i in range(n+1)]

# #dp[i][j] = i,j 까지 있을 때, 가장 큰 정사각형의 한 변의 길이
# #dp[i][j] = min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1]) + 1
# dp = [[0 for _ in range(m+1)] for i in range(n+1)]

# for i in range(n):
#     for idx, j in enumerate(list(map(int,list(input())))):
#         a[i+1][idx+1] = j 

# mx = 0

# for i in range(1,n+1):
#     for j in range(1,m+1):
#         if a[i][j]:
#             dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1
#             mx = max(dp[i][j],mx)

# print(mx**2)
# # print(max([max(i) for i in dp]) ** 2)


# 1439.
# 1.
# s = input() 

# num_0 = 0
# num_1 = 0

# if s[0] == '0':
#     num_0 += 1 
# else:
#     num_1 += 1

# for i in range(len(s)-1):
#     if s[i] != s[i+1]:
#         if s[i+1] == '0':
#             num_0 += 1 
#         else:
#             num_1 += 1 

# print(min(num_0,num_1))

# 2.
# s = input() 
# tot = 0
# for i in range(1,len(s)):
#     if s[i] != s[i+1]:
#         tot += 1 

# print((tot+1)//2)

# 16676.

# n = input() #최대값 

# if int(n) == 0:
# 	print(1)

# else:
# 	standard = '1' * len(n) 
# 	if int(n) < int(standard):
# 		print(len(standard)-1)
# 	else:
# 		print(len(standard))
# 2.
# n = input()
# s= '1'*len(n)

# if len(n) == 1:
# 	print(1)
# elif int(n) >= int(s):
# 	print(len(n))
# else:
# 	print(len(n)-1)

# 12849.
# # 0: 정보과학관 
# # 1: 전산관
# # 2: 미래관
# # 3: 신양관
# # 4: 한경직기념관
# # 5: 진리관
# # 6: 학생회관
# # 7: 형남공학관

# #dp는 0분에 어떤 지정장소에 도착할 수 있는 상태(정보과학관에서 각 도착지점까지의 경로수)
# dp = [1, 0, 0, 0, 0, 0, 0, 0] 

# #각 지점에 올 수 있는 경우는 각 지점의 이웃에서 1분 후에 오는 경우 
# def nxt(state): # 1분뒤에 초기 지점(정보과학관) 부터 각 지점에 올 수 있는 경우
# 	tmp = [0 for _ in range(8)]
# 	tmp[0] = state[1] + state[2]
# 	tmp[1] = state[0] + state[2] + state[3]
# 	tmp[2] = state[0] + state[1] + state[3] + state[4]
# 	tmp[3] = state[1] + state[2] + state[4] + state[5]
# 	tmp[4] = state[2] + state[3] + state[5] + state[7]  
# 	tmp[5] = state[3] + state[4] + state[6]
# 	tmp[6] = state[5] + state[7]
# 	tmp[7] = state[4] + state[6]

# 	for i in range(8):
# 		tmp[i] %= 1000000007  
	
# 	return tmp


# n = int(input())

# for i in range(n):
# 	dp = nxt(dp)

# print(dp[0])


# 1080.
# n,m = map(int,input().split()) #행,열 

# a = [list(map(int,input())) for _ in range(n)]
# b = [list(map(int,input())) for _ in range(n)]

# def change(x,y,a):
# 	for i in range(x,x+3):
# 		for j in range(y,y+3):
# 			a[i][j] = 1-a[i][j]
			
# count = 0
# for i in range(n-2):
# 	for j in range(m-2):
# 		if a[i][j] != b[i][j]:
# 			change(i,j,a)
# 			count += 1 

# print(count if a == b else -1)

# 2437.

# n = int(input()) #추의 개수
# wei = list(map(int,input().split()))
# wei.sort() 

# ans = 0

# for i in wei:
# 	if i <= ans + 1:
# 		ans += 1 
	
# 	else:
# 		break 
# print(ans + 1)

1181.
# 1.
# n = int(input())

# words = []
# for _ in range(n):
# 	word = input()
# 	words.append((word,len(word)))

# words = list(set(words))
# words.sort(key=lambda x: (x[1], x[0]))

# for w,n in words:
# 	print(w)

# 4153.

# import math 

# while True:
# 	a = list(map(int,input().split()))
# 	a.sort()
# 	if a[0] == 0:
# 		break 

# 	elif (a[0]**2 + a[1]**2) != a[2]**2:
# 		print('wrong')
	
# 	else:
# 		print('right')

# 1978.
# n = int(input()) #수의 개수

# a = list(map(int,input().split())) 

# def check(x):
# 	if x == 1:
# 		return False 
# 	for i in range(2,x):
# 		if x % i == 0:
# 			return False 
	
# 	return True 

# count = 0
# for i in a:
# 	if check(i):
# 		count += 1

# print(count)

# 2014.
# import heapq 
# import copy 

# k,n = map(int,input().split()) #k개의 소수, n번째항 찾기

# prime = list(map(int,input().split()))

# lst, ck = copy.deepcopy(prime), set()

# heapq.heapify(lst)
# ith = 0

# while ith < n:
# 	mn = heapq.heappop(lst)
# 	if mn in ck:
# 		continue 
# 	ith += 1
# 	ck.add(mn)
# 	for i in prime:
# 		heapq.heappush(lst,mn*i)

# print(mn)

# 2609.
# a,b = map(int,input().split())

# def gcd(x,y):
#     for i in range(min(x,y),0,-1):
#         if x % i == 0 and y % i == 0:
#             return i 

# def lcm(x,y):
#     return a//gcd(x,y)*b

# print(gcd(a,b))
# print(lcm(a,b))

# 1929.

# m,n = map(int,input().split())

# def prime(x):
#     if x <= 1:
#         return False 
    
#     for i in range(2,x):
#         if x % i == 0:
#             return False 
        
#         if i*i > x:
#             break 
#     return True 

# for i in range(m,n+1):
#     if prime(i):== 
#         print(i)

# 11050.
# n,k = map(int,input().split())

# def fac(y):
#     if y == 0:
#         return 1
#     for i in range(1,y):
#         y *= i 
#     return y 

# def bi(x,r):
#     n_fac = fac(x)
#     r_fac = fac(r)
#     n_r_fac = fac((x-r))
#     return n_fac // (r_fac * n_r_fac)
    
# print(bi(n,k))

# 11066.
# 1.
# def process():
#     n,a = int(input()), [0] + list(map(int,input().split()))
#     #s[i]는 1번부터 i번까지의 누적합
#     s = [0 for _ in range(n+1)] 
#     for i in range(1,n+1):
#         s[i] = s[i-1] + a[i]
#     print(s)
#     #dp[i][j]: i에서 j까지 합하는데 필요한 최소 비용
#     #dp[i][k] + dp[k+1][j] + sum(a[i]~a[j])
#     dp = [[0 for i in range(n+1)] for _ in range(n+1)]
#     for i in range(2,n+1): #부분파일의 길이
#         for j in range(1,n+2-i): #시작점 
#             dp[j][j+i-1] = min([dp[j][j+k] + dp[j+k+1][j+i-1] for k in range(i-1)]) + (s[j+i-1] - s[j-1])
    
#     for i in dp:
#         print(i)

#     print(dp[1][n])

# for _ in range(int(input())):
#     process()


# 2.
# tc = int(input())
# s,dp = 0,0 

# def f(i,j):
#     global s, dp 
#     if i == j:
#         return 0 
    
#     if dp[i][j] != -1:
#         return dp[i][j]

#     for k in range(i,j):
#         tmp = f(i,k) + f(k+1,j) + s[j] - s[i-1]
#         if dp[i][j] == -1 or dp[i][j] > tmp:
#             dp[i][j] = tmp 
#     return dp[i][j] 

# def process():
#     global s, dp
#     n, a = int(input()), list(map(int,input().split()))
#     s, dp = [0 for _ in range(n+1)], [[-1 for _ in range(n+1)] for i in range(n+1)]
#     for i in range(1, n+1):
#         s[i] = s[i-1] + a[i-1] 
#         print(f(1,n))

# for _ in range(tc):
#     process() 

# 10250.

# def check(hei,wei,num):
#     count = 0 
#     for i in range(1,wei+1):
#         for j in range(1,hei+1):
#             count+= 1

#             if count >= num:
#                 if len(str(i)) < 2:
#                     return str(j) + '0'+ str(i)
                
#                 return str(j)+str(i)

# for _ in range(int(input())):
#     h,w,n = map(int,input().split())
    
#     print(check(h,w,n))


# 1085.
# x,y,w,h = map(int,input().split())

# print(min(x,y,(w-x),(h-y)))

# 2164.
# from collections import deque 

# n = int(input())
# card = deque() 

# for i in range(1,n+1):
#     card.append(i)

# for i in range(n-1):
#     card.popleft()
#     card.append(card.popleft())

# print(card[0])

# 10816.
# import sys
# input = sys.stdin.readline

# n = int(input()) #숫자 카드 수 
# card = list(map(int,input().split())) #숫자 카드의 숫자

# m = int(input()) # 찾아야할 수 개수 
# num = list(map(int,input().split())) #찾아야할 수 

# num_d = {}

# for i in card:
#     if i not in num_d:
#         num_d[i] = 1 
#     else:
#         num_d[i] += 1 
# result = []
# for j in num:
#     if j in num_d:
#         result.append(str(num_d[j])+" ")
#     else:
#         result.append('0 ')

# for k in result:
#     print(k, end = "")

# 10828.
# import sys
# from collections import deque
# input = sys.stdin.readline
# stack = deque()

# for _ in range(int(input())):
#     a = list(input().split())
#     if len(a) == 2:
#         if a[0] == 'push':
#             stack.append(a[-1])
#     else:
#         if a[0] == 'pop':
#             if len(stack) == 0:
#                 print(-1)
            
#             else:
#                 stack.pop() 

#         elif a[0] == 'size':
#             print(len(stack))

#         elif a[0] == 'empty':
#             if len(stack) == 0:
#                 print(1)
#             else:
#                 print(0)

#         elif a[0] == 'top':
#             if len(stack) == 0:
#                 print(-1)
#             else:
#                 print(stack[-1])


# 10845.
# import sys
# from collections import deque

# input = sys.stdin.readline 
# queue = deque() 

# for _ in range(int(input())):
#     a = list(input().split())
#     if len(a) == 2:
#         if a[0] == 'push':
#             queue.append(a[-1])
    
#     else:
#         if a[0] == 'pop':
#             if len(queue) == 0:
#                 print(-1)
#             else:
#                 print(queue.popleft())

#         elif a[0] == 'size':
#             print(len(queue))

#         elif a[0] == 'empty':
#             if len(queue) == 0:
#                 print(1)
#             else:
#                 print(0)

#         elif a[0] == 'front':
#             if len(queue) == 0:
#                 print(-1)
#             else:
#                 print(queue[0]) 

#         elif a[0] == 'back':
#             if len(queue) == 0:
#                 print(-1)
#             else:
#                 print(queue[-1])

# 10866.
# import sys
# from collections import deque 

# input = sys.stdin.readline
# de = deque() 

# for _ in range(int(input())):
#     a = list(input().split())
#     if len(a) == 2:
#         if a[0] == 'push_front':
#             de.appendleft(a[-1])
        
#         elif a[0] == 'push_back':
#             de.append(a[-1])
    
#     else:
#         if a[0] == 'pop_front':
#             if len(de) == 0:
#                 print(-1)
#             else:
#                 print(de.popleft())
        
#         elif a[0] == 'pop_back':
#             if len(de) == 0:
#                 print(-1)
#             else:
#                 print(de.pop())

#         elif a[0] == 'size':
#             print(len(de))

#         elif a[0] == 'empty':
#             if len(de) == 0:
#                 print(1)
#             else:
#                 print(0)

#         elif a[0] == 'front':
#             if len(de) == 0:
#                 print(-1)
#             else:
#                 print(de[0])

#         elif a[0] == 'back':
#             if len(de) == 0:
#                 print(-1) 
#             else:
#                 print(de[-1])

# 9012.

# for _ in range(int(input())):
#     a = list(input())

#     for i in range(len(a)-1):
#         if a[i] == '(':
#             for j in range(i+1,len(a)):
#                 if a[j] == ')':
#                     a[i] = 0
#                     a[j] = 0
#                     break 
#     a = set(a)
    
#     if len(a) == 1 and list(a)[0] == 0:
#         print('YES')
    
#     else:
#         print('NO')

# 1018.
# #첫 글자를 정해주지 말고 w,b일때 모두 계산해서 최소 값을 구해야 한다.
# n,m = map(int,input().split()) #행, 열
# board = [list(input()) for _ in range(n)]
# min_num = 64

# def change(x,y):
#     count_b = 0
#     count_w = 0
#     for i in range(x,x+8):
#         for j in range(y,y+8):
#             if i % 2 == 1 and j % 2 == 1:
#                 if board[i][j] == 'W':
#                     count_b+= 1
                
#             elif i % 2 == 1 and j % 2 == 0:
#                 if board[i][j] == 'B':
#                     count_b += 1 
#             elif i % 2 == 0 and j % 2 == 1:
#                 if board[i][j] == 'B':
#                     count_b += 1 
#             elif i % 2 == 0 and j % 2 == 0:
#                 if board[i][j] == 'W':
#                     count_b += 1              

#     for i in range(x,x+8):
#         for j in range(y,y+8):
#             if i % 2 == 1 and j % 2 == 1:
#                 if board[i][j] == 'B':
#                     count_w+= 1
            
#             elif i % 2 == 1 and j % 2 == 0:
#                 if board[i][j] == 'W':
#                     count_w += 1 
#             elif i % 2 == 0 and j % 2 == 1:
#                 if board[i][j] == 'W':
#                     count_w += 1 
            
#             elif i % 2 == 0 and j % 2 == 0:
#                 if board[i][j] == 'B':
#                     count_w += 1              
#     return min(count_b,count_w)

# for x in range(n-7):
#     for y in range(m-7):
#         count = change(x,y)
#         min_num = min(count, min_num)
# print(min_num)

# 11866.

# n,k = map(int,input().split())

# board = [i for i in range(1,n+1)]

# idx = 0 #현재 index
# result = []

# while board:
#     idx = (idx+k-1) % len(board)

#     result.append(board[idx])
#     board.remove(board[idx])

# spring = '<'
# for i in range(len(result)):
#     if i == len(result)-1:
#         spring += str(result[i]) + '>'
#     else:
#         spring += str(result[i])+', '

# print(spring)

# 9095.
# import sys
# sys.setrecursionlimit(100000)

# def cal(x):
#     if x == 0:
#         return 0 
#     elif x == 1:
#         return 1
#     elif x == 2:
#         return 2 
#     elif x == 3:
#         return 4
    
#     return cal(x-3) + cal(x-2) + cal(x-1)

# for _ in range(int(input())):
#     n = int(input())
#     print(cal(n))    
    
# 1259.

# while True:
#     n = input() 
#     if n == '0':
#         break 
#     if n == n[::-1]:
#         print('yes')
#     else:
#         print('no')


# 1764.
# n,m = map(int,input().split()) #듣못사람, 보못사람 

# ear = set() 
# eye = set() 
# result = []
# for _ in range(n):
#     a = input() 
#     ear.add(a)

# for _ in range(m):
#     b = input() 
#     eye.add(b)

# for i in ear:
#     if i in eye:
#         result.append(i)
# result.sort()
# print(len(result))
# for i in result:
#     print(i)

# 1003.
# dp_0 = [1,0] 
# dp_1 = [0,1] 

# def fi(x):
#     if len(dp_0) < x+1:
#         for i in range(len(dp_0), x+1):
#             dp_0.append(dp_0[i-1] + dp_0[i-2])
#             dp_1.append(dp_1[i-1] + dp_1[i-2])
#     print(dp_0[x], dp_1[x])

# for _ in range(int(input())):
#     fi(int(input()))

    
# 1260.
# from collections import defaultdict

# def dfs(x):
#     result = []
#     visit = []
#     visit.append(x)

#     while visit:
#         num = visit.pop()
#         if num not in result:
#             result.append(num)
#             visit.extend(graph[num])
    
#     return result

# def bfs(x):
#     result = []
#     visit = []
#     visit.append(x)

#     while visit:
#         num = visit.pop(0)
#         if num not in result:
#             result.append(num)
#             visit.extend(graph[num])
#     return result


# n,m,v = map(int,input().split()) #정점의 개수, 간선의 개수, 탐색 시작하는 정점번호

# graph = defaultdict(list)

# for _ in range(m):
#     no1,no2 = map(int,input().split()) 
#     graph[no1].append(no2) 
#     graph[no2].append(no1) 

# #bfs
# for i in graph:
#     graph[i].sort()
# bfs_result = bfs(v)

# #dfs
# for j in graph:
#     graph[j].reverse()
# dfs_result = dfs(v)


# print(' '.join(map(str,dfs_result)))
# print(' '.join(map(str,bfs_result)))

# 2606.
# from collections import defaultdict 

# computer = defaultdict(list)

# def dfs(x):
#     result = []
#     visit = []
#     visit.append(x)
#     while visit:
#         num = visit.pop()
#         if num not in result:
#             result.append(num)
#             visit.extend(computer[num])
#     return len(result)-1

# c_num = int(input()) # 컴퓨터 수 
# for _ in range(int(input())):
#     co1,co2 = map(int,input().split())
#     computer[co1].append(co2)
#     computer[co2].append(co1)

# for i in computer:
#     computer[i].sort(reverse= True)


# print(dfs(1))

# 1436.

# def cal(x):
#     for i in range(2,x+1):
#         if i % 3 == 0:
#             dp[i] = min(dp[i//3], dp[i-1]) +1
        
#         elif i % 2 == 0:
#             dp[i] = min(dp[i//2],dp[i-1])+1

#         else:
#             dp[i] = dp[i-1] + 1
#     return dp[x]


# n = int(input())
# dp = [0 for _ in range(n+1)]
# print(cal(n))


# 17219.
# from collections import defaultdict 

# n,m = map(int,input().split()) # 사이트 주소 수, 찾으려는 사이트주소의 수
# memo = defaultdict(str)

# for _ in range(n):
#     site, password = map(str,input().split())
#     memo[site] = password 

# for _ in range(m):
#     f_site = input() 
#     if f_site in memo:
#         print(memo[f_site])


# 1436.

# n = int(input())
# init = 666

# while n:
#     if '666' in str(init):
#         n -= 1
#     init += 1 

# print(init -1)


# 1074.
# import sys
# sys.setrecursionlimit(100000)

# n,r,c = map(int,input().split())
# count = 0

# def z(m,a,b):
#     global count
#     if m < 3:
#         if r == a and c == b:
#             print(count)
#         count += 1

#         if r == a and c == b+1:
#             print(count)
#         count += 1 

#         if r == a+1 and c == b:
#             print(count)
#         count += 1

#         if r == a+1 and c == b+1:
#             print(count)
#         count += 1 
#         return 

#     z(m/2, a, b)
#     z(m/2, a, b+m/2)
#     z(m/2, a+m/2, b)
#     z(m/2, a+m/2, b+m/2)


# z(2**n,0,0)

# 1012.

# dx, dy = [0,1,0,-1],[1,0,-1,0]

# def dfs(x,y,board):
#     for i in range(4):
#         xx,yy = x+dx[i], y+dy[i]
#         if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#             continue 
        
#         if board[xx][yy] == 1:
#             board[xx][yy] = 0
#             dfs(xx,yy,board)
#     return board


# for _ in range(int(input())):
#     m,n,k = map(int,input().split()) #가로, 세로, 위치개수 
#     board = [[0 for _ in range(m)] for _ in range(n)]    
#     count = 0
#     for _ in range(k):
#         b,a = map(int,input().split())
#         board[a][b] = 1 

#     for i in range(n):
#         for j in range(m):
#             if board[i][j] == 1:
#                 dfs(i,j,board)
#                 count += 1
#     print(count)

# 1676.
# n = int(input())

# print(n//5 + n//25 + n//125)

# 11399.
# n = int(input()) #사람의 수

# time = list(map(int,input().split())) # 각 번호에서의 수
# dp = [0 for _ in range(n)]

# time.sort()

# for i in range(n):
#     dp[i] = dp[i-1] + time[i]
# print(sum(dp))


# 11724.
# 1.
# from collections import defaultdict 

# def dfs(x):
#     result = []
#     visit = []
#     visit.append(x)
    
#     while visit:
#         num = visit.pop() 
#         if num not in result and not ck[num]:
#             result.append(num)
#             ck[num] = True
#             visit.extend(board[num])  
#     return 

# n,m = map(int,input().split()) #정점개수, 간선의 개수
# board = defaultdict(list)
# ck = [False for _ in range(n+1)]
# count = 0

# for _ in range(m):
#     u, v = map(int,input().split())
#     board[u].append(v)
#     board[v].append(u)

# count = 0
# for i in range(1,n+1):
#     if ck[i] == False:
#         dfs(i)
#         count += 1 
# print(count)

# 2.
# def dfs(x):
#     ck[x] = True 

#     for i in board[x]:
#         if not ck[i]:
#             dfs(i)
    
        
# n,m = map(int,input().split()) 
# board = [[] for i in range(n+1)]
# ck = [False] * (n+1)
# count = 0 

# for i in range(m):
#     a,b = map(int,input().split())
#     board[a].append(b)
#     board[b].append(a)

# for j in range(1,n+1):
#     if not ck[j]:
#         count += 1 
#         dfs(j)
# print(count)

# 11279.
# import sys
# import heapq 
# input = sys.stdin.readline
# n = int(input()) #연산의 개수
# heap = []

# for _ in range(n):
#     num = int(input())
#     heapq.heappush(heap,-num)
    
#     if num == 0:
#         print(-1*heapq.heappop(heap))


# 2630.

# import sys
# sys.setrecursionlimit(10000)
# input = sys.stdin.readline

# def cal(x,a,b):
#     global b_cnt, w_cnt 

#     cnt = 0 

#     for i in range(a,a+x):
#         for j in range(b,b+x):
#             if board[i][j]:
#                 cnt += 1

#     if cnt == x**2:
#         b_cnt += 1

#     elif not cnt:
#         w_cnt += 1 
    
#     else:
#         cal(x//2, a,b)
#         cal(x//2, a+x//2,b)
#         cal(x//2, a,b+x//2)
#         cal(x//2, a+x//2, b+x//2)

#     return 

# n = int(input()) # 전체 종이의 한변의 길이 

# board = [list(map(int,input().split())) for _ in range(n)]
# b_cnt, w_cnt = 0, 0

# cal(n,0,0)
# print(w_cnt)
# print(b_cnt)

# 1620.
# import sys
# input = sys.stdin.readline

# n,m = map(int,input().split()) # 포켓몬 수 , 문제 수
# po_num = []
# board = dict()
# for i in range(n):
#     name = input().strip()
#     po_num.append(name)
#     board[name] = i+1 

# for j in range(m):
#     quiz = input().strip() 
#     if quiz.isdigit():
#         print(po_num[int(quiz)-1])
#     else:
#         print(board[quiz])

# 9375.
# #각 옷+1 을 모두 곱해준 다음 -1 하면 전체 수가 나온다.
# from collections import defaultdict

# def cal(x):
#     result = 1
#     for i in c:
#         result *= (len(c[i])+1)
#     return result - 1

# c = defaultdict(list) 

# for _ in range(int(input())):
#     c = defaultdict(list) 
#     n = int(input())
#     for _ in range(n):
#         dress, body = input().split()
#         c[body].append(dress)
#     print(cal(c))

# 11726.
# import sys
# input = sys.stdin.readline

# n = int(input()) 
# dp = [0] * (n+1)

# for i in range(1,n+1):
#     if i == 1:
#         dp[i] = 1
#         continue 
#     elif i == 2:
#         dp[i] = 2
#         continue 
#     dp[i] = dp[i-1] + dp[i-2]

# ans = dp[n] % 10007
# print(ans)

# 1931.
# n = int(input()) #회의 개수
# meet = []
# cnt = 0
# en = 0

# for _ in range(n):
#     start, end = map(int,input().split())
#     meet.append((start,end))

# meet.sort(key=lambda x: (x[1], x[0]))

# for s,e in meet:
#     if en <= s:
#         en = e 
#         cnt += 1

# print(cnt)

    
# 15829.
# print(ord('a'))# 문자 -> 숫자(a -> 97)
# print(chr(97)) # 숫자 -> 문자 
# print(pow(2,n))# 2의 n제곱승

# 1.
# l = int(input()) #문자열의 길이
# word = list(input())

# result = 0
# for i in range(l):
#     result += ((ord(word[i])-96) * (31**i))

# result %= 1234567891 
# print(result)

# 2579.
# n = int(input())
# dp = list()
# value = list()

# for _ in range(n):
#     value.append(int(input()))

# dp.append(value[0])
# dp.append(value[0] + value[1])
# dp.append(max(value[0]+value[2], value[1]+value[2]))

# for i in range(3,n):
#     dp.append(max(value[i]+value[i-1]+dp[i-3],value[i]+dp[i-2]))

# print(dp[1])

# 2292.

# def cal(x):
#     start = 2 
#     n = 6
#     j = 2
#     if x == 1:
#         return 1 
#     while True:
#         if x in range(start,start+n):
#             return j
#         start = (start+n) 
#         j += 1 
#         n += 6
# n = int(input())
# print(cal(n))

# 11727.

# def cal(x):
#     for i in range(x+1):
#         if i == 1:
#             dp[i] = 1
#             continue
#         elif i == 2:
#             dp[i] = 3
#             continue 
    
#         dp[i] = dp[i-1] + dp[i-2]*2
#     return dp[x]

# n = int(input())
# dp = [0]*(n+1)

# result = cal(n) % 10007
# print(result)

# 9465.

# for _ in range(int(input())):
#     n = int(input())
#     sticker = [list(map(int,input().split())) for _ in range(2)]
#     sticker[0][1] += sticker[1][0]
#     sticker[1][1] += sticker[0][0]

#     for y in range(2,n):
#         for x in range(2):
#             if x == 0:
#                 sticker[x][y] = max(sticker[x+1][y-1],sticker[x+1][y-2]) + sticker[x][y]
#             elif x == 1:
#                 sticker[x][y] = max(sticker[x-1][y-1],sticker[x-1][y-2]) + sticker[x][y]
#     print(max(sticker[0][n-1],sticker[1][n-1]))

# 1699. 
# import sys 
# input = sys.stdin.readline
# def cal(x):
#     for i in range(1,x+1):
#         dp[i] = i 
#         for j in range(1,i):
#             if i < (j*j):
#                 break 
#             dp[i] = min(dp[i],dp[i-j*j]+1)
#     return dp[x]
   
# n = int(input())
# dp = [0 for _ in range(n+1)]
# print(cal(n))

# 11723.
# from collections import deque
# import sys 
# input = sys.stdin.readline 

# def cal(x,y):
#     global s
#     if x == 'empty':
#         s.clear()
#     elif x == 'all':
#         s = deque(str(i) for i in range(1,21))
#     elif x == 'add':
#         if y not in s:
#             s.append(y)
#     elif x == 'remove':
#         if y in s:
#             s.remove(y)
#     elif x == 'check':
#         if y in s:
#             print(1)
#         else:
#             print(0)
#     elif x == 'toggle':
#         if y in s:
#             s.remove(y)
#         else:
#             s.append(y)

# m = int(input()) #연산의 수 
# s = deque()

# for _ in range(m):
#     w = input().split()
#     if w[0] == 'all' or w[0] == 'empty':
#         cal(w[0],0)
#     else:
#         cal(w[0],w[1])


# 7576.
# from collections import deque 
# import sys 
# input = sys.stdin.readline

# def bfs(M, N, box):
#     dx = [0, 0, 1, -1]
#     dy = [-1, 1, 0, 0]
#     days = -1
#     while ripe:
#         days += 1
#         for _ in range(len(ripe)):
#             x, y = ripe.popleft()

#             for i in range(4):
#                 nx = x + dx[i]
#                 ny = y + dy[i]

#                 if (0 <= nx < N) and (0 <= ny < M) and (box[nx][ny] == 0):
#                     box[nx][ny] = 1
#                     ripe.append([nx, ny])

#     for b in box:
#         if 0 in b:
#             return -1
#     return days


# M, N = map(int, input().split())
# box, ripe = [], deque()
# for i in range(N):
#     row = list(map(int, input().split()))
#     for j in range(M):
#         if row[j] == 1:
#             ripe.append([i, j])
#     box.append(row)

# print(bfs(M, N, box))

# 10953.
# t = int(input())
# for _ in range(t):
#     a,b = map(int,input().split(','))
#     print(a+b)


# 11021.
# t = int(input())

# for i in range(t):
#     a,b = map(int,input().split())
#     print("Case #%d:"%(i+1),a+b)

# 11022.
# for i in range(int(input())):
#     a,b = map(int,input().split())
#     print("Case #%d: %d + %d = %d" %((i+1),a,b,(a+b)))

# 2442.
# n = int(input()) 

# for i in range(1,n+1):
#     print(' '*(n-i)+ '*'*(2*i-1))


# 2445.
# n = int(input())
# for i in range(1,n+1):
#     print('*'*i +' '*(2*(n-i))+'*'*i)

# for j in range(n-1,0,-1):
#     print('*'*j+' '*(2*(n-j))+'*'*j)


# 2522.
# n = int(input())

# for i in range(1,n+1):
#     print(' '*(n-i)+'*'*i)

# for j in range(n-1,0,-1):
#     print(' '*(n-j)+'*'*j)


# 2446.
# n = int(input())
# for i in range(1,n+1):
#     print(' '*(i-1)+'*'*(2*(n-i)+1))
# for j in range(n-1,0,-1):
#     print(' '*(j-1)+'*'*(2*(n-j)+1))


# 10992.
# n = int(input())

# for i in range(1,n):
#     if i == 1:
#         print(' '*(n-1)+'*')
#     else:
#         print(' '*(n-i),'*',' '*(2*i-3),'*',sep='')
# print('*'*(2*n-1))


# 10991. 
# n = int(input())

# for i in range(1,n+1):
#     print(' '*(n-i),end='')
#     for j in range(1,i+1):
#         print('*'*(1),end=' ')
#     print()

# 11004.
# a,k = map(int,input().split())
# num = list(map(int,input().split()))
# num.sort()
# print(num[k-1])


# 10844.
#해당 자리수는 전 자리수에서 (+,-)1한 값이기 때문입니다.
#dp[n(자리수)][m(값)] = dp[n-1][m+1] - 0인경우
#dp[n(자리수)][m(값)] = dp[n-1][m-1] + dp[n-1][m+1] - 1~8인경우
#dp[n(자리수)][m(값)] = dp[n-1][m-1] - 9인경우

# n = int(input())
# dp = [[0 for _ in range(10)]for _ in range(101)]
# dp[1][0] = 0
# for i in range(1,10):
#     dp[1][i] = 1
# def cal(x):
#     for i in range(2,x+1):
#         for j in range(10):
#             if (j == 0):
#                 dp[i][j] = dp[i-1][j+1]
#             elif (j == 9):
#                 dp[i][j] = dp[i-1][j-1]
#             else:
#                 dp[i][j] = dp[i-1][j-1] + dp[i-1][j+1]
#     return sum(dp[x])

# print(cal(n)%1000000000)


# 11057.
#해당 자리수는 전 자리수보다 같거나 크면 됩니다.
#dp[n(자리수)][m(값)] = sum(dp[n-1]) - 0인경우
#dp[n(자리수)][m(값)] = dp[n][m-1] - dp[n-1][m-1] - 1~9인경우
#출력은 dp[n][0] 입니다.

# 1.
# n = int(input())
# dp = [[0 for _ in range(10)] for _ in range(n+1)]
# for i in range(10):
#     dp[1][i] = (10-i)

# def cal(x):
#     for i in range(2,x+1):
#         for j in range(10):
#             if (j == 0):
#                 dp[i][j] = sum(dp[i-1])
#             else:
#                 dp[i][j] = dp[i][j-1] - dp[i-1][j-1]
#     return dp[x][0]

# print(cal(n) % 10007)
# 2.
# n = int(input())
# dp = [[0]*10 for _ in range(1001)]
# for i in range(10):
#     dp[1][i] = 1 
# for i in range(2,1001):
#     for j in range(10):
#         for k in range(j+1):
#             dp[i][j] += dp[i-1][k]
# print(sum(dp[n])%10007)


# Program 에서 page를 만들어서 Copy_control_program로 page를 보내서 검사하여 Printer에서 복사하는 과정입니다.
# class Printer(object):
#     def __init__(self,check_page):
#         self.check_page = check_page
#     #Receiver
#     def print_copy(self):
#         print("{} 프린터 하였습니다.".format(self.check_page))

# class Copy_control_program(Printer): 
#     # Invoker 
#     def __init__(self, p_page): #page: 프린터 할 page
#         self._p_page = p_page

#     def Select_printer(self): #여러 프린터중 1개를 선택하는 과정
#         print("프린터를 선택했습니다.")

#     def Execute(self): #page가 1개인지 check하는 과정
#         if (self._p_page == 1):
#             self.Select_printer()
#             Printer(self._p_page).print_copy()
#         else:
#             print("여러 page가 들어갔습니다.")
    
# class Program(object):
#     #Client 
#     def __init__(self,page_cmd):
#         self.page_cmd = page_cmd
#         program = Copy_control_program(self.page_cmd)
#         program.Execute()
        
# if __name__ == "__main__":
#     print("<여러개 들어갔을 때>")
#     program_page = Program(2)
#     print("<1개 들어갔을 때>")
#     program_page2 = Program(1) 
    

# 2193.
# n = int(input())
# dp = [0]*(n+1)
# dp[0] = 0
# dp[1] = 1
# for i in range(2, n+1):
#     dp[i] = dp[i-1] + dp[i-2]
# print(dp[n])


# 2156.
# n = int(input())
# w = [0]
# for i in range(n):
#     w.append(int(input()))

# dp = list(0 for i in range(n+1))
# dp[1] = w[1]

# for i in range(2,n+1):
#     if i == 2:
#         dp[i] = w[1] + w[2]
#     else:
#         dp[i] = max(dp[i-1], dp[i-2]+w[i], dp[i-3]+w[i-1]+w[i])
# print(dp[n])


# 11722
# n = int(input())
# arr = list(map(int,input().split()))
# dp=[1]*n

# for i in range(1,n):
#     for j in range(i):
#         if arr[i] < arr[j]:
#             dp[i] = max(dp[i],dp[j]+1)

# print(max(dp))
    

# 1912
# import copy 
# n = int(input())
# arr = list(map(int,input().split()))

# dp = copy.deepcopy(arr) 

# for i in range(1, n):
#     dp[i] = max(arr[i-1]+arr[i], dp[i-1]+arr[i], arr[i])

# print(max(dp))

# 2133
# n = int(input())
# dp = [0] * (31)
# dp[2] = 3 

# for i in range(4, 31, 2):
#     dp[i] = 3*dp[i-2]
#     for j in range(4,i,2):
#         dp[i] += dp[i-j]*2
#     dp[i]+=2 
# print(dp[n])

# 11651
# n = int(input())
# location = list()

# for i in range(n):
#     x,y = map(int,input().split())
#     location.append((x,y))

# s_location = sorted(location , key= lambda x: (x[1],x[0]))

# for i in s_location:
#     print(i[0], i[1])

# 9461
# dp = list(0 for i in range(1000001))
# dp[1] = 1
# dp[2] = 1

# for i in range(int(input())):
#     n = int(input())
#     for j in range(3, n+1):
#         dp[j] = dp[j-2] + dp[j-3]
#     print(dp[n])

# 10825
# dp = list()
# for i in range(int(input())):
#     name, korea, english, math = list(input().split())
#     dp.append((name,int(korea),int(english),int(math)))

# s_dp = sorted(dp, key = lambda x: (-x[1], x[2], -x[3], x[0]))

# for i in s_dp:
#     print(i[0])

# 1158
# n , k = map(int,input().split())
# arr = list(i for i in range(1,n+1))
# result = []

# i = k-1
# while True:
#     result.append(arr.pop(i))
#     if not arr:
#         break
#     i = (i+k-1) % len(arr)

# print('<'+', '.join(map(str,result))+'>')
    

# 1168
# n,k = map(int,input().split())
# arr = list(i for i in range(1, n+1))
# result = []
# i = k-1

# while True:
#     result.append(arr.pop(i))
#     if not arr:
#         break 
#     i = (i+k-1)%len(arr)

# print('<' + ', '.join(map(str, result))+ '>')


# 1934
# def gcd(x,y):
#     for i in range(min(x,y),0,-1):
#         if x % i == 0 and y % i == 0:
#             return i

# def lcm(x,y):
#     return x//gcd(x,y)*y

# for i in range(int(input())):
#     a,b = map(int,input().split())
#     print(lcm(a,b))
    
# 1373
# b = int(input(),2)
# print(oct(b)[2:])


# 2745
# a = int(input(),8)
# print(bin(a)[2:])

# 11653
# n = int(input())
# i = 2 
# while n!= 1:
#     if n % i == 0:
#         n/=i 
#         print(i)
#     else:
#         i+=1


# 2743
# n = list(input())
# print(len(n))


# 10451
# import sys 
# sys.setrecursionlimit(10000)

# def dfs(x):
# 	visit_bool[x] = True 
# 	next_visit = cycle_array[x]
# 	if visit_bool[next_visit] == False:
# 		dfs(next_visit)


# for i in range(int(input())):
# 	count = 0
# 	cycle_num = int(input())
# 	cycle_array = [0] + list(map(int,input().split()))
# 	visit_bool = [True]+ [False]*cycle_num
	
# 	for i in range(1, cycle_num+1):
# 		if visit_bool[i] == False:
# 			dfs(i)
# 			count+=1 
# 	print(count)


# 2331
# A, p = map(int,input().split())
# idx = 0
# array = []
# array.append(A)
# num = A

# while True:
# 	next_num = 0

# 	for i in list(str(num)):
# 		next_num += int(i)**p

# 	if next_num not in array:
# 		array.append(next_num) 
# 		num = next_num 
# 	else:
# 		idx = array.index(next_num)
# 		print(idx)
# 		break

# 10808
# alpha_array = [0 for i in range(26)]
# alpha = list(input())

# for i in alpha:
# 	alpha_idx = ord(i)-97 
# 	alpha_array[alpha_idx]+=1

# print(' '.join(map(str,alpha_array)))

# 2875
# import sys 
# input = sys.stdin.readline
# girl, boy, intership = map(int,input().split()) 

# team_num = 0

# while True:
# 	if girl > 1 and boy > 0 and girl+boy > intership+2:
# 		girl-=2
# 		boy-=1
# 		team_num+=1 
# 	else:
# 		break
# print(team_num)	

# 11052
# n = int(input())
# price = [0]+ list(map(int,input().split()))
# dp = [0] *(n+1)
# dp[1] = price[1]
# for i in range(2,n+1):
#     for j in range(1,i+1):
#         dp[i] = max(dp[i], dp[i-j]+price[j])

# print(dp[n])


# 2667

# n = int(input())
# house_map = [list(map(int,input())) for _ in range(n)]
# check_visit = [[False]*n for _ in range(n)]
# dx,dy = [-1,0,1,0], [0,-1,0,1] 
# nums = []
# num = 0
# count = 0

# def dfs(a,b):
#     global num 
#     check_visit[a][b] = True 
#     num+=1
#     for i in range(4):
#         aa, bb = a+dx[i], b+dy[i]
        
#         if aa > n-1 or aa < 0 or bb < 0 or bb > n-1:
#             continue 
            
#         if house_map[a][b] == house_map[aa][bb] and check_visit[aa][bb] == False:
#             dfs(aa,bb)
#     return num 

# for y in range(n):
#     for x in range(n):
#         if house_map[x][y] != 0 and check_visit[x][y] == False:
#             nums.append(dfs(x,y))
#             num = 0
#             count+= 1

# nums.sort()
# print(count)
# for i in nums:
#     print(i)

# 10820

# import sys 
# input = sys.stdin.readline
# while True:
#     line = input().strip('\n')
#     lo, up, di, sp = 0,0,0,0 

#     if not line:
#         break 
    
#     for i in line:
#         if i.islower():
#             lo+=1
#         elif i.isupper():
#             up+=1
#         elif i.isdigit():
#             di+=1
#         elif i.isspace():
#             sp+=1
    
#     print("{} {} {} {}".format(lo,up,di,sp))


# 10824
# a,b,c,d = input().split()
# ab = a+b 
# cd = c+d 
# print(int(ab)+int(cd))


# 2004
# n, m = map(int,input().split())

# def cal(x, k):
#     count = 0
#     while x!=0:
#         x=x//k
#         count+=x 
#     return count 

# two_cal = cal(n,2) - cal(n-m,2) - cal(m,2)
# five_cal = cal(n,5) - cal(n-m,5) - cal(m,5)
# print(min(two_cal,five_cal))

# 2225
# n, k = map(int,input().split())

# dp = [[1]*201 for _ in range(201)]

# for n_x in range(1,201):
#     for k_y in range(2,201):
#         dp[n_x][k_y] = dp[n_x-1][k_y] + dp[n_x][k_y-1]
# print(dp[n][k]%1000000000)

# 4963
# dx,dy = [1,-1,0,0,1,-1,1,-1], [0,0,-1,1,-1,-1,1,1]

# def bfs(a,b):
#     island[a][b] = 0
#     queue = [[a,b]]
#     while queue:
#         x,y = queue.pop(0)
        
#         for i in range(8):
#             xx,yy = x+dx[i], y+dy[i]

#             if xx < 0 or xx > h-1 or yy < 0 or yy > w-1:
#                 continue 

#             if island[xx][yy] == 1:
#                 island[xx][yy] = 0
#                 queue.append([xx,yy])

# while True:
#     w, h = map(int,input().split())
#     island = [list(map(int,input().split())) for _ in range(h)]
#     count = 0

#     if w == 0 and h == 0:
#         break 

#     for i in range(h):
#         for j in range(w):
#             if island[i][j] == 1:
#                 bfs(i,j)
#                 count+=1
#     print(count)


# 11652
# card_num = {}
# for _ in range(int(input())):
#     card = int(input())
#     if card not in card_num:
#         card_num[card] = 0
#     else:
#         card_num[card]+=1

# card_num = sorted(card_num.items(), key= lambda x:(-x[1],x[0]))
# print(card_num[0][0])


# 9613
# def gcd(x,y):
#     for i in range(min(x,y),0,-1):
#         if x % i == 0 and y % i == 0:
#             return i

# for _ in range(int(input())):
#     nums = list(map(int,input().split()))
#     gcd_sum = 0

#     for i in range(1,len(nums)-1):
#         for j in range(i+1,len(nums)):
#             gcd_sum+=gcd(nums[i], nums[j])
#     print(gcd_sum)


# 2011
# #첫자리 dp[i]+= dp[i-1] 1 ~ 9
# #두자리 dp[i]+= dp[i-2] 10 ~ 26

# num = input()
# nums =[int(i) for i in list(num)]
# dp = [0]*(len(nums)+1)

# dp[0] = 1

# for i in range(1, len(nums)+1):  
#     if 0 < nums[i-1] < 10:
#         dp[i]+= dp[i-1]
    
#     if i == 1:
#         continue 

#     tmp = (nums[i-2]*10 + nums[i-1])

#     if 9 < tmp < 27 :
#         dp[i]+= dp[i-2]

# print(dp[-1]%1000000)

# 2178
# #bfs 사용 
# dx,dy = [-1,0,1,0], [0,1,0,-1]

# def bfs(x,y):
#     arr[0][0] = 0
#     queue = [(0,0)]
#     dist[0][0] = 1

#     while queue:
#         px, py = queue.pop(0)
        
#         if px == (x-1) and py == (y-1):
#             break 

#         for i in range(4):
#             xx,yy = px+dx[i], py+dy[i]
            
#             if xx < 0 or xx > x-1 or yy < 0 or yy > y-1:
#                 continue 
            
#             if arr[xx][yy] == 1:
#                 arr[xx][yy] = 0
#                 dist[xx][yy] = dist[px][py] + 1
#                 queue.append((xx,yy))

# n, m = map(int,input().split()) 
# arr = [list(map(int,input())) for _ in range(n)]
# dist = [[0]*m for _ in range(n)]
# bfs(n,m)
# print(dist[n-1][m-1])


# 1406
# import sys 
# input = sys.stdin.readline

# left_string = list(input()[:-1])
# right_string = []

# for _ in range(int(input())):
#     command = input().split()
#     if command[0] == 'L':
#         if left_string:
#             right_string.append(left_string.pop())
#         else:
#             continue 
#     elif command[0] == 'D':
#         if right_string:
#             left_string.append(right_string.pop())
#         else:
#             continue

#     elif command[0] == 'B':
#         if left_string:
#             left_string.pop()
#         else:
#             continue 

#     elif command[0] == 'P':
#         left_string.append(command[1])

# print(''.join(left_string + right_string[::-1]))

# 1541
# expression = input().split('-')
# nums = []

# for i in expression:
#     plus_num = 0
#     plus_split_num = i.split('+')
#     for num in plus_split_num:
#         plus_num += int(num) 

#     nums.append(plus_num)

# result = nums[0]
# for k in range(1,len(nums)):
#     result-=nums[k]
# print(result)


11725
#dfs, bfs 둘다 사용가능
# 1.
# import sys 
# sys.setrecursionlimit(100000)
# input = sys.stdin.readline 

# def dfs(x):
#     for i in graph[x]:
#         if parent[i] == 0:
#             parent[i] = x
#             dfs(i)

# node = int(input())
# graph = [[] for _ in range(node+1)]
# parent = [0 for _ in range(node+1)]

# for _ in range(node-1):
#     node1, node2 = map(int,input().split())
#     graph[node1].append(node2)
#     graph[node2].append(node1)

# dfs(1)

# for i in range(2,node+1):
#     print(parent[i])


# 10610
#30으로 나눠줘야 하는 것이므로 
#맨 뒤의 숫자가 0이여야 하며 총 수의 합이 3으로 나눠져야 한다.

# n = list(input())

# n.sort(reverse = True)
# sum_num = 0
# for i in n:
#     sum_num+= int(i) 
# if sum_num % 3 != 0 or '0' not in n:
#     print(-1)
# else:
#     print(''.join(n))


# 2089
# 값을 -2로 나누었을 때 나머지가 1이면 이진수에 1
# 값을 -2로 나누었을 때 나머지가 0이면 이진수에 0
# n = // (-2) + 1 인 이유는 소수가 나오는 것을 방지 하기 위해서 입니다.

# n = int(input())
# binary = []

# if n == 0:
#     print(0)
# else:
#     while n:
#         if n % (-2):
#             binary = [1] + binary 
#             n = n//(-2) + 1
#         else:
#             binary = [0] + binary 
#             n = n//(-2)

# for i in binary:
#     print(i, end='')

# 11656
# word = input() 
# array = []
# for i in range(len(word)):
#     array.append(word[i:])
# array.sort() 

# for i in array:
#     print(i)

# 11655.
# sentence = input()

# rot13 = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 
# 'g':'t', 'h':'u', 'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 
# 'n':'a', 'o':'b', 'p':'c', 'q':'d', 'r':'e', 's':'f', 't':'g', 
# 'u':'h', 'v':'i', 'w':'j', 'x':'k', 'y':'l', 'z':'m', 'A':'N', 
# 'B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S', 'G':'T', 'H':'U', 
# 'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A', 'O':'B', 
# 'P':'C', 'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I', 
# 'W':'J', 'X':'K', 'Y':'L', 'Z':'M', ' ':' ', '0':'0', '1':'1', 
# '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', 
# '9':'9'}

# for word in sentence:
#     print(rot13[word], end='')



# 1992
# n = int(input())
# tree = [list(map(int,input())) for _ in range(n)]
# array = []

# def quad_tree(x,a,b):
#     check = set()

#     if x < 1:
#         return 

#     for i in range(a,x+a):
#         for j in range(b,x+b):
#             check.add(tree[i][j])
    
#     if len(check) != 1: 
#         array.append('(')
#         quad_tree(x//2,a,b)
#         quad_tree(x//2,a,b+x//2)
#         quad_tree(x//2,a+x//2,b)
#         quad_tree(x//2,a+x//2,b+x//2)
#         array.append(')')

#     else:
#         array.append(tree[a][b])

# quad_tree(n,0,0)
# for i in array:
#     print(i, end= '')


# 1850
# import sys 
# input = sys.stdin.readline 

# def gcd(x,y):
#     while True:
#         if x % y:
#             temp = y 
#             y = x % y
#             x = temp 
#         else:
#             return y


# a, b = map(int,input().split())

# num = gcd(max(a,b),min(a,b))%10000000

# for i in range(num):
#     print(1,end='')


# 11005
# def convert(n,b):
#     T = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     quotient = n // b  #몫
#     remainder = n % b  #나머지
#     if quotient == 0:
#         return T[remainder]
#     else:
#         return convert(quotient, b) + T[remainder]


# n, b = map(int,input().split())
# print(convert(n,b))


# 11576
# def convert(n,b):
#     T = []
#     for i in range(b):
#        T.append(i)
#     T = list(map(str,T))
#     q = n // b 
#     r = n % b 

#     if q == 0:
#         return T[r] 
#     else:
#         return convert(q,b) + ' ' +  T[r]

# a_base, b_base = map(int,input().split())
# digits = int(input())
# each_digits_num = list(map(int,input().split()))
# each_digits_num.reverse()

# base_10 = 0
# for i in range(len(each_digits_num)):
#     base_10 += ((a_base**i)*each_digits_num[i]) 

# print(convert(base_10,b_base))


# 10799
# bracket = list(input())
# iron_bar = 0
# stack = []
# length_bracket = len(bracket)

# for i in range(length_bracket):
#     if bracket[i] == '(':
#         stack.append('(')
    
#     else:
#         if bracket[i-1] == '(':
#             stack.pop() 
#             iron_bar += len(stack)
#         else:
#             stack.pop()
#             iron_bar+=1 
# print(iron_bar)

# 6588
# def primary_check(n):
#     num = [True]*(n+1)

#     for i in range(2,len(num)//2+1):
#         if num[i] == True:
#             for j in range(i+i, n, i):
#                 num[j] = False
#     return [[i for i in range(2,n) if num[i] == True], num]

# num_primary = primary_check(1000000)[0]
# bool_primary = primary_check(1000000)[1]

# while True:
#     n = int(input())

#     if n == 0:
#         break 

#     for i in range(n//2):
#         if bool_primary[n-num_primary[i]] == True:
#             print("{} = {} + {}".format(n,num_primary[i],n-num_primary[i]))
#             break


# 11054
# 홀수 짝수 이렇게 생각을 안해도 된다. 
# 증가 부분이랑 감소부분을 따로 계산해서 더해준다.
# 증가 부분: 각 자리 숫자 와 현재 자리 전 까지 값을 비교하여 증가하면 +1해준다(max값 선택)
# 감소 부분: arr를 거꾸로 하여 각 증가 부분과 같게 해야 된다.
# => 자신의 값을 과거의 값들을 돌아봐서 그 중 최대의 값으로 자신의 최대 길이를 정하는 문제 

# n = int(input())
# arr = list(map(int,input().split()))

# increase = [1]*n
# decrease = [1]*n 
# plus_increase_decrease = [0]*n 

# for i in range(n):
#     for j in range(i):
#         if arr[i] > arr[j]:
#             increase[i] = max(increase[i], increase[j]+1)

# arr.reverse() 

# for i in range(n):
#     for j in range(i):
#         if arr[i] > arr[j]:
#             decrease[i] = max(decrease[i],decrease[j]+1)

# decrease.reverse()

# for i in range(n):
#     plus_increase_decrease[i] = increase[i] + decrease[i]

# print(max(plus_increase_decrease)-1)


# 11728
# a_size, b_size = map(int,input().split())
# a_arr = list(map(int,input().split()))
# b_arr = list(map(int,input().split()))
# total_arr = a_arr + b_arr
# total_arr.sort() 
# for i in total_arr:
#     print(i, end=' ')


# 10773
# account_book = []
# total = 0

# for i in range(int(input())):
#     num = int(input())
#     if num != 0:
#         account_book.append(num)
#     else:
#         account_book.pop(-1)

# for i in account_book:
#     total+=i
# print(total)


# 11729
# # 다시 보기
# # 첫 번째 규칙 1->2, 1->3, 2->3을 잘 생각해서 재귀 하면 된다.
# # 재귀할 때 해당 code 다 끝나면 돌아온다.
# def hanoi(n, from_, to_, by_):
#     if n == 1:
#         print(from_, by_)
#     else:
#         hanoi(n-1, from_, by_, to_)
#         print(from_, by_)
#         hanoi(n-1, to_, from_, by_)

# n = int(input())

# total = 2**n - 1 
# print(total)
# hanoi(n,1,2,3)

# 1783
# # 위의 4 조건 모두 만족하지 않으면 이동이 불가능
# # 그러나 이동횟수가 4번보다 적은 경우에는 이동방법에 제약이 없다(4가 최대)
# n, m = map(int,input().split())

# if n == 1 or m == 1:
#     print(1)
# elif n == 2:
#     print(min(4, (m+1)//2)) 
#     #(m-1) 2,3 방법만 사용가능
# else:
#     if m < 7: 
#         print(min(4,m))
#         # 1,4만 사용가능
#     else:
#         print(m-2)


# 10815
# sang_n = int(input())
# sang_card = list(map(int,input().split()))
# check_n = int(input())
# check_card = list(map(int,input().split()))

# sang_cards = dict() 

# for i in sang_card:
#     sang_cards[i] = 1

# for i in check_card:
#     if i not in sang_cards:
#         print(0, end= ' ')
#     else:
#         print(1, end = ' ')


# 2805
# # 높이의 최대값 구하기 
# tree_num, need_tree_length = map(int,input().split())
# tree = list(map(int,input().split()))
# start, end = 0, max(tree)

# while start <= end:
#     mid = (start+end) // 2 
#     length = 0

#     for i in tree:
#         if i > mid:
#             length += (i-mid)
    
#     if length >= need_tree_length:
#         start = mid + 1
#     else:
#         end = mid - 1
# print(end)


# 1654
# import sys 
# input = sys.stdin.readline
# k, n = map(int,input().split())
# line = [int(input()) for _ in range(k)]
# start, end = 1, max(line)

# while start <= end:
#     mid = (start+end) // 2 
#     num = 0 

#     for i in line:
#         num += (i // mid)

#     if num >= n:
#         start = mid + 1
#     else:
#         end = mid - 1 

# print(end)


# 2407
# #조합 문제 
# def factorial(x):
#     num = 1
#     for i in range(2,x+1):
#         num*=i

#     return num

# n, m = map(int,input().split())
# n_fac = factorial(n)
# m_fac = factorial(m)
# minus_fac = factorial(n-m)

# print(n_fac//(m_fac*minus_fac))


# 15649
# backtracking(전체 다 검사하는데 조건 안맞으면 가지친다)
# 트리 구조 기반으로 DFS탐색을 진행하며 각 루트가 조건에 
# 맞는지 check -> 조건에 맞지 않으면 DFS탐색 안하고 가지친다(pruning)

# n,m = map(int,input().split())
# check = [0 for _ in range(n+1)]
# result = [0 for _ in range(m)]

# def dfs(index, n ,m):
#     if index == m:
#     # 배치가 된 상태
#         for i in range(m):
#             print(result[i], end=' ')
#         print() 
#         return 
    
#     for i in range(1,n+1):
#     # n개 값을 앞에서 부터 인자를 check한다
#         if check[i] == 1:
#             continue 
#         result[index] = i 
#         check[i] = 1 
#         dfs(index+1,n,m)
#         check[i] = 0 

# dfs(0,n,m)



# 15650

# n,m = map(int,input().split())

# check = [0 for _ in range(n+1)]
# num = [0 for _ in range(m)]

# def dfs(index, n, m):
#     if index == m:
#         for i in range(m):
#             print(num[i], end=' ')
#         print()
#         return 

#     for i in range(1,n+1):
#         if check[i] == 0:
#             check[i] = 1 
#             num[index] = i
#             dfs(index+1,n,m)
#             for j in range(i+1,n+1):
#                 check[j] = 0

# dfs(0,n,m)


# 15651

# n,m = map(int,input().split())
# num = [0 for _ in range(m)]

# def dfs(index, n, m):
#     if index == m:
#         for i in range(m):
#             print(num[i], end = ' ')
#         print()
#         return 

#     for i in range(1,n+1):
#         num[index] = i 
#         dfs(index+1,n,m)

# dfs(0,n,m)


# 15652
# import sys 
# sys.setrecursionlimit(10000)
# n,m = map(int,input().split())

# num = [0 for _ in range(m)]


# def dfs(index,n,m,idx):
#     if index == m:
#         for i in range(m):
#             print(num[i],end = ' ')
#         print() 
#         return 

#     for j in range(idx,n+1):
#         num[index] = j
#         dfs(index+1,n,m,j)
        
# dfs(0,n,m,1)


# 15654
# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# check = [0 for _ in range(n)]
# num = [0 for _ in range(m)]

# use_nums.sort() 

# def dfs(index,n,m):
#     if index == m:
#         for i in range(m):
#             print(num[i], end=' ')
#         print() 
#         return 

#     for j in range(n):
#         if check[j] == 0:
#             check[j] = 1 
#             num[index] = use_nums[j]
#             dfs(index+1,n,m)
#             check[j] = 0

# dfs(0,n,m)


# 15655
# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# check = [0 for _ in range(n+1)]
# num = [0 for _ in range(m)]

# use_nums.sort() 

# def dfs(index,n,m):
#     if index == m:
#         for i in range(m):
#             print(num[i],end=' ')
#         print()
#         return 

#     for j in range(n):
#         if check[j] == 0:
#             check[j] = 1
#             num[index] = use_nums[j]
#             dfs(index+1,n,m)
#             for k in range(j+1,n):
#                 check[k] = 0

# dfs(0,n,m)



# 15656
# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# num = [0 for _ in range(m)]

# use_nums.sort()

# def dfs(index,n,m):
#     if index == m:
#         for i in range(m):
#             print(num[i], end = ' ')
#         print() 
#         return 

#     for j in range(n):
#         num[index] = use_nums[j]
#         dfs(index+1,n,m)

# dfs(0,n,m)


# 15657
# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# num = [0 for _ in range(m)]

# use_nums.sort() 

# def dfs(index,n,m,idx):
#     if index == m:
#         for i in range(m):
#             print(num[i], end = ' ')
#         print() 
#         return 

#     for j in range(idx,n):
#         num[index] = use_nums[j]
#         dfs(index+1,n,m,j)

# dfs(0,n,m,0)

# 15663
# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# check = [0 for _ in range(len(use_nums))]
# num = [0 for _ in range(m)]
# result = []

# use_nums.sort()


# def dfs(index,n,m):
#     nums = []
#     if index == m:
#         for i in range(m):
#             nums.append(num[i])
#             tmp = ' '.join(map(str,nums))
#         if tmp not in result:
#             result.append(tmp)
#         return 
    
#     for j in range(len(use_nums)):
#         if check[j] == 0:
#             check[j] = 1 
#             num[index] = use_nums[j]
#             dfs(index+1,n,m)
#             check[j] = 0 

# dfs(0,n,m)
# for i in result:
#     print(i)


# 15664 
# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# check = [0 for _ in range(n)]
# num = [0 for _ in range(m)]
# result = [] 

# use_nums.sort()

# def dfs(index,idx,n,m):
#     nums = []
#     if index == m:
#         for i in range(m):
#             nums.append(num[i])
#         tmp = ' '.join(map(str,nums))
#         if tmp not in result:
#             result.append(tmp)
#         return


#     for j in range(idx, n):
#         if check[j] == 0:
#             check[j] = 1
#             num[index] = use_nums[j]
#             dfs(index+1,j+1,n,m)
#             check[j] = 0

# dfs(0,0,n,m)
# for i in result:
#     print(i)

# 15665

# n, m = map(int,input().split())
# use_nums = list(map(int,input().split()))
# nums = [0 for _ in range(m)]
# result = []

# use_nums.sort()

# def dfs(index,n,m):
#     if index == m:
#         for i in range(m):
#             print(nums[i], end = ' ')
#         print()
#         return 

#     same_num = 0
#     for j in range(len(use_nums)):
#         if same_num != use_nums[j]:
#             nums[index] = use_nums[j]
#             same_num = use_nums[j]
#             dfs(index+1,n,m)

# dfs(0,n,m)
# for i in result:
#     print(i)


# 1149
# n = int(input())

# dp = []

# for i in range(n):
#     dp.append(list(map(int,input().split())))

# for i in range(1, n):
#     dp[i][0] = min(dp[i-1][1]+ dp[i][0], dp[i-1][2]+ dp[i][0]) 
#     dp[i][1] = min(dp[i-1][0]+ dp[i][1], dp[i-1][2]+ dp[i][1]) 
#     dp[i][2] = min(dp[i-1][0]+ dp[i][2], dp[i-1][1]+ dp[i][2]) 

# print(min(dp[n-1][0],dp[n-1][1],dp[n-1][2]))


# 15666
# n,m = map(int,input().split())
# use_nums = list(set(map(int,input().split())))
# num = [0 for _ in range(m)]

# use_nums.sort()

# def dfs(index,idx,n,m):
#     if index == m:
#         for i in range(m):
#             print(num[i], end=' ')
#         print()
#         return 

#     for j in range(idx,len(use_nums)):
#         num[index] = use_nums[j]
#         dfs(index+1,j,n,m)

# dfs(0,0,n,m)


# 1629
# # 나머지를 구할 때 %를 계속 사용하면 속도를 줄일 수 있다. 
# # 각 %를 해준 뒤 마지막에 %를 해줘야 원하는 결과를 얻을 수 있다.
# # => 각 값의 나머지를 곱한 수 %한 값 이랑 원래 수 %한 값이랑 같다
# a,b,c = map(int,input().split()) 

# def sol(a,b,c):
#     if b == 1:
#         return a % c  
#     else:
#         tmp = sol(a,b//2,c)
#         if b % 2 == 1:
#             return tmp * tmp * a % c 
#         else:
#             return tmp * tmp % c 

# print(sol(a,b,c))


# 5639
# node - x , left_child - 2x, right_child - 2x+1 
# 같은 class 내에서 활동하는 것에는 self. 해줘야 한다.
# 1.
# import sys 
# sys.setrecursionlimit(10**8)

# class Node(object):
#     def __init__(self,val):
#         self.val = val 
#         self.left = None
#         self.right = None 

# class binaryTree:
#     def __init__(self):
#         self.head = Node(None)

#     def insert(self,val):
#         if self.head == None:
#             self.head = Node(val)
#         else:
#             cur = self.head 
#             while True:
#                 if cur.val > val:
#                     if cur.left == None:
#                         cur.left = Node(val)
#                         break 
#                     cur = cur.left 
#                 if cur.val < val:
#                     if cur.right == None:
#                         cur.right = Node(val)
#                         break 
#                     cur = cur.right 

#     def post_order(self,cur_node):
#         if cur_node.left != None:
#             self.post_order(cur_node.left)
#         if cur_node.right != None:
#             self.post_order(cur_node.right)
#         print(cur_node.val)


# if __name__=="__main__":

#     b_tree = binaryTree() 
#     while True:
#         try:
#             num = int(input())
#             print(num)
#             b_tree.insert(num)
#         except: 
#             break  
#     b_tree.post_order(b_tree.head)


# 2.
# 전위 순회에서 맨 처음 값이 root이며 이를 기준으로 해당
# 값보다 크면 right_child 이고 작으면 left_child이다
# import sys
# sys.setrecursionlimit(10000)
# input = sys.stdin.readline

# def post_order(start,end):
#     if start > end:
#         return

#     div = end+1
#     #오른쪽 구분되는 부분을 찾아주는 곳
#     for i in range(start+1, end+1):
#         if nums[start] < nums[i]:
#             div = i 
#             break

#     post_order(start+1,div-1) #왼쪽
#     post_order(div,end) #오른쪽
#     print(nums[start])
    

# nums=[]
# while True:
#     try:
#         num = int(input())
#         nums.append(num)
#     except:
#         break

# post_order(0,len(nums)-1)

# 1789
# 1.
# import sys 
# sys.setrecursionlimit(100000)

# n = int(input())
# board = [list(map(int,input().split())) for _ in range(n)]
# minus_count = 0
# zero_count = 0
# plus_count = 0


# def cal(n,a,b):
#     global minus_count, zero_count, plus_count
    
#     num = set()

#     for x in range(a,n+a):
#         for y in range(b,n+b):
#             num.add(board[x][y])

#     if len(num) != 1:
#         cal(n//3,a,b)
#         cal(n//3,a,b+(n//3))
#         cal(n//3,a,b+2*(n//3))
#         cal(n//3,a+(n//3),b)
#         cal(n//3,a+(n//3),b+(n//3))
#         cal(n//3,a+(n//3),b+2*(n//3))
#         cal(n//3,a+2*(n//3),b)
#         cal(n//3,a+2*(n//3),b+(n//3))
#         cal(n//3,a+2*(n//3),b+2*(n//3))
    
#     else:
#         if list(num)[0] == -1:
#             minus_count+= 1
#             return

#         if list(num)[0] == 0:
#             zero_count+= 1
#             return
        
#         if list(num)[0] == 1:
#             plus_count+= 1
#             return
            
# cal(n,0,0)

# print(minus_count)
# print(zero_count)
# print(plus_count)


# 11660
# import sys 
# input = sys.stdin.readline
# n, cnt = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# dp = [[0]*(n+1) for _ in range(n+1)]

# for x in range(1,n+1):
#     for y in range(1,n+1):
#         dp[x][y] = dp[x-1][y] + dp[x][y-1] + board[x-1][y-1] - dp[x-1][y-1]

# for _ in range(cnt):
#     x1,y1,x2,y2 = map(int,input().split())
#     print(dp[x2][y2] - dp[x1-1][y2] - dp[x2][y1-1] + dp[x1-1][y1-1])

# 2447
# 1.
# n = int(input())
# star = [['*']*n for _ in range(n)]

# div = n 
# count = 0
# while div != 0:
#     div //= 3 
#     count += 1

# for cnt in range(count):
#     empty_idx = [i for i in range(n) if (i // (3 ** cnt)) % 3 == 1]
#     for x in empty_idx:
#         for y in empty_idx:
#             star[x][y] = ' '

# for i in range(n):
#     for j in range(n):
#         print(star[i][j], end='')
#     print()

# 2.
# def con(r1,r2):
#     return[''.join(x) for x in zip(r1,r2,r1)]

# def star10(n):
#     if n == 1:
#         return['*']
#     n//=3 
#     x = star10(n)
#     a = con(x,x)
#     b = con(x,[' '*n]*n)

#     return a+b+a 
# print('\n'.join(star10(int(input()))))


# 16953
# a, b = map(int,input().split())
# result = []
# cnt = 1

# def cal(a,b,cnt):
#     if a == b:
#         result.append(cnt)
#         return 

#     if a > b:
#         return 

#     else:  
#         a*=2
#         cnt+=1 
#         cal(a,b,cnt)
        
#         a//=2 
#         a = (a*10+1)
#         cal(a, b, cnt)

# cal(a,b,cnt)

# if len(result) == 0:
#     print(-1)
# else:
#     print(min(result))


# 11286
# import sys 
# import heapq

# input = sys.stdin.readline
# heap = []
# for _ in range(int(input())):
#     num = int(input())

#     if num != 0:
#         heapq.heappush(heap,(abs(num),num))
#         print(heap)
#     else:
#         if len(heap) == 0:
#             print(0)
#         else:
#             print(heapq.heappop(heap)[1])

# 11403
# import sys 
# sys.setrecursionlimit(1000)

# n = int(input())
# board = [list(map(int,input().split())) for _ in range(n)]

# def dfs(x):
#     for y in range(n):
#         if board[x][y] == 1 and y not in visit:
#             visit.append(y)
#             dfs(y)

# for i in range(n):
#     visit = []
#     dfs(i)
#     for j in visit:
#         board[i][j] = 1 

# for i in board:
#     print(' '.join(map(str,i)))


# 10026
# import copy
# import sys 
# sys.setrecursionlimit(10000)
# input = sys.stdin.readline 

# dx,dy = [1,0,-1,0], [0,1,0,-1]

# n = int(input())
# color = [list(input()) for _ in range(n)]
# check = [[0]*n for _ in range(n)]

# gr_color = copy.deepcopy(color)
# gr_check = copy.deepcopy(check)

# person = 0
# gr_person = 0

# def dfs(x,y,arr,ch):
#     ch[x][y] = 1

#     for i in range(4):
#         xx,yy = x+dx[i], y+dy[i]
#         if xx < 0 or xx > n-1 or yy < 0 or yy > n-1:
#             continue 

#         if ch[xx][yy] == 0 and arr[xx][yy] == arr[x][y]:
#             dfs(xx,yy, arr,ch)


# for i in range(n):
#     for j in range(n):
#         if gr_color[i][j] == 'R':
#             gr_color[i][j] = 'G'

# for i in range(n):
#     for j in range(n):
#         if check[i][j] == 0:
#             dfs(i,j,color,check)
#             person += 1


# for i in range(n):
#     for j in range(n):
#         if gr_check[i][j] == 0:
#             dfs(i,j,gr_color,gr_check)
#             gr_person += 1

# print(person)
# print(gr_person)


# 1107
# brButNum이 0이면 brButton은 존재하지 않는다.
# 그러면 밑의 code가 돌아가지 않는다. 그래서 항상 있는 enButton으로 실행을 해야한다.
# import sys 
# input = sys.stdin.readline 

# wish_num = int(input())
# brButNum = int(input())
# enButton = set(str(i) for i in range(10))
# if brButNum == 0:
#     pass
# else:
#     brButton = set(input().split())
#     enButton -= brButton

# cnt = abs(wish_num-100) # 100에서 시작 

# for ch in range(1000000):
#     check = True
#     for num in str(ch):
#         if num not in enButton:
#             check = False
#             break

#     if check:
#         cnt = min(cnt, abs(wish_num - ch)+len(str(ch)))

# print(cnt)


# # 1043
# # 과장된 이야기를 할수 있는 파티공간
# # 모든 파티가 동시에 열리고 있는 상황으로 먼저 진실된 사람과 같이 있는 사람들을
# # 모두 진실된 사람으로 만들어야 한다.
# import sys
# import copy 

# people, party = map(int,input().split())
# true_person = list(map(int,input().split())) 
# total_party = [] 
# p_party = 0

# if true_person[0] == 0:
#     print(party)
#     sys.exit()
    
# true_person = set(true_person[1:])

# for _ in range(party):
#     join_party = list(map(int,input().split()))
#     total_party.append(join_party[1:])
    
# while True:
#     true_persons = copy.deepcopy(true_person)
#     for j in total_party:
#         for k in j:
#             if k in true_person:
#                 true_person.update(j)
#                 break
#     if true_person == true_persons: #while문을 끝내주기 위해 변화가 없으면 break
#         break
          
# for j in total_party:
#     check = True 
#     for k in j:
#         if k in true_person:
#             check = False 
#             break 
#     if check:
#         p_party+= 1

# print(p_party)


# 7568
# n = int(input())
# pe = []
# seq = []

# for _ in range(n):
#     wei, hei = map(int, input().split())
#     pe.append((wei,hei))

# for a in range(len(pe)):
#     cnt = 1
#     for b in range(len(pe)):
#         if pe[a][0] < pe[b][0] and pe[a][1] < pe[b][1]:
#             cnt += 1 
    
#     seq.append(cnt)

# for i in seq:
#     print(i, end = ' ')

# 1916
# import heapq 
# import sys 

# city = int(input())
# bus = int(input())

# wei = [sys.maxsize] * (city+1)
# graph = [[] for _ in range(city+1)]

# for _ in range(bus):
#     st, en, we = map(int,input().split())
#     graph[st].append((we, en))

# start, end = map(int,input().split())


# def djikstra(s, e):
#     heap = []
#     heapq.heappush(heap,(0,s))
#     wei[s] = 0 

#     while heap:
#         w, n = heapq.heappop(heap)

#         for next_w, next_n in graph[n]: 
#             if next_w + wei[n]< wei[next_n]:
#                 wei[next_n] = next_w + wei[n]
#                 heapq.heappush(heap,(next_w + wei[n], next_n)) 

#     return wei[end]

# print(djikstra(start, end))

# 14502 
# import sys 
# import copy
# from collections import deque

# sys.setrecursionlimit(1000)
# input = sys.stdin.readline 

# dx, dy = [0,1,0,-1], [1,0,-1,0]

# n,m = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]

# result = 0

# def wall(cnt):
#     if cnt == 3:
#         bfs()
#         return 
    
#     for i in range(n):
#         for j in range(m):
#             if board[i][j] == 0:
#                 board[i][j] = 1 
#                 wall(cnt+1)
#                 board[i][j] = 0 

# def bfs():
#     global result

#     boards = copy.deepcopy(board)
#     visit = deque() 
#     count = 0
    
#     for i in range(n):
#         for j in range(m):
#             if boards[i][j] == 2:
#                 visit.append((i,j))

#     while visit:
#         x,y = visit.popleft()
        
#         for i in range(4):
#             xx,yy = x+dx[i], y+dy[i]

#             if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#                 continue 
            
#             if boards[xx][yy] == 0:
#                 boards[xx][yy] = 2
#                 visit.append((xx,yy))

#     for i in boards:
#         for j in i:
#             if j == 0:
#                 count+=1 

#     result = max(result, count)

# wall(0)
# print(result)

# 12851
# # 이미 방문했더라도 이동거리 같으면 time[move] == time[loc]+1 
# from collections import deque 
# a,b = map(int,input().split())
# time = [0] * 100001  #방법의 수를 나타내는 곳 

# def bfs(x,y):
#     q = deque() 
#     q.append(x)
#     cnt = 0 

#     while not cnt:
#         size = len(q)  

#         while size: #같은 level의 node를 한번에 다 파악하기 위함(빠른 속도파악)
#             size -= 1 
#             loc = q.popleft() 
#             if loc == b:
#                 #여러번 돌아가면서 속도가 같으면서 해당 방법에 도착하면 +1
#                 cnt+=1 
            
#             for move in (loc-1, loc+1, 2*loc):
#                 if move < 0 or move > 100000:
#                     continue 

#                 if time[move] == 0 or time[move] == time[loc]+1: 
#                     # time[move]가 0 이고 들어오면 1번 가는 것이다.
#                     # time[move] == time[loc] + 1은 if loc==b:를 위한 조건입니다.
#                     # 해당 조건이 없으면 cnt+=1 도착 속도가 다른 것도 포함됩니다.
#                     time[move] = time[loc]+1  #해당 값은 2번째 조건에 상관이 크게 없습니다.(주로 time[move] == 0)
#                     q.append(move)
    
#     print(time[b])
#     print(cnt)

# bfs(a,b)


# 13549
# 2*x의 위치고 이동할 때 0초 x-1, x+1은 1초이므로
# q에 넣을 때 2*x를 먼저 pop할 수 있게 계산 해야합니다.
# 0초 일때가 존재하기 때문에 -1로 해서 값을 짜야합니다. 

# from collections import deque
# import sys 

# input = sys.stdin.readline

# x,y = map(int,input().split())
# time = [-1] * 100001 

# def bfs(a,b):
#     q = deque() 
#     q.append(a)
#     time[a] = 0

#     while q:
#         loc = q.popleft() 

#         for move in (2*loc, loc-1, loc+1):
#             if 0 <= move < 100001 and time[move] == -1:
#                 if move == 2*loc:
#                     time[move] = time[loc] 
#                     q.appendleft(move)
#                 else:   
#                     time[move] = time[loc] + 1
#                     q.append(move)

#     return time[b]
# print(bfs(x,y))


# 5525
# n = int(input())
# m = int(input()) 
# s = input() 

# count = 0
# ioi = 0
# a = 1

# for i in range(0,len(s)-2,a):
#     if 'I' == s[i]:
#         if 'O' == s[i+1] and 'I' == s[i+2]:
#             ioi += 1
#             a = 2
            
#             if ioi == n:
#                 count+=1 
#                 ioi-= 1 
#         else:
#             ioi = 0
#             a=1

# print(count)

# 15686
# from collections import deque

# n, m = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# result_v = 100000
# home, chicken = deque(), deque()
# chi_combination = deque()

# def sol(idx, cnt):
#     # idx는 chicken에 있는 변수를 조합하기 위해 사용하는 수
#     # cnt는 chicken집 수 파악
#     global result_v 
#     print(chi_combination, 'idx')

#     if idx > len(chicken):
#         # idx는 chicken의 변수를 조합하기 위한 변수로 
#         # chicken의 개수를 넘어가면 안된다.
#         return 

#     if cnt == m:
#         short_l = 0
#         for hx,hy in home:
#             result = 1000000

#             for i in chi_combination:
#                 cx, cy = chicken[i]
#                 result = min(result, (abs(hx-cx) + abs(hy - cy)))

#             short_l += result

#         result_v = min(result_v, short_l)
#         return 

#     #cnt수가 원하는 수에 맞게 조합하는 재귀 
#     # 01 -> 02 -> 03 ...
#     chi_combination.append(idx)
#     sol(idx+1, cnt+1)
#     chi_combination.pop()
#     # pop해주지 않으면 계속 쌓여간다.
#     sol(idx+1, cnt)

# for i in range(n):
#     for j in range(n):
#         if board[i][j] == 1:
#             home.append((i+1,j+1))

#         elif board[i][j] == 2:
#             chicken.append((i+1,j+1))

# sol(0, 0)
# print(result_v)

# 14500
# 전체를 회전하면 -로 인해 이상한 도형이 발생할 수 있습니다.
# import sys 
# input = sys.stdin.readline 

# tech = [[(0,0),(0,1),(0,2),(0,3)], [(0,0),(0,1),(1,0),(1,1)],[(0,0),(1,0),(2,0),(0,1)],[(0,0),(1,0),(0,1),(-1,1)],[(0,0),(0,1),(0,2),(-1,1)]]

# n, m = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# result = 0
# big = 0

# if n > m:
#     big = n
# else:
#     big = m

# boards = [[0] * big for _ in range(big)]

# for i in range(n):
#     for j in range(m):
#         boards[i][j] = board[i][j]


# def rotation90(b):
#     global big
#     nb = [[0]*big for _ in range(big)]

#     for i in range(big):
#         for j in range(big):
#             nb[j][big-i-1] = b[i][j]

#     return nb

# def sol(x,y):
#     global result, big

#     for t in tech:
#         total = 0
#         for tx, ty in t:
#             xx, yy = x+tx, y+ty

#             if xx < 0 or xx > big-1 or yy < 0 or yy > big-1:
#                 break
#             total += boards[xx][yy]
#         result = max(result, total)
                
# #board 4회전하기
# for _ in range(5):
#     boards = rotation90(boards)
#     for l in range(big):
#         for m in range(big):
#             sol(l,m)

# print(result)

# 2.
# import sys 
# input = sys.stdin.readline 

# n, m = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# result = 0

# tech = [
#     [(0,1), (1,0), (1,1)], # ㅁ
#     [(0,1), (0,2), (0,3)], # ㅡ
#     [(1,0), (2,0), (3,0)], # ㅣ
#     [(0,1), (0,2), (1,0)], # ㄴ
#     [(0,1), (0,2), (-1,2)], # z
#     [(1,0), (1,1), (1,2)], # ㄱ
#     [(0,1), (0,2), (1,2)], #...
#     [(1,0), (2,0), (2,1)],
#     [(0,1), (1,1), (2,1)],
#     [(0,1), (1,0), (2,0)],
#     [(1,0), (2,0), (2,-1)],
#     [(1,0), (1,1), (2,1)],
#     [(0,1), (1,0), (-1,1)],
#     [(0,1), (1,0), (1,-1)],
#     [(0,1), (1,1), (1,2)],
#     [(0,1), (0,2), (1,1)],
#     [(1,0), (1,1), (1,-1)],
#     [(1,0), (2,0), (1,-1)],
#     [(1,0), (1,1), (2,0)]
# ]

# def sol(x, y):
#     global result

#     for i in range(19): #전체 모양
#         total = board[x][y]
#         for j in range(3):
#             xx = x+tech[i][j][0]
#             yy = y+tech[i][j][1]
            
#             if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#                 break

#             total +=board[xx][yy]

#         result = max(result, total)

# for i in range(n):
#     for j in range(m):
#         sol(i, j)

# print(result)


# 16236
# # 최소 거리를 minheap을 이용하여 해결하였습니다.(djikstra 같이)
# import heapq
# dx,dy = [1, -1, 0, 0], [0, 0, -1, 1] #상하좌우 

# n = int(input())

# board = [list(map(int,input().split())) for _ in range(n)]
# size, eat, result = 2, 0, 0

# def bfs(a, b):
#     global size, eat, result
#     heap = []
#     heapq.heappush(heap,(0,a,b))
#     board[a][b] = 0 #0으로 두어 다음 먹이로 갈때 지나 갈수 있게 해준다.
#     check= [[False]*n for _ in range(n)]
#     while heap:
#         dist, x, y = heapq.heappop(heap)
#         if 0 < board[x][y] < size:
#             eat+=1 
#             #0으로 두어 다음 먹이로 갈때 지나 갈수 있게 해준다.
#             board[x][y] = 0 
            
#             if size == eat:
#                 size+= 1 
#                 eat = 0 

#             # 이동한 곳에서 부터 거리를 측정
#             check= [[False]*n for _ in range(n)]
#             result+= dist
#             dist = 0 

#             while heap:
#                 heap.pop() 
            

#         for i in range(4):
#             ndist, xx, yy = dist+1, x+dx[i], y+dy[i]
        
#             if xx < 0 or xx > n-1 or yy < 0 or yy > n-1:
#                 continue 

#             if board[xx][yy] > size or check[xx][yy] == True:
#                 # check가 없으면 갔던 곳을 간다.
#                 continue 

#             check[xx][yy] = True
#             heapq.heappush(heap,(ndist,xx,yy))


# # 상어 위치 파악
# for i in range(n):
#     for j in range(n):
#         if board[i][j] == 9:
#             sx, sy = i,j 

# bfs(sx, sy)
# print(result)

# 1504
# # 1->x->y->arrive이면 
# # 1->x + x->y + y->arrive의 값
# # 1->y + y->x + x->arrive의 값 을 비교하여 최소값을 구하면 된다.
# import heapq

# n, e = map(int,input().split())
# graph = [[] for _ in range(n+1)]

# for _ in range(e):
#     a, b, c = map(int,input().split()) #a에서 b까지 양방향, c: 거리
#     graph[a].append((c, b))
#     graph[b].append((c, a))

# no1, no2 = map(int,input().split())

# def djikstra(s, e):
#     wei = [100000]*(n+1)
#     heap = []
#     heapq.heappush(heap,(0,s))
#     wei[s] = 0

#     while heap:
#         length, node = heapq.heappop(heap)

#         for leng, no in graph[node]:
#             if wei[no] > wei[node] + leng:
#                 wei[no] = wei[node] + leng 
#                 heapq.heappush(heap,(wei[no], no))
#     return wei[e]

# path1 = djikstra(1, no1) + djikstra(no1, no2) +djikstra(no2, n)
# path2 = djikstra(1, no2) + djikstra(no2, no1) +djikstra(no1, n)
                    
# result = min(path1,path2)
# if result < 100000:
#     print(result)
# else:
#     print(-1)


# 1967.
# 끝들로 부터 최대 값을 구하면 된다.
# 1로 시작해서 끝부분을 찾고 가장 긴 곳을 찾는다 그리고 그 가장 긴 끝에서 
# djikstra로 가장 긴 쪽을 찾는다 .
# djikstar가 짧은 것을 계산하지만 모두 한 방향밖에 없으므로 그냥 하면 된다.
# import heapq 
# import sys 
# input = sys.stdin.readline 
# sys.setrecursionlimit(10000)

# n = int(input())
# graph = [[] for _ in range(n+1)]

# def djikstra(start):
#     global n
#     wei = [1000000] * (n+1) 
#     heap = [] 
#     heapq.heappush(heap,(0, start))
#     while heap:
#         weight, node = heapq.heappop(heap)
#         for we, no in graph[node]:
#             if wei[no] > weight + we:
#                 wei[no] = weight + we
#                 heapq.heappush(heap,(weight+we,no))
#     return wei

# for _ in range(n-1):
#     pa, ch, we = map(int,input().split()) 
#     graph[pa].append((we,ch))
#     graph[ch].append((we,pa))


# w1 = djikstra(1)

# end_value = 0
# end_idx = 0
# for i in range(1, n+1):
#     if end_value < w1[i]:
#         end_value = w1[i]
#         end_idx = i 

# print(max(djikstra(end_idx)[1:]))

# 9466
# import sys 
# input = sys.stdin.readline 

# sys.setrecursionlimit(10000)

# def dfs(x):
#     global n_solo
#     check[x] = True
#     visit.append(x)
#     next_n = link[x]
    
#     if check[next_n] == True:
#         if next_n in visit: #cycle
#             n_solo.extend(visit[visit.index(next_n):]) 
#         return
#     else:
#         dfs(next_n)
     
# for _ in range(int(input())):
#     n = int(input())
#     link = [0] + list(map(int,input().split()))
#     check = [True] + [False]*n 
#     n_solo = []

#     for i in range(1,n+1):
#         if check[i] == False:
#             visit = []
#             dfs(i)
           
#     print(n-len(n_solo))

# 2096
# import sys
# input = sys.stdin.readline 

# n = int(input()) #줄 
# board = []

# for _ in range(n):
#     board.append(list(map(int,input().split())))

# Max = board[0]
# Min = board[0]

# for i in range(1,n): 
#     Max = [max(Max[0],Max[1])+board[i][0], max(Max[0],Max[1],Max[2])+board[i][1], max(Max[1],Max[2])+board[i][2]]
#     Min = [min(Min[0],Min[1])+board[i][0], min(Min[0],Min[1],Min[2])+board[i][1], min(Min[1],Min[2])+board[i][2]] 

# print(max(Max))
# print(min(Min))

#2206
#벽을 1번 부실수 있는데 모든 경우를 파악하기 위해 3차원 배열을 사용합니다. 
#bfs를 하는 이유는 먼저 오는 것이 가장 짧다는 가정하에 하는 것이다.

# import sys 
# from collections import deque 

# input = sys.stdin.readline 

# dx,dy = [0,-1,0,1],[1,0,-1,0]
# n,m = map(int,input().split())
# board = [list(input()) for _ in range(n)]
# dist = [[[0,0] for _ in range(m)] for _ in range(n)] #앞 0,0은 0일때 or 1일때 / [x좌표][y좌표][벽 부순 여부]

# def bfs():
#     q = deque()
#     q.append((0,0,0))
#     dist[0][0][0] = 1 

#     while q:
#         d,x,y = q.popleft()
#         if x == n-1 and y == m-1:
#             return dist[x][y][d]

#         for i in range(4):
#             xx,yy = x+dx[i], y+dy[i]
            
#             if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#                 continue 
            
#             if board[xx][yy] == '0' and dist[xx][yy][d] == 0:
#                 dist[xx][yy][d] = dist[x][y][d] + 1 
#                 q.append((d,xx,yy))
            
#             if board[xx][yy] == '1' and d == 0:
#                 dist[xx][yy][1] = dist[x][y][d] + 1
#                 q.append((1,xx,yy))
#     return -1 

# print(bfs())


# 1389 
# from collections import deque 
# import sys 

# input = sys.stdin.readline 

# n, m = map(int,input().split())
# link = [[] for _ in range(n+1)]
# result = []

# for _ in range(m):
#     a,b = map(int,input().split())
    
#     link[a].append(b)
#     link[b].append(a)

# def bfs(start, end):
#     q = deque()
#     q.append((0,start))

#     while q:

#         dist, num = q.popleft() 
#         if num == end:
#             return dist  

#         for i in link[num]:
#             q.append((dist+1,i))
    

# for i in range(1,n+1):
#     total = 0
#     for j in range(1,n+1):
#         if i != j:
#             total += bfs(i,j)
#     result.append(total)


# for i in range(n):
#     if result[i] == min(result):
#         print(i+1)
#         break 


# 11404
# import sys 

# input = sys.stdin.readline 

# n = int(input()) #도시의 개수
# m = int(input()) #버스의 개수
# dist = [[10000001]*n for _ in range(n)]

# for _ in range(m):
#     a,b,c = map(int,input().split()) #시작도시, 도착도시, 비용 
#     if dist[a-1][b-1] > c:
#         dist[a-1][b-1] = c 

# def floid():
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 if j!=k and dist[j][k] > dist[j][i]+dist[i][k]:
#                     dist[j][k] = dist[j][i] + dist[i][k]

# floid()

# for i in range(n):
#     for j in range(n):
#         if dist[i][j] == 10000001:
#             print(0, end = ' ')
#         else:
#             print(dist[i][j], end= ' ')
#     print()



# 1918
# from collections import deque 

# notation = list(input())
# sign = {'+':1, '-':1, '*':2, '/':2, '(':0}

# def postfix():
#     result = deque()
#     si = deque() 
    
#     for ch in notation:
#         if 'A' <= ch <= 'Z':
#             result.append(ch)

#         elif ch == '(':
#             si.append(ch)

#         elif ch == ')':
#             while si and si[-1] != '(':
#                 result.append(si.pop())
#             si.pop()

#         else:
#             while si and sign[ch] <= sign[si[-1]]:
#                 result.append(si.pop())
#             si.append(ch)

#     while si:
#         result.append(si.pop())

#     return result

# print(''.join(postfix()))


# 1238
# import heapq

# num, bridge_num, x = map(int,input().split())
# bridge = [[] for _ in range(num+1)]
# result = []

# for _ in range(bridge_num):
#     start, end, time = map(int,input().split())
#     bridge[start].append((time, end))


# def djikstra(s, e):
#     heap = []
#     heapq.heappush(heap,(0,s))
#     wei = [100001] * (num+1)
#     wei[s] = 0
    
#     while heap:
#         t, n = heapq.heappop(heap)

#         for ti, no in bridge[n]:
#             if wei[no] > ti + t:
#                 wei[no] = ti+t 
#                 heapq.heappush(heap,(ti+t,no))
#     return wei[e] 


# for i in range(1,num+1):
#     # 숫자 -> x
#     result.append(djikstra(i,x))
#     # x-> 숫자
#     result[i-1] += djikstra(x,i)

# print(max(result))


# 1167
# #트리로 djikstra로 끝들을 찾은 후 그중 가장긴 node를 선택한다.
# #이 node를 가지고 가장 긴 점을 찾는다 그것이 지름이다.

# import heapq 
# import sys 

# v = int(input())
# graph = [[] for _ in range(v)]

# for i in range(v):
#     tree = list(map(int,input().split()))
#     node = tree.pop(0)
#     while tree[0] != -1:
#         node2 = tree.pop(0)
#         weight = tree.pop(0)
#         graph[node-1].append((weight, node2-1))

# def djikstra(s):
#     heap = []
#     heapq.heappush(heap, (0,s))
#     wei = [sys.maxsize] * v 
#     wei[s] = 0 

#     while heap:
#         we, no = heapq.heappop(heap)
#         for w, n in graph[no]:
#             if wei[n] > we + w:
#                 wei[n] = we + w 
#                 heapq.heappush(heap,(we+w, n))

#     return wei 

# dist = djikstra(0)

# end_idx = 0
# end_value = 0
# for j in range(v): #맨 끝 자식들 찾기(가장 긴 자식이 시작점이다.)
#     if end_value < dist[j]:
#         end_value = dist[j]
#         end_idx = j 

# print(max(djikstra(end_idx)))


# 11779
# 1. 역 추적
# import sys
# import heapq

# n = int(input())
# m = int(input())
# graph = [[] for _ in range(n+1)]
# ex_visit = [0 for _ in range(n+1)] #전의 값을 저장하는 곳

# for _ in range(m):
#     start_c, end_c, cost = map(int,input().split())
#     graph[start_c].append((cost,end_c))

# start, end = map(int,input().split())

# def djikstra(s, e):
#     global ex_visit
#     heap = [] 
#     wei = [sys.maxsize]*(n+1)
#     heapq.heappush(heap,(0,s))
#     wei[s] = 0 

#     while heap:
#         cos, nod = heapq.heappop(heap)

#         for co, no in graph[nod]:
#             if wei[no] > cos + co:
#                 wei[no] = cos + co 
#                 ex_visit[no] = nod 
#                 #가장 최소의 값을 저장한다. 
#                 #이러면 전과 최소로 연결되어 있어 최소값을 가지는 최소경로를 알 수 있다.
#                 heapq.heappush(heap,(co+cos, no))
    
#     return wei[e]

# print(djikstra(start,end))

## 역 추적 하는 방법
# path = [end]
# tmp = ex_visit[end]
# while tmp != 0:
#     path.append(tmp)
#     tmp = ex_visit[tmp]

# print(len(path))
# print(' '.join(map(str,path[::-1]))) #print(*path[::-1])

# 2.경로까지 모두 저장하는 방법 
# import sys 
# import heapq 

# n = int(input())
# m = int(input())
# graph = [[] for _ in range(n+1)]


# for _ in range(m):
#     start_c, end_c, cost = map(int,input().split())
#     graph[start_c].append((cost, end_c))

# start, end = map(int,input().split())
# visit = [[''] for _ in range(n+1)]
# visit[start] = str(start)

# def djikstra(s,e):
#     heap = []
#     wei = [sys.maxsize] * (n+1)
#     wei[s] = 0
#     heapq.heappush(heap,(0,s))

#     while heap:
#         cos,nod = heapq.heappop(heap)

#         for co,no in graph[nod]:
#             if wei[no] > co + cos:
#                 wei[no] = co + cos 
#                 visit[no] = visit[nod] + ',' + str(no) 
#                 heapq.heappush(heap,(cos+co, no))

# djikstra(start, end)
# print(visit)

# 14681 
# x = int(input())
# y = int(input())

# def sol(a,b):
#     if x > 0 and y > 0:
#         print(1)
#         return
#     elif x < 0 and y > 0:
#         print(2)
#         return 
#     elif x < 0 and y < 0:
#         print(3)
#         return 
#     else:
#         print(4)
#         return 

# sol(x,y)

# 17070
# from collections import deque 
# import sys 
# input = sys.stdin.readline 

# n = int(input())
# board = [list(map(int,input().split())) for _ in range(n)]
# #가로, 세로, 대각선(0,1,2) idx를 3차부분에 넣어준다
# pipe = [[[0,0,0] for _ in range(n)] for _ in range(n)]
# pipe[0][1][0] = 1

# for y in range(1,n):
#     if board[0][y] == 0:
#         pipe[0][y][0] = pipe[0][y-1][0]


# for px in range(n):
#     for py in range(1,n):
#         #대각선 
#         if board[px][py] == board[px-1][py] == board[px][py-1] == 0:
#             pipe[px][py][2] = pipe[px-1][py-1][0] + pipe[px-1][py-1][1] + pipe[px-1][py-1][2]
        
#         if board[px][py] == 0:
#             #가로
#             pipe[px][py][0] = pipe[px][py-1][0] + pipe[px][py-1][2]
#             #세로
#             pipe[px][py][1] = pipe[px-1][py][1] + pipe[px-1][py][2]

# print(sum(pipe[-1][-1]))


# 7662
# import sys 
# import bisect
# from collections import deque 

# input = sys.stdin.readline 

# T = int(input())

# for _ in range(T):
#     q = deque()
#     d = dict()
#     Q = int(input())

#     for i in range(Q):
#         word = input().split()
#         value = int(word[1])

#         if word[0] == 'I':
#             try:
#                 d[value] += 1 
#             except:
#                 d[value] = 1
#                 bisect.insort_left(q, value)
#         else:
#             if not q:
#                 continue 
            
#             if value == 1:
#                 if d[q[-1]] == 1:
#                     d.pop(q[-1])
#                     q.pop()
#                 else:
#                     d[q[-1]]-=1
            
#             else:
#                 if d[q[0]] == 1:
#                     d.pop(q[0])
#                     q.popleft()
#                 else:
#                     d[q[0]]-=1
#     if not q:
#         print('EMPTY')
#     else:
#         print(q[-1], q[0])

#1157
# word = input() 
# words = dict()

# for w in word:
#     w = w.upper()
#     if w in words:
#         words[w] += 1
#     else:
#         words[w] = 1

# max_num = 0
# max_word = '' 
# for n in words:
#     if words[n] > max_num:
#         max_num = words[n]
#         max_word = n 

# check = False
# for k in words:
#     if max_word != k:
#         if max_num == words[k]:
#             check = True
#             break

# if check:
#     print('?')
# else:
#     print(max_word)

# 2108 
# from collections import Counter 

# nums = [] 
# cnt = int(input())

# for _ in range(cnt):
#     n = int(input())
#     nums.append(n)


# def Ari(num): #산술평균
#     n = round(sum(num)/len(num))
#     return n  

# def med(num): #중앙값
#     num.sort()
#     mid = num[len(num)//2]
#     return mid

# def mode(num): #최빈값
#     nums_dict = Counter(num)

#     if cnt == 1:
#         return num[0]

#     numss = nums_dict.most_common() 
    
#     return (numss[1][0] if numss[0][1] == numss[1][1] else numss[0][0]) 


# def ran(num): #범위
#     return max(num) - min(num)


# print(Ari(nums))
# print(med(nums))
# print(mode(nums))
# print(ran(nums))

# 9935

# word = input() 
# bomb = input() 
# result = []

# for ch in word:
#     result.append(ch)

#     if len(result) >= len(bomb):
#         check = True

#         for i in range(1,len(bomb)+1):
#             if result[-i] != bomb[-i]:
#                 check = False 
#                 break 

#         if check:
#             for j in range(len(bomb)):
#                 result.pop() 

# if result:
#     print(''.join(result))
# else:
#     print("FRULA")


# 17144
# from collections import deque 
# import copy 

# dx, dy = [0,-1,0,1], [1,0,-1,0] #시계방향

# r,c,t = map(int,input().split()) #행, 열, 초 
# home = [list(map(int,input().split())) for _ in range(r)]

# def dust():
#     h = [[0]*c for _ in range(r)]
#     q = deque()

#     for x in range(r):
#         for y in range(c):
#             if home[x][y] > 0:
#                 q.append((x,y))
#             if home[x][y] == -1:
#                 h[x][y] = -1
#     while q:
#         a,b = q.pop() 
#         cnt = 0

#         for i in range(4):
#             aa,bb = a+dx[i], b+dy[i]

#             if aa < 0 or aa > r-1 or bb < 0 or bb > c-1:
#                 continue 

#             if home[aa][bb] != -1:
#                 h[aa][bb] += home[a][b]//5
#                 cnt+=1
            
#         h[a][b] += (home[a][b] - (home[a][b]//5)*cnt)

#     return h

# def air():
#     #공기 청정기 위치
#     for i in range(r):
#         if home[i][0] == -1:
#             upx = i 
#             downx = i+1 
#             break 
    
#     #위쪽 반시계방향 
#     # 우측으로 이동
#     tmp_r = home[upx][c-1]
#     for i in range(c-1,1,-1):
#         home[upx][i] = home[upx][i-1] 
#     home[upx][1] = 0 #공기청정기 바로 앞이기 때문

#     #위로 이동 
#     tmp_u = home[0][c-1]
#     for i in range(upx-1):
#         home[i][c-1] = home[i+1][c-1]
#     home[upx-1][c-1] = tmp_r 

#     #왼쪽으로 이동 
#     tmp_l = home[0][0]
#     for i in range(c-2):
#         home[0][i] = home[0][i+1]
#     home[0][c-2] = tmp_u 

#     #밑으로 이동 
#     for i in range(upx-1, 1, -1):
#         home[i][0] = home[i-1][0]
#     home[1][0] = tmp_l

#     #아래쪽 시계방향 
#     # 우측 이동
#     tmp_r = home[downx][c-1]
#     for i in range(c-1,1,-1):
#         home[downx][i] = home[downx][i-1]
#     home[downx][1] = 0

#     #아래로 이동
#     tmp_d = home[r-1][c-1]
#     for i in range(r-1,downx+1,-1):
#         home[i][c-1] = home[i-1][c-1]
#     home[downx+1][c-1] = tmp_r 

#     #왼쪽으로 이동
#     tmp_l = home[r-1][0]
#     for i in range(c-2):
#         home[r-1][i] = home[r-1][i+1]
#     home[r-1][c-2] = tmp_d

#     #위로 이동
#     for i in range(downx+1,r-1):
#         home[i][0] = home[i+1][0]
#     home[r-2][0] = tmp_l

# for _ in range(t):
#     home = dust()
#     air() 

# result = 0 
# for i in range(r):
#     for j in range(c):
#         if home[i][j] > 0:
#             result+= home[i][j]
# print(result)


# # 2638
# # 바깥 공기 부분을 검사합니다.(0,0) 부터 bfs() 
# # 그래서 cheese가 있는 부분이 있으면 cheese에 +1을 해주고 
# # cheese가 3이상이면 둘러 쌓여져 있지 않은 공기를 2번 이상 닿기 때문에 사라지게 됩니다.
# from collections import deque 

# dx,dy = [0,1,0,-1],[1,0,-1,0] #동,북,서,남

# n,m = map(int,input().split()) #행, 열 
# cheese = [list(map(int,input().split())) for _ in range(n)]
# time = 0

# def bfs():
#     q = deque() 
#     q.append((0,0))
#     check = [[0]*m for _ in range(n)]
#     check[0][0] = 1 

#     while q:
#         x, y = q.popleft() 

#         for i in range(4):
#             xx,yy = x+dx[i], y+dy[i]

#             if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#                 continue 
            
#             if check[xx][yy] == 0:
#                 if cheese[xx][yy] >= 1:
#                     cheese[xx][yy] += 1 

#                 else:
#                     q.append((xx,yy))
#                     check[xx][yy] = 1 


# def melt():
#     #치즈를 녹이는 곳이다, 치즈가 있는지 파악도 해야 된다. 
#     check_melt = False
#     for x in range(n):
#         for y in range(m):
#             if cheese[x][y] >=3:
#                 cheese[x][y] = 0 
#                 check_melt = True

#             elif cheese[x][y] == 2:
#                 cheese[x][y] = 1 
    
#     return check_melt


# while True:
    
#     bfs() 

#     if melt():
#         time += 1
#     else:
#         break  

# print(time)


# 12852
# import sys 
# input = sys.stdin.readline 

# n = int(input())
# dp = [[0,[]] for _ in range(n+1)]
# dp[1][0] = 0
# dp[1][1] = [1]

# for i in range(2,n+1):
#     dp[i][0] = dp[i-1][0] + 1 
#     dp[i][1] = dp[i-1][1] + [i]

#     if i % 3 == 0 and dp[i][0] > dp[i//3][0] + 1:
#         dp[i][0] = dp[i//3][0] + 1 
#         dp[i][1] = dp[i//3][1] + [i]

#     if i % 2 == 0 and dp[i][0] > dp[i//2][0] + 1:
#         dp[i][0] = dp[i//2][0] + 1
#         dp[i][1] = dp[i//2][1] + [i]

# print(dp[n][0])
# for j in dp[n][1][::-1]:
#     print(j, end=' ')

# 2263.
# import sys 
# sys.setrecursionlimit(1000000)

# n = int(input())
# in_order = list(map(int,input().split())) #in_order로 left, right를 나눠야 한다.
# post_order = list(map(int,input().split()))
# in_location = [0 for _ in range(n+1)] # in_order에서 node의 idx를 알려주는 곳 
# tree = [[0,0] for _ in range(n+1)] #[left, right]

# for i in range(n):
#     in_location[in_order[i]] = i

# def find(i_l, i_r, p_l, p_r):
#     if p_l <= p_r: #post_order 에서는 맨 끝이 root이다 .
#         parent = post_order[p_r] 
        
#         #in_order 에서 left, right를 나눠야 한다.
#         parent_idx = in_location[parent] #in_order에서의 parent_idx를 찾는다.

#        #tree를 넣어 줄때는 post_order로 해줘야 한다.
#         l_cnt = parent_idx - i_l #왼쪽 node 개수, in_order을 기준으로 계산
#         if l_cnt > 0: #트리에 왼쪽 node 추가 
#             tree[parent][0] = post_order[p_l+l_cnt-1]
        
#         r_cnt = i_r - parent_idx #오른쪽 node 개수, in_order을 기준으로 계산 
#         if r_cnt > 0:# 트리에 오른쪽 node 추가 
#             tree[parent][1] = post_order[p_r - 1]

#         find(i_l, parent_idx -1, p_l, p_l+l_cnt-1) #왼쪽 
#         find(parent_idx+1, i_r, p_r-r_cnt, p_r-1) #오른쪽 

# find(0,n-1,0,n-1)

# def pre_order(root):
#     print(root, end=' ')
#     if tree[root][0] != 0:
#         pre_order(tree[root][0])
#     if tree[root][1] != 0:
#         pre_order(tree[root][1])

# pre_order(post_order[-1])

# 2.
# import sys 
# sys.setrecursionlimit(1000000)

# n = int(input())
# in_order = list(map(int,input().split()))
# post_order = list(map(int,input().split()))
# in_location = [0 for _ in range(n+1)]

# for i in range(n):
#     in_location[in_order[i]] = i 

# def find(i_l,i_r,p_l,p_r):
#     if p_l <= p_r:
#         parent = post_order[p_r]
#         parent_idx = in_location[parent]

#         print(parent, end=' ')

#         l_cnt = parent_idx - i_l
#         r_cnt = i_r - parent_idx

#         find(i_l, parent_idx-1, p_l, p_l+l_cnt-1) #왼쪽 
#         find(parent_idx+1, i_r, p_r-r_cnt ,p_r-1) #오른쪽 

# find(0,n-1,0,n-1) 

# 2448
# # 3-> 6 -> 12 -> 24 옆으로 이동
# import math 

# star =  ['  *   ', ' * *  ', '***** ']#이거를 3개 합친걸 생각해서 해야한다.


# def make(s):
#     for i in range(len(star)):
#         star.append(star[i]+star[i])
#         star[i] = (' '*s + star[i] + ' '*s)


# n = int(input())
# k = int(math.log(n//3,2))

# for move in range(k):
#     make(int(pow(2,move))*3)

# for s in range(n):
#     print(star[s])

#모험가 길드
# n = int(input())
# scary = list(map(int,input().split())) #공포도
# team = 0
# count = 0
# scary.sort() 

# for i in scary:
#     count += 1 
#     if count >= i:
#         team+=1 
#         count = 0 

# print(team)

#곱하기 혹은 더하기 
# nums = input() 
# result = int(nums[0])

# for i in range(1, len(nums)): 
#     num = int(nums[i])
#     if num <= 1 or result <= 1:
#         result+= num
#     else:
#         result*= num
# print(result)

#문자열 뒤집기 
# n = input() 
# cnt_one = 0
# cnt_zero = 0 

# if n[0] == '0':
#     cnt_zero += 1 
# else:
#     cnt_one += 1


# for i in range(len(n)-1):
#     if n[i] == '0' and n[i+1] == '1':
#         cnt_one += 1 

#     elif n[i] == '1' and n[i+1] == '0':
#         cnt_zero+=1 

# print(min(cnt_zero, cnt_one))  

#만들 수 없는 금액 
# n = int(input())
# coin = list(map(int,input().split()))
# coin.sort() 

# target = 1
# for i in coin:
#     if target < i:
#         break 
#     target += i

# print(target)

#볼링공 고르기 

# n, m = map(int,input().split())
# weight = list(map(int,input().split()))
# arr = [0 for _ in range(n+1)]
# result = 0

# for i in weight:
#     arr[i] += 1 

# for j in range(1,m+1):
#     n -= arr[j]
#     result += arr[j] * n

#럭키 스트레이트 
# n = list(map(int,input()))

# def sol():
#     left = 0 
#     right = 0 

#     for i in range(len(n)):
#         if i < (len(n)//2):
#             left+=n[i]
#         else:
#             right+=n[i]
    
#     if left != right:
#         print('READY')
#         return 
#     else:
#         print('LUCKY')
#         return 

# sol()

#문자열 재정렬 
# s = input() 

# def sol():
#     num = 0
#     alpha = []
#     for i in range(len(s)):
#         if s[i].isdigit(): #숫자
#             num+=int(s[i])
#         else:
#             alpha.append(s[i])
    
#     alpha.sort() 
#     alpha.append(num)
    
#     print(''.join(alpha))
#     return 

# sol()

# #무지의 먹방 라이브 
# # 시간 초과가 나오기 때문에 heapq에 넣고 숫자가 적은 것부터 계산하는 방식을 사용해야 합니다.
# # 적은 숫자 * 해당 번호 값이 적은 수를 없애는데 걸린 총 시간입니다.  
# #(now - past_time)엥서 이전 초에서 현재초를 빼는 이유는 순서가 1부터 끝까지 가서 되돌아 오기 때문에 
# # 가장 적은 초가 있는 번호를 가기 위해서는 현재 초를 들러야 하기 때문입니다. 

# food = list(map(int,input().split()))
# time = int(input())

# import heapq 

# def solution(food_times, k):
#     if sum(food_times) <= k:
#         return -1 

#     q = [] 

#     # 해당 숫자를 넣기 
#     for i in range(len(food_times)):
#         heapq.heappop(q,(food_times[i],i+1))


#     sum_time = 0 # 총 걸린 시간 
#     past_time = 0 #이전의 시간 => 순서대로 돌아가면서 초가 돌기 때문에 현재초에서 이전초를 빼줘야 합니다.
#     leng = len(food_times) #남아 있는 음식의 개수 

#     while sum_time + (q[0][0]-past_time)*leng <= k:
#         now = heapq.heappop(q)
#         sum_time += (now[0]-past_time) * leng 
#         leng-= 1 
#         past_time = now[0]

#     result = sorted(q, key=lambda x: x[1]) #번호에 맞게 정렬 

#     return result[(k-sum_time)%leng][1]


# 2437
# n = int(input())
# wei = list(map(int,input().split()))
# wei.sort() 

# def sol():
#     value = 1
#     for i in wei:
#         if value < i:
#             break
#         value += i 

#     return value

# print(sol())


# #1946 
# # 서류, 면접점수가 모두 다른 지원자 보다 작으면 선발X
# # 서류 or 면접 점수가 다른 지원자 보다 크면 선발O

# for _ in range(int(input())):
#     n = int(input())
#     q = []
#     cnt = 0
#     for _ in range(n):
#         paper, meet = map(int,input().split())
#         q.append((paper,meet)) 

#     q = sorted(q, key = lambda x: x[0])
#     print(q)
#     min_value = q[0][1]
#     for i in range(n):
#         if q[i][1] > min_value:
#             cnt += 1
#         else:
#             min_value = q[i][1] 

#     print(n-cnt)

# 문자열 압축 
# s = input() 

# def solution(a):
#     result = len(a)
#     for jump in range(1,len(a)//2+1):
#         #몇 자리를 계산 할 것인지 알아보는 곳
#         combine_w = '' # 합쳐진 문자
#         w = a[:jump]  # 비교하는 문자 
#         w_cnt = 1 # 같은 word가 몇번 나왔는지 알려주는 변수 

#         # 겹치는 것이 없다(jump만큼 띈다.)
#         for i in range(jump,len(a)+1,jump):
#             # 다음 w랑 a[i:i+jump]까지 같은 경우 수 += 1 
#             if w == a[i:i+jump]:
#                 w_cnt+=1 
            
#             else:
#                 if w_cnt == 1:
#                     combine_w += w 
#                 else:
#                     combine_w += (str(w_cnt) + w)
                
#                 w = a[i:i+jump]
#                 w_cnt = 1 
        
#         combine_w += w #꼬다리 더해주기

#         result = min(result, len(combine_w))
        
#     return result

# solution(s)

# 5567
# from collections import deque 

# n = int(input())
# m = int(input())
# graph = [[] for _ in range(n+1)]
# check = [False]*(n+1)

# for _ in range(m):
#     a,b = map(int,input().split())
#     graph[a].append(b)
#     graph[b].append(a)

# def bfs(s):
#     q = deque() 
#     q.append(s)
#     check[s] = True 
#     link = 0
#     fri_cnt = 0

#     while q:
#         link+= 1
#         for _ in range(len(q)):
#             x = q.popleft()
#             for fri in graph[x]:
#                 if check[fri] == False:
#                     check[fri] = True 
#                     q.append(fri)
#                     fri_cnt += 1

#         if link == 2:
#             break 

#     return fri_cnt 

# 자물쇠와 열쇠 
# 1.
# import copy 

# n, m = map(int,input().split())

# key = [list(map(int,input().split())) for _ in range(n)]
# lock = [list(map(int,input().split())) for _ in range(m)]

# def rotate(key1):
#     dp = [[0]* n for _ in range(n)]

#     for i in range(n):
#         for j in range(n):
#             dp[j][n-i-1] = key1[i][j]
#     return dp

# def sol(key2):
#     k = []
#     l = [] 
#     lock_c = copy.deepcopy(lock)

#     for i in range(n):
#         for j in range(n):
#             if key2[i][j] == 1:
#                 k.append((i,j))

    
#     for ii in range(m):
#         for jj in range(m):
#             if lock[ii][jj] == 0:
#                 l.append((ii,jj))

#     check = False
#     for x, y in k:
#         for xx,yy in l:
#             dx = xx - x
#             dy = yy - y
            
#             for xxx, yyy in k:
#                 if xxx+dx < 0 or yyy+dy < 0 or xxx + dx > m-1 or yyy+dy > m-1:
#                     continue 
#                 lock_c[xxx+dx][yyy+dy] = 1 

#             for i in lock_c:
#                 if 0 in i:
#                     check = False 
#                     lock_c = copy.deepcopy(lock)
#                 else:
#                     check = True 
                   
#     return check 

# for _ in range(4):
#     key = rotate(key)
#     print(sol(key))
#     break


# 2.
# 2차원 리스트 90도 회전 
# def rotate_a_matrix_by_90_degree(a):
#     n = len(a) #행 길이 계산
#     m = len(a[0]) #열 길이 계산

#     result = [[0]*n for _ in range(m)] #결과 리스트

#     for i in range(n):
#         for j in range(m):
#             result[j][n-i-1] = a[i][j]
    
#     return result 

# # 좌물쇠의 중간 부분이 모두 1인지 확인
# def check(new_lock):
#     length = len(new_lock) // 3 

#     for i in range(length, length*2):
#         for j in range(length, length*2):
#             if new_lock[i][j] != 1:
#                 return False 
#     return True 

# def solution(key, lock):
#     n = len(lock)
#     m = len(key)

#     #좌물쇠의 크기를 기존의 3배로 변환 
#     new_lock = [[0] * (n*3) for _ in range(n*3)]

#     #새로운 자물쇠의 중앙 부분에 기존의 자물쇠 넣기
#     for i in range(n):
#         for j in range(n):
#             new_lock[i+n][j+n] = lock[i][j]

    
#     # 4가지 방향에 대해서 확인
#     for rotation in range(4):
#         key = rotate_a_matrix_by_90_degree(key)
#         for x in range(n*2):
#             for y in range(n*2):
#                 #자물쇠에 열쇠를 끼어 넣기 
#                 for i in range(m):
#                     for j in range(m):
#                         new_lock[x+i][y+i] += key[i][j]

#                 #새로운 자물쇠에 열쇠가 정화히 들어 맞는지 검사 
#                 if check(new_lock) == True:
#                     return True 
                
#                 #자물쇠에서 열쇠를 다시 빼기
#                 for i in range(m):
#                     for j in range(m):
#                         new_lock[x+i][y+i] -= key[i][j]
#     return False 


#3190 
# 뱀의 몸통이 2개 이상일 수 있습니다.
# 그래서 뱀이 있는 부분을 모두 저장해 놓고 
# 사과를 못 먹으면 맨 꼬리부분을 없애 줘야 한다. 
# 새로운 worm을 만들어주면 시간이 오래걸려 

# from collections import deque 

# n = int(input())
# k = int(input())
# board = [[0]*(n+1) for _ in range(n+1)]
# change = [] #회전 정보

# for _ in range(k):
#     a,b = map(int,input().split())
#     board[a][b] = 1  #사과 넣어주기 

# l = int(input())
# for _ in range(l):
#     x, c = input().split() # 문자 L - 왼쪽, D - 오른쪽
#     change.append((int(x), c))

# dx,dy = [0, 1, 0, -1], [1, 0, -1, 0]

# def change_d(direct, word):
#     if word == "L":
#         direct = (direct-1) % 4
#     else:
#         direct = (direct+1) % 4
#     return direct

# def move_worm():
#     x, y = 1, 1
#     q = deque()
#     q.append((x,y))
#     time = 0
#     direction = 0
#     idx = 0
#     board[x][y] = 2
    
#     while True:
#         nx, ny = x+dx[direction], y+dy[direction]

#         # 맵안에 있고 뱀의 몸통 부딪치지 않을 때
#         if 0 < nx and nx < n+1 and 0 < ny and ny < n+1 and board[nx][ny] != 2:
#             #사과 없을 때 
#             if board[nx][ny] == 0:
#                 board[nx][ny] = 2
#                 q.append((nx,ny))
#                 px, py = q.popleft()
#                 board[px][py] = 0

#             if board[nx][ny] == 1:
#                 board[nx][ny] = 2
#                 q.append((nx,ny))

#         else:
#             time+=1 
#             break 
#         x, y = nx, ny 
#         time+= 1
#         if idx < l and time == change[idx][0]:
#             direction = change_d(direction, change[idx][1])
#             idx += 1
#     return time 
# print(move_worm())

#기둥과 보 설치
# n = int(input()) #정사각 격자
# build_frame = []
# while True:
#     try:
#         build_frame.append(list(map(int,input().split())))
#     except:
#         break 
    
# def check(result):
#     for x,y,a in result:
#         #기둥
#         if a == 0:
#             if y == 0 or ([x-1,y,1] in result) or ([x,y,1] in result) or ([x,y-1,0] in result):
#                 return True 
#         #보
#         else:
#             if ([x,y-1, 0] in result) or ([x+1, y-1, 0] in result) or (([x-1, y, 1] in result) and ([x+1, y, 1] in result)):
#                 return True
    
#     return False

# def solution(build_frame, n):
#     result = [] 

#     for x,y,a,b in build_frame:
#         #설치 
#         if b == 1:
#             result.append([x,y,a])
#             if not check(result):
#                 result.remove([x,y,a])
#         #삭제
#         if b == 0:
#             result.remove([x,y,a])
#             if not check(result):
#                 result.append([x,y,a])

        
#     return sorted(result)
# print(solution(build_frame, n))

#2812

# import sys 
# input = sys.stdin.readline

# n,k = map(int,input().split()) 
# num = list(input()) 
# cnt = k
# result = []

# for i in range(n):
#     while cnt > 0 and result and result[-1] < num[i]:
#         del result[-1]
#         cnt -= 1
#     result.append(num[i])

# print(''.join(result[:n-k]))

#15686
# from collections import deque 

# n, m = map(int,input().split())
# city = [list(map(int,input().split())) for _ in range(n)]
# home, chicken = deque(), deque() 
# chi_combination = deque() 
# answer = []

# def solution(idx, cnt):
#     if idx > len(chicken):
#         return 

#     if cnt == m:
#         result = 0
#         for x,y in home:
#             min_chi = 100000
#             for i in chi_combination:
#                 min_chi = min(min_chi, (abs(x-chicken[i][0]) + abs(y-chicken[i][1])))
#             result += min_chi 
#         answer.append(result)
          
#     chi_combination.append(idx)
#     solution(idx+1, cnt+1)
#     chi_combination.pop()
#     solution(idx+1, cnt)

# for i in range(n):
#     for j in range(n):
#         if city[i][j] == 1:
#             home.append((i,j))
#         elif city[i][j] == 2:
#             chicken.append((i,j))

    
# solution(0,0)
# print(min(answer))

# 2.
# from itertools import combinations 

# n,m = map(int,input().split())
# chicken, house = [], [] 

# for r in range(n):
#     data = list(map(int,input().split()))
#     for c in range(n):
#         if data[c] == 1:
#             house.append((r,c))
#         elif data[c] == 2:
#             chicken.append((r,c))

# candidates = list(combinations(chicken, m))

# def get_sum(candidate):
#     result = 0 
#     for hx, hy in house:
#         temp = 1e9
#         for cx, cy in candidate:
#             temp = min(temp, abs(hx-cx)+abs(hy-cy))
#         result += temp
#     return result 

# result = 1e9 
# for candidate in cadidates:
#     result = min(result, get_sum(candidate))

# print(result)


#1188
# 최대공약수 
# n, m = map(int,input().split())

# def GCD(a,b):
#     if a % b == 0:
#         return b 
#     return GCD(b, a%b)
# if n == m:
#     print(0)
# else:
#     gcd = GCD(n,m)
#     print(m-gcd)

#외벽 점검 
# 1.
# from collections import deque 
# from itertools import permutations 

# #친구를 투입하고 나서, 다음 친구를 투입할 때 
# def next_idx(queue, d, start_index=0):
#     start_num = queue[start_index]
#     for i in range(1, d+1):
#         try:
#             if queue[start_index + 1] == start_num + i:
#                 start_index = start_index + 1 
#         except:
#             break 
#     return (start_index + 1)


# def solution(n, weak, dist):
#     dist.sort(reverse=True)
#     weak = deque(weak)

#     for i in range(1,len(dist)+1):
#         if i == 1:
#             for _ in range(len(weak)):
#                 d = dist[0]
#                 if weak[-1] <= weak[0] + d:
#                     return 1 
#                 else:
#                     weak.rotate(-1) #우측으로 1칸 회전시킨다.
#                     weak[-1] = weak[-1] + n 
#             # weak 원상 복구 
#             weak = deque(map(lambda x:x%n, weak))
#         else:
#             dist_2 = list(itertools.permutations(dist[:i]))
#             for select_set in dist_2:
#                 for _ in range(len(weak)):
#                     start_idx = 0 
#                     for d in select_set:
#                         #다음번 친구가 투입될 위치: start_idx 
#                         start_idx = next_idx(weak, d, start_idx)
#                         if start_idx == len(weak):
#                             return i 
#                     weak.rotate(-1)
#                     weak[-1] = weak[-1] + n 
#                     #weak 원상복구
#                 weak = deque(map(lambda x: x%n, weak))
#         return -1


# # 2.
# from itertools import permutations 

# def solution(n,weak, dist):
#     dist.sort(reverse = True)
#     #길이를 2배로 놀려서 원형을 일자 형태로 변형 
#     length = len(weak)
#     for i in range(length):
#         weak.append(weak[i]+n)

#     answer = len(dist) + 1 #투입할 친구 수의 최솟값을 찾아야 하므로 len(dist) + 1로 초기화 

#     #0 부터 length-1까지의 위치를 각각 시작점으로 설정 
#     for start in range(length):
#         #친구를 나열하는 모든 경우의 수 각각에 대하여 확인
#         for friends in list(permutations(dist, len(dist))):
#             count = 1 #투입할 친구의 수 
#             #해당 친구가 점검할 수 있는 마지막 위치 
#             position = weak[start] + friends[count-1]
#             #시작점부터 모든 취약 지점을 확인 
#             for index in range(start, start+length):
#                 #점검할 수 있는 위치를 벗어나는 경우
#                 if position < weak[index]:
#                     count += 1  #새로운 친구를 투입
#                     if count > len(dist): # 더 투입이 불가능하다면 종료 
#                         break 

#                     position = weak[index] + friends[count - 1]
#             answer = min(answer, count) #최솟값 계산 
#     if answer > len(dist):
#         return -1 
#     return answer 


# 2636

# from collections import deque 

# dx,dy = [1,0,-1,0], [0,1,0,-1]

# a,b = map(int,input().split()) #세로, 가로
# board = [list(map(int,input().split())) for _ in range(a)]
# time = 0 
# result = []

# def bfs(sx,sy):
#     q = deque()
#     check = [[False]*b for _ in range(a)]
    
#     q.append((sx,sy))
#     check[sx][sy] = True

#     cnt = 0

#     while q:
#         x,y = q.popleft()

#         for i in range(4):
#             xx,yy = x+dx[i], y+dy[i]
            
#             if xx < 0 or yy < 0 or xx > a-1 or yy > b-1:
#                 continue 
            
#             if check[xx][yy] == False and board[xx][yy] == 0:
#                 check[xx][yy] = True 
#                 q.append((xx,yy))
            
#             elif check[xx][yy] == False and board[xx][yy] == 1:
#                 cnt+= 1
#                 board[xx][yy] = 2
#                 check[xx][yy] = True

#     return cnt



# while True:

#     check_b = True

#     for cheese in board:
#         if 1 in cheese:
#             check_b = False

#     result.append(bfs(0,0))
    
#     for i in range(a):
#         for j in range(b):
#             if board[i][j] == 2:
#                 board[i][j] = 0
    
#     if check_b:
#         print(time)
#         print(result[-2])
#         break 
#     else:
#         time += 1


#18352
# from collections import deque 

# n,m,k,x = map(int,input().split()) 

# city = [[] for _ in range(n+1)]  
# distance = [-1] * (n+1)

# for _ in range(m):
#     a,b = map(int,input().split())
#     city[a].append(b)

# def bfs(start):
#     q = deque()
#     q.append(start)
#     distance[start] = 0

#     while q:
#         num = q.popleft() 

#         for node in city[num]:
#             if distance[node] == -1:
#                 distance[node] = distance[num] + 1
#                 q.append(node)

# bfs(x)

# check = False 
# for c in range(1,n+1):
#     if distance[c] == k:
#         print(c)
#         check = True 

# if check == False:
#     print(-1)


#14502
# 1.
# from itertools import combinations 
# from copy import deepcopy 

# dx,dy = [1,0,-1,0], [0,1,0,-1]

# n,m = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# wall,virus = [],[] 
# result = 0


# def bfs(w1,w2,w3):
#     b = deepcopy(board)

#     b[w1[0]][w1[1]] = 1
#     b[w2[0]][w2[1]] = 1
#     b[w3[0]][w3[1]] = 1
    
#     for sx,sy in virus:
#         q = [] 
#         q.append((sx,sy))

#         while q:
#             x,y = q.pop(0)

#             for i in range(4):
#                 xx,yy = x+dx[i], y+dy[i]

#                 if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#                     continue 

#                 if b[xx][yy] == 0:
#                     b[xx][yy] = 2 
#                     q.append((xx,yy)) 

#     return b

# for i in range(n):
#     for j in range(m):
#         if board[i][j] == 0:
#             wall.append((i,j))
#         elif board[i][j] == 2:
#             virus.append((i,j))

# for k in list(combinations(wall, 3)):
#     virus_board = bfs(k[0],k[1],k[2])

#     safe = 0
#     for i in range(n):
#         for j in range(m):
#             if virus_board[i][j] == 0:
#                 safe+=1 
    
#     result = max(safe, result)

# print(result)

# 2.
# n,m = map(int,input().split())
# data = [] #초기 맵 리스트 
# temp = [[0]*m for _ in range(n)] #벽을 설치한 뒤의 맵 리스트 

# for _ in range(n):
#     data.append(list(map(int,input().split())))


# #4가지 이동 방향에 대한 리스트
# dx = [-1,0,1,0]
# dy = [0,1,0,-1]

# result = 0

# #깊이 우선 탐색(dfs)을 이용해 각 바이러스가 사방으로 퍼지도록 하기
# def virus(x,y):
#     for i in range(4):
#         nx = x+dx[i]
#         ny = y+dy[i]
        
#         #상, 하, 좌, 우 중에서 바이러스가 퍼질 수 있는 경우
#         if nx >= 0 and nx < n and ny >= 0 and ny < m:
#             if temp[nx][ny] == 0:
#                 #해당 위치에 바이러스 배치하고, 다시 재귀적으로 수행 
#                 temp[nx][ny] = 2 
#                 virus(nx,ny)

# #현재 맵에서 안전 영역의 크기 계산하는 매서드
# def get_score():
#     score = 0 
#     for i in range(n):
#         for j in range(m):
#             if temp[i][j] == 0:
#                 score += 1 
#     return score

# #깊이 우선 탐색(DFS)을 이용해 울타리를 설치하면서, 매번 안전 영역의 크기 계산 
# def dfs(count):
#     global result 
#     #울타리가 3개 설치된 경우 
#     if count == 3:
#         for i in range(n):
#             for j in range(m):
#                 temp[i][j] = data[i][j]

#         # 각 바이러스의 위치에서 전파 진행 
#         for i in range(n):
#             for j in range(m):
#                 if temp[i][j] == 2:
#                     virus(i,j)

#         #안전 영역의 최댓값 계산 
#         result = max(result, get_score())
#         return 
    
#     #빈 공간에 울타리 설치 
#     for i in range(n):
#         for j in range(m):
#             if data[i][j] == 0:
#                 data[i][j] = 1 
#                 count += 1 
#                 dfs(count)
#                 data[i][j] = 0 
#                 count -= 1 

# dfs(0)
# print(result)



# #18405
# #tip: for 개수가 최소가 되도록 하자 
# from collections import deque 
# import sys 

# input = sys.stdin.readline 

# dx,dy = [1,-1,0,0], [0,0,-1,1]

# n,k = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# s,x,y = map(int,input().split())
# virus = []

# for i in range(n):
#     for j in range(n):
#         if board[i][j] != 0:
#             virus.append((board[i][j],0,i,j))

# virus.sort() 

# def bfs():
#     q = deque(virus) 

#     while q:
#         vi, ti, xx, yy = q.popleft() 

#         if ti == s:
#             return 

#         for i in range(4):
#             xxx, yyy = xx+dx[i], yy+dy[i]

#             if xxx < 0 or yyy < 0 or xxx > n-1 or yyy > n-1:
#                 continue 

#             if board[xxx][yyy] == 0:
#                 board[xxx][yyy] = vi 
#                 q.append((vi,ti+1,xxx,yyy))   

# bfs() 
# print(board[x-1][y-1])



#괄호 변환
# n = input()

# def balance(a):
#     left = 0 

#     for i in range(len(a)):
#         if a[i] == '(':
#             left+=1 
#         else:
#             left-=1 
        
#         if left == 0:
#             return i 

# def right(a):
#     left = 0 
#     for i in a:
#         if i == '(':
#             left+= 1
        
#         else:
#             if left == 0:
#                 return False
#             left -= 1

#     return True


# def sol(w):
#     result = ''

#     if w == '':
#         return result 

#     idx = balance(w)
#     u = w[:idx+1]
#     v = w[idx+1:]

#     if right(u):
#         result = u + sol(v)

#     else:
#         result = '('
#         result += sol(v)
#         result += ')'
#         u = list(u[1:-1])

#         for i in range(len(u)):
#             if u[i] == '(':
#                 u[i] = ')'
#             else:
#                 u[i] = '('

#         result += ''.join(u)

#     return result 

# print(sol(n))

#14888
# from itertools import permutations 
# import sys 
# input = sys.stdin.readline 

# operation = ['+','-','*','/']
# num = int(input()) #수의 개수 
# nums = input().split()
# op_cnt = list(map(int,input().split())) #덧셈, 뺄셈, 곱셈, 나눗셈 
# op = []


# # 연산자 구하기
# for o in range(4):
#     if op_cnt[o] != 0:
#         for _ in range(op_cnt[o]):
#             op.append(operation[o])
# op_set = set(permutations(op, num-1))

# 1.
# max_result = -sys.maxsize
# min_result = sys.maxsize

# for i in op_set:
#     i = list(i)
#     stack = [] 

#     for j in range(num):
#         stack.append(nums[j])

#         if len(stack) == 3:
#             cal_result = int(eval(''.join(stack)))
#             stack = [str(cal_result)]

#         if j == num-1:
#             break 
#         else:
#             stack.append(i[j])

#     result = eval(''.join(stack))

#     if result > max_result:
#         max_result = result

#     if result < min_result:
#         min_result = result 

# print(max_result)
# print(min_result)

# 2.
# result = [] 
# for i in op_set:
#     i = list(i)
#     cal = int(nums[0])
    
#     for j in range(len(i)):
#         if i[j] == '+':
#             cal += int(nums[j+1])
#         elif i[j] == '-':
#             cal -= int(nums[j+1])
#         elif i[j] == '*':
#             cal *= int(nums[j+1])
#         else:
#             cal = cal//int(nums[j+1]) if cal > 0 else ((-cal)//int(nums[j+1]))*(-1)
    
#     result.append(cal)

# print(max(result))
# print(min(result))

# 3. 재귀

# num = int(input()) #수의 개수 
# nums = list(map(int,input().split()))
# op_cnt = list(map(int,input().split())) #덧셈, 뺄셈, 곱셈, 나눗셈 

# max_result = -1e9 
# min_result = 1e9

# def cal(idx, result):

#     if idx == num:
#         max_result = max(max_result, result)
#         min_result = min(min_result, result)


#     else:
#         if op_cnt[0] > 0:
#             op_cnt[0] -= 1
#             cal(idx+1, result+nums[idx])
#             op_cnt[0] += 1 
        
#         if op_cnt[1] > 0:
#             op_cnt[1] -= 1
#             cal(idx+1, result-nums[idx])
#             op_cnt[1] += 1

#         if op_cnt[2] > 0:
#             op_cnt[2] -= 1 
#             cal(idx+1, result*nums[idx])
#             op_cnt[2] += 1 

#         if op_cnt[3] > 0:
#             op_cnt[3] -= 1
#             cal(idx+1, int(result/nums[idx]))
#             op_cnt[3] += 1 

# cal(1, nums[0])

# print(max_result)
# print(min_result)


#18428
# 1.
# import copy 
# import sys 
# sys.setrecursionlimit(10000)

# n = int(input())
# board = [list(input().split()) for _ in range(n)]
# answer = False
# teacher = []

# def bfs():
#     global teacher
#     boards = copy.deepcopy(board)
#     teachers = copy.deepcopy(teacher)

#     while teachers:
#         x, y = teachers.pop()

#         for i in range(4):
#             if i == 0: #동 
#                 for j in range(y,n):
#                     if boards[x][j] == 'S':
#                         return False

#                     if boards[x][j] == 'O':
#                         break
                    
#             if i == 1: #북
#                 for j in range(x,-1,-1):
#                     if boards[j][y] == 'S':
#                         return False

#                     if boards[j][y] == 'O':
#                         break

#             if i == 2: #서
#                 for j in range(y,-1,-1):
#                     if boards[x][j] == 'S':
#                         return False

#                     if boards[x][j] == 'O':
#                         break

#             if i == 3: #남
#                 for j in range(x,n):
#                     if boards[j][y] == 'S':
#                         return False

#                     if boards[j][y] == 'O':
#                         break
#     return True                    


# def wall(cnt):
#     global answer
#     if cnt == 3: 
#         if bfs():
#             answer = True
#         return 
    
#     for i in range(n):
#         for j in range(n):
#             if board[i][j] == 'X':
#                 board[i][j] = 'O'
#                 wall(cnt+1)
#                 board[i][j] = 'X'


# for i in range(n):
#     for j in range(n):
#         if board[i][j] == 'T':
#             teacher.append((i,j))

# wall(0)

# if answer:
#     print('YES')
# else:
#     print('NO')

# 2.
# from itertools import combinations 

# n = int(input())
# board = [] 
# teachers = [] 
# spaces = [] 

# for i in range(n):
#     board.append(list(input().split()))

#     for j in range(n):
#         if board[i][j] == 'T':
#             teachers.append((i,j))
#         if board[i][j] == 'X':
#             spaces.append((i,j))


# def watch(x,y,direction):
#     if direction == 0:
#         while y >= 0:
#             if board[x][y] == 'S':
#                 return True 
#             if board[x][y] == 'O':
#                 return False 
#             y-=1 

#     if direction == 1:
#         while y < n:
#             if board[x][y] == 'S':
#                 return True 
#             if board[x][y] == 'O':
#                 return False 
#             y+= 1

#     if direction == 2:
#         while x >= 0:
#             if board[x][y] == 'S':
#                 return True 
#             if board[x][y] == 'O':
#                 return False 
#             x-= 1 

#     if direction == 3:
#         while x < n:
#             if board[x][y] == 'S':
#                 return True 
#             if board[x][y] == 'O':
#                 return False 
#         x+= 1 

#     return False 

# def process():
#     for x, y in teachers:
#         for i in range(4):
#             if watch(x,y,i):
#                 return True 
#     return False 

# find = False 

# for data in combinations(spaces, 3):
#     for x,y in data:
#         board[x][y] == 'O'
    
#     if not process():
#         find = True 
#         break 

#     for x, y in data:
#         board[x][y] = 'X'

# if find:
#     print('YES')
# else:
#     print('NO')


#카카오1
# new_id = input()
# def solution(new_id):
    
#     # 1단계
#     new_id = new_id.lower() 
    
#     # 2,3단계 
#     new_id = list(new_id)

#     point = 0
#     for i in range(len(new_id)):
#         if not new_id[i].isalnum():
#             if new_id[i] == '-' or new_id[i] == '_' or new_id[i] == '.':            
#                 if new_id[i] == '.':
#                     point += 1
#                     if point > 1:
#                         new_id[i] = ''
#                 else:
#                     point = 0
            
#             else:
#                 new_id[i] = ''
#         else:
#             point = 0
            
#     new_id = ''.join(new_id)
#     new_id = list(new_id)

#     #4단계
#     if new_id[0] == '.':
#         new_id[0] = ''
#     if new_id[-1] == '.':
#         new_id[-1] = ''

#     new_id = ''.join(new_id)
#     new_id = list(new_id)

#     #5단계
#     if new_id == []:
#         new_id.append('a')
        
#     #6단계
#     if len(new_id) > 15:
#         new_id = new_id[:15]
#         if new_id[-1] == '.':
#             new_id[-1] = ''

#     new_id = ''.join(new_id)
#     new_id = list(new_id)

#     # 7단계
#     if len(new_id) < 3:
#         while True:
#             if len(new_id) == 3:
#                 break

#             new_id.append(new_id[-1])
        
#     new_id = ''.join(new_id)
    
#     return new_id

# print(solution(new_id))

#카카오2 
# import sys 

# n,s,a,b = map(int,input().split())
# fares = [[4, 1, 10], [3, 5, 24], [5, 6, 2], [3, 1, 41], [5, 1, 24], [4, 6, 50], [2, 4, 66], [2, 3, 22], [1, 6, 25]]
# graph = [[] for _ in range(n+1)]

# for aa,bb,ff in fares:
#     graph[aa].append((bb,ff))
#     graph[bb].append((aa,ff))


# def djikstra(s,e):
#     wei = [1e9] * (n+1)
#     wei[s] = 0 
#     heap = []
#     heapq.heappush(heap,(s,0))

#     while heap:
#         node, fare = heapq.heappop(heap)

#         for next_node, fa in graph[node]:
#             if wei[next_node] > wei[node] + fa:
#                 wei[next_node] = wei[node] + fa
#                 heapq.heappush(heap,(next_node, wei[node]+fa))

#     return wei[e]

# case_one = djikstra(s,a) + djikstra(a,b)
# case_two = djikstra(s,b) + djikstra(b,a)
# case_three = djikstra(s,a) + djikstra(s,b)

# print(min(case_one, case_two, case_three))


#프로그래머스 풀이
# import sys 
# import heapq 
# input = sys.stdin.readline 

# def djikstra(gr, num, st ,e):
#     wei = [1e9]*(num+1)
#     wei[st] = 0
#     heap = [] 
#     heapq.heappush(heap,(st,0))
    
#     while heap:
#         node, fare = heapq.heappop(heap)
        
#         for next_n, fa in gr[node]:
#             if wei[next_n] > wei[node] + fa:
#                 wei[next_n] = wei[node] + fa
#                 heapq.heappush(heap,(next_n, wei[node]+fa))
                
#     return wei[e]


# def solution(n, s, a, b, fares):
#     graph = [[] for _ in range(n+1)]
    
#     for aa,bb,ff in fares:
#         graph[aa].append((bb,ff))
#         graph[bb].append((aa,ff))
        
#     case_one = djikstra(graph,n,s,a) + djikstra(graph,n,a,b)
#     case_two = djikstra(graph,n,s,b) + djikstra(graph,n,b,a)
    
#     case_three = 1e9 
#     for i in range(1,n+1):
#         if not graph[i]:
#             continue 
#         case_three = min((djikstra(graph,n,s,i) +djikstra(graph,n,i,a) + djikstra(graph,n,i,b)), case_three)

#     answer = min(case_one, case_two, case_three)
    
#     return answer

# 카카오 3
# from itertools import permutations 
# orders = ["ABCFG", "AC", "CDE", "ACDE", "BCFG", "ACDEH"]
# data = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}

# def check_in(order_menu, set_m):
#     for menu in set_m:
#         if not menu in order_menu:
#             return False
#     return True

# def check_order(set_m):
#     for i in range(len(set_m)-1):
#         if ord(set_m[i]) > ord(set_m[i+1]):
#             return False 
#     return True

# def solution():
#     answer = []
#     for order in orders:
#         for i in range(2,len(order)+1):
#             set_menus = list(permutations(list(order),i))
#             for set_menu in set_menus:
#                 set_menu = ''.join(set_menu)
                    
#                 if check_order(set_menu):
#                     count = 0
#                     for order2 in orders:
#                          if check_in(order2, set_menu):
#                              count += 1
                    
#                     if count > 2:
#                         answer.append(set_menu)

#     return set(answer) 

# print(solution())

#카카오4
# infos = ["java backend junior pizza 150","python frontend senior chicken 210","python frontend senior chicken 150","cpp backend senior pizza 260","java backend junior chicken 80","python backend senior chicken 50"]
# querys = ["java and backend and junior and pizza 100","python and frontend and senior and chicken 200","cpp and - and senior and pizza 250","- and backend and senior and - 150","- and - and - and chicken 100","- and - and - and - 150"]

# def check(info, q):
#     for i in range(5):
#         if i != 4:
#             if q[i] != '-' and q[i] != info[i]:
#                 return False
#         else:
#             if int(q[i]) > int(info[i]):
#                 return False
#     return True

# def sol(query, infos):
#     count = 0
#     for info in infos:
#         info = info.split(' ')

#         if check(info, query):
#             count += 1
        
#     return count


# def solution(infos, querys):
#     result = []
#     for query in querys:
#         query = query.split('and ')
#         query = ''.join(query)
#         query = query.split(' ')
#         result.append(sol(query,infos))

#     return result

# print(solution(infos, querys))


#16234
# 1.
# from collections import deque 

# dx,dy = [1,0,-1,0],[0,1,0,-1]
# n,l,r = map(int,input().split())
# a = [list(map(int,input().split())) for _ in range(n)]

# def bfs(sx,sy):
#     q = deque() 
#     move = deque() 
#     q.append((sx,sy))
#     move.append((sx,sy))
#     move_cnt = a[sx][sy]
#     check[sx][sy] = True

#     while q:
#         x,y = q.popleft() 

#         for i in range(4):
#             xx,yy = x+dx[i], y+dy[i]

#             if xx < 0 or xx > n-1 or yy < 0 or yy > n-1:
#                 continue 

#             if check[xx][yy] == False:
#                 if l <= abs(a[x][y] - a[xx][yy]) <= r:
#                     q.append((xx,yy))
#                     check[xx][yy] = True
#                     move.append((xx,yy))
#                     move_cnt += a[xx][yy]

#     change_cnt = (move_cnt // len(move))

#     for mx,my in move:
#         a[mx][my] = change_cnt

#     return               
    

# result = 0
# while True:
#     check = [[False]*n for _ in range(n)]
#     end_cnt = 0

#     for i in range(n):
#         for j in range(n):
#             if check[i][j] == False:
#                 bfs(i,j)
#                 end_cnt += 1
    
#     if end_cnt == (n*n):
#         break
#     result += 1

# print(result)

# 2.
# from collections import deque 

# n,l,r = map(int,input().split()) 

# graph = [] 
# for _ in range(n):
#     graph.append(list(map(int,input().split())))

# dx = [-1,0,1,0]
# dy = [0,-1,0,1]

# result = 0

# # 특정 위치에서 출발하여 모든 연합을 체크한 뒤에 데이터 갱신
# def process(x, y, index): 
#     #(x,y)의 위치와 연결된 나라(연합) 정보를 담는 리스트 
#     united = []
#     united.append((x,y))
#     q = deque() 
#     q.append((x,y))
#     union[x][y] = index 
#     summary = graph[x][y]
#     count = 1 

#     while q:
#         x,y = q.popleft()
#         for i in range(4):
#             nx = x+dx[i]
#             ny = y+dy[i]
#             if 0 <= nx < n and 0 <= ny < n and union[nx][ny] == -1:
#                 if l <= abs(graph[nx][ny] - graph[x][y]) <= r:
#                     q.append((nx,ny))
#                     union[nx][ny] = index 
#                     summary += graph[nx][ny]
#                     count += 1 
#                     united.append((nx,ny))
    
#     for i, j in united:
#         graph[i][j] = summary // count 
    
#     return count 

# total_count = 0

# while True:
#     union = [[-1]*n for _ in range(n)]
#     index = 0 
#     for i in range(n):
#         for j in range(n):
#             if union[i][j] == -1:
#                 process(i,j,index)
#                 index+=1 
    
#     if index == n*n:
#         break 

#     total_count += 1

# print(total_count)


#14503 
# from collections import deque 

# dx, dy = [-1,0,1,0],[0,1,0,-1]
# n,m = map(int,input().split())
# r,c,d = map(int,input().split()) #d - 0:북쪽, 1:동쪽, 2:남쪽, 3:서쪽 
# board = [list(map(int,input().split())) for _ in range(n)]

# def change_d(d):
#     if d == 0:
#         return 3
#     elif d == 1:
#         return 0
#     elif d == 2:
#         return 1
#     elif d == 3:
#         return 2

# def back(d):
#     if d == 0:
#         return 2
#     elif d == 1:
#         return 3
#     elif d == 2:
#         return 0
#     elif d == 3:
#         return 1

# def bfs(sx,sy,d):
#     q = deque()
#     q.append((sx,sy,d))
#     board[sx][sy] = 2
#     result = 1

#     while q:
#         x, y, d = q.popleft() 
#         tmp_d = d

#         for i in range(4):
#             tmp_d = change_d(tmp_d)
#             xx, yy = x+dx[tmp_d], y+dy[tmp_d] #b

#             if xx < 0 or xx > n-1 or yy < 0 or yy > m-1:
#                 continue 
        
#             #a
#             elif board[xx][yy] == 0:
#                 result += 1
#                 board[xx][yy] = 2  # 1번 
#                 q.append((xx,yy,tmp_d))
#                 break 
            
#             #c
#             elif i == 3:
#                 back_d = back(d)
#                 xx_d, yy_d = x+dx[back_d], y+dy[back_d]
#                 q.append((xx_d, yy_d, d))

#                 if board[xx_d][yy_d] == 1:
#                     return result

# print(bfs(r,c,d))

#15685
#한 변이 10인 정사각형 일때만 가능하다.
# board = [[0]*101 for _ in range(101)]

# for _ in range(int(input())):
#     x,y = map(int,input().split())
    
#     for i in range(x, x+10):
#         for j in range(y, y+10):
#             board[i][j] = 1 

# result = 0
# for ii in board:
#     for jj in ii:
#         if jj == 1:
#             result+=1 

# print(result)


#2947
# num = list(map(int,input().split()))

# for i in range(1,len(num)):
#     for j in range(len(num)-i):
#         if num[j] > num[j+1]:
#             num[j], num[j+1] = num[j+1], num[j]
#             print(*num)


#1004
# for _ in range(int(input())):
#     sx,sy,ex,ey = map(int,input().split())
#     planet_num = int(input())
#     result = 0
#     for i in range(planet_num):
#         cx,cy,r = map(int,input().split())
#         start_center = ((sx-cx)**2 + (sy-cy)**2)**0.5
#         end_center = ((ex-cx)**2 + (ey-cy)**2)**0.5 
#         if (start_center < r) and (end_center > r):
#             result += 1
#         elif (start_center > r) and (end_center < r):
#             result+= 1

#     print(result)

# 14891
# 1.
# gear = [list(map(int,input())) for _ in range(4)]
# k = int(input())

# def clock(g):
#     #시계 방향
#     tmp = g[-1]
#     for i in range(len(g)-1,0,-1):
#         g[i] = g[i-1]
#     g[0] = tmp

# def anticlock(g):
#     #반시계 방향
#     tmp = g[0]
#     for i in range(len(g)-1):
#         g[i] = g[i+1]
#     g[-1] = tmp 

# def rotation_gear(rg,di):
#     check = [0]*4 
#     check[rg-1] = di
#     #마주한 극 확인 

#     #start
#     if rg-1 == 0:
#         for i in range(rg-1,3):
#             if gear[i][2] != gear[i+1][6]:
#                 check[i+1] = check[i]*(-1)
#             else:
#                 break 
#     elif rg-1 == 1:
#         if gear[0][2] != gear[rg-1][6]:
#             check[0] = check[rg-1] *(-1)

#         for i in range(rg-1,3):
#             if gear[i][2] != gear[i+1][6]:
#                 check[i+1] = check[i]*(-1)
#             else:
#                 break 

#     elif rg-1 == 2:
#         if gear[-1][6] != gear[rg-1][2]:
#             check[-1] = check[rg-1] * (-1)
        
#         for i in range(rg-1,0,-1):
#             if gear[i][6] != gear[i-1][2]:
#                 check[i-1] = check[i]*(-1)
#             else:
#                 break
            
#     elif rg-1 == 3:
#         for i in range(rg-1,0,-1):
#             if gear[i][6] != gear[i-1][2]:
#                 check[i-1] = check[i]*(-1)
#             else:
#                 break
    
#     for g in range(4):
#         if check[g] == -1: #반시계
#             anticlock(gear[g])
#         elif check[g] == 1:
#             clock(gear[g])


# for _ in range(k):
#     ro_gear, d = map(int,input().split())
#     rotation_gear(ro_gear, d)


# result = 0 
# for g in range(4):
#     result+= (2**g) * gear[g][0]
# print(result)

# 2.
# import sys 
# from collections import deque 

# def check_right(start, dirs):
#     if start > 4 or gears[start-1][2] == gears[start][6]:
#         return 

#     if gears[start-1][2] != gears[start][6]:
#         check_right(start+1, -dirs)
#         gears[start].rotate(dirs)

# def check_left(start, dirs):
#     if start < 1 or gears[start][2] == gears[start+1][6]:
#         return 

#     if gears[start+1][6] != gears[start][2]:
#         check_left(start-1, -dirs)
#         gears[start].rotate(dirs)


# gears = {} 

# for i in range(1,5):
#     gears[i] = deque(list(map(int,list(input()))))

# n = int(input())

# for _ in range(n):
#     num, dirs = map(int,input().split())

#     check_right(num+1, -dirs)
#     check_left(num-1, -dirs)
#     gears[num].rotate(dirs)

# result = 0 
# for i in range(4):
#     result += (2**i) * gears[i+1][0]

# print(result)


#1475 
# n = input()
# number = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0}

# for i in n:
#     if i in ['6','9']:
#         number['6']+=1
#     else:
#         number[i] += 1 

# number['6'] = (number['6']+1)//2 

# print(max(number.values()))


#2824 
# import sys 
# sys.setrecursionlimit(100000)

# n = int(input())
# nums = map(int,input().split())
# m = int(input())
# mums = map(int,input().split())

# def multi(x):
#     result = 1 

#     for i in x:
#         if i == 0:
#             return 0
#         result*=i 
#     return result 

# def gcd(x, y):
#     if y == 0:
#         return 0
#     if x % y == 0:
#         return y 
#     return gcd(y,x%y)

# num = multi(nums)
# mum = multi(mums)

# if num > mum:
#     answer = str(gcd(num, mum))
#     if len(answer) > 9:
#         print(answer[len(answer)-9:])
#     else:
#         print(answer)
# else:
#     answer = str(gcd(mum, num))
#     if len(answer) > 9:
#         print(answer[len(answer)-9:])
#     else:
#         print(answer)


# 4307 
# for _ in range(int(input())):
#     bar, ant_n = map(int,input().split())
#     max_time = 0
#     min_time = 0

#     for _ in range(ant_n):
#         ant_s = int(input())

#         check_min = min(bar-ant_s, ant_s)
#         check_max = max(bar-ant_s, ant_s)
#         max_time = max(max_time, check_max)
#         min_time = max(min_time, check_min)

#     print(min_time, max_time)


# 15683
# 재귀
# from collections import deque 
# import copy 
# import sys 

# input = sys.stdin.readline 

# dx = [1, -1, 0, 0]
# dy = [0, 0, 1, -1]  #북, 남, 동, 서
# direction = [[], [[0], [1], [2], [3]], [[0, 1], [2, 3]], [[3, 0], [0, 2], [2, 1], [1, 3]], [[1, 3, 0], [3, 0, 2], [0, 2, 1], [2, 1, 3]], [[0, 1, 2, 3]]]
# n,m = map(int,input().split())
# board = [list(map(int,input().split())) for _ in range(n)]
# q = deque() 
# cctv_cnt = 0
# result = 1e9

# def check_cctv(b, sx,sy, di):
#     for d in di:
#         xx, yy = sx,sy
#         while True:
#             xx += dx[d]
#             yy += dy[d]

#             if 0 <= xx < n and 0 <= yy < m:
#                 if b[xx][yy] == 6:
#                     break 

#                 elif b[xx][yy] == 0:
#                     b[xx][yy] = '#'
#             else:
#                 break

# def dfs(b, cnt):
#     global result 
    
#     tmp_d = copy.deepcopy(b)
#     if cnt == cctv_cnt:
#         zero_num = 0
#         for point in tmp_d:
#             zero_num += point.count(0)

#         result = min(result, zero_num)
#         return 
    
#     x, y, cctv = q[cnt]
#     for direct in direction[cctv]:
#         check_cctv(tmp_d, x, y, direct)
#         dfs(tmp_d, cnt+1)
#         tmp_d = copy.deepcopy(b)


# for row in range(n):
#     for col in range(m):
#         if board[row][col] not in [0, 6]:
#             q.append((row,col,board[row][col]))
#             cctv_cnt += 1

# dfs(board, 0)
# print(result)


# 2960
# n, k = map(int,input().split())

# def sol():
#     check = [True]*(n+1)
#     cnt = 0
#     for num in range(2, n+1):
#         for i in range(num, n+1, num):
#             if check[i] == True:
#                 check[i] = False 
#                 cnt += 1
#                 if cnt == k:
#                     print(i)
#                     return  
#     return 
# sol()


# 2504 
# 스택에 넣는 방법 
# 맨 가운데 것은 어째든 마주보게 되어 있다.

# sign = input()

# def check(a):
#     check_s = [] 
#     for s in a:
#         if s == '(' or s == '[':
#             check_s.append(s)

#         elif s == ')' and check_s:
#             if check_s[-1] == '(':
#                 check_s.pop() 
#             else:
#                 return False 

#         elif s == ']' and check_s:
#             if check_s[-1] == '[':
#                 check_s.pop() 
#             else:
#                 return False
    
#     if check_s:
#         return False 
#     else:
#         return True


# def sol(a):
#     sol_s = []

#     for s in a:
#         if s == '(' or s == '[':
#             sol_s.append(s)

#         elif s == ')':
#             if sol_s[-1] == '(':
#                 sol_s.pop()
#                 sol_s.append(2)
#             else:
#                 num = 0
#                 for j in range(len(sol_s)-1,-1,-1):
#                     if sol_s[j] == '(':
#                         sol_s[-1] = num * 2 
#                         break 
#                     else:
#                         num += sol_s[j]
#                         sol_s.pop()
                    
#         elif s == ']':
#             if sol_s[-1] == '[':
#                 sol_s.pop()
#                 sol_s.append(3)
#             else:
#                 num = 0
#                 for j in range(len(sol_s)-1,-1,-1):
#                     if sol_s[j] == '[':
#                         sol_s[-1] = num * 3 
#                         break 
#                     else:
#                         num += sol_s[j]
#                         sol_s.pop() 
#     return sum(sol_s)
                    
# if check(sign):
#     print(sol(sign))
# else:
#     print(0)

        