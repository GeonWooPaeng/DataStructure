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

10828.







