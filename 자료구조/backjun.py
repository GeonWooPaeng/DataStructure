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

1939.
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