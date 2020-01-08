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
#             print(result)
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



    