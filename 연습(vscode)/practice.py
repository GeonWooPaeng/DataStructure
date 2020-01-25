def bubble(data):
    for index in range(len(data)-1):
        swap = False
        for index2 in range(len(data) - index -1):
            if data[index2] > data[index +1]:
                data[index2], data[index2+1] = data[index2+1],data[index2]
                swap = True 
        if swap == False:
            break 


def selection(data):
    for stand in range(len(data)-1):
        lowest = stand 
        for num in range(stand,len(data)):
            if data[lowest] > data[num]:
                lowest = num 
        data[stand],data[lowest] = data[lowest],data[stand]
    return data



def insertion(data):
    for index in range(len(data)-1):
        for index2 in range(index+1,0,-1):
            if data[index2] < data[index2 -1]:
                data[index2], data[index2-1] = data[index2 -1], data[index2]
            else:
                break 
        return data


def function(data):
    if data > 1:
        return function(data-1)
    else:
        return data 

def function(data):
    if data <= 1:
        return data
    return function(data-1)

def qsort(data):
    if data <= 1:
        return data

    left, right = list(), list() 
    pivot = data[0]

    for index in range(1,len(data)):
        if data[index] < pivot:
            left.append(data[index])
        else:
            right.append(data[index])

    return qsort(left) + [pivot] + qsort(right)


def split(data):
    if len(data) <= 1:
        return data

    medium = len(data)//2

    left = split(data[:medium])
    right = split(data[medium:])

    return merge(left,right)

def merge(left,right):
    merged = list()
    left_point,right_point = 0,0 

    #case1 - left/right 둘다 있을때
    while len(left) > left_point and len(right) > right_point:
        if left[lp] > right[right_point]:
            merged.append(right[right_point])
            right_point += 1
        else:
            merged.append(left[left_point])
            left_point += 1 
    #case2 - left 만 있을 때 
    while len(left) > left_point:
        merged.append(left[left_point])
        right_point += 1

    #case3 - right 만 있을 때 
    while len(right) > right_point:
        merged.append(right[right_point])
        right_point += 1
    
    return merged
    

def binary(data,search):
    data.sort() 
    print(data)
    if len(data) == 1 and search == data[0]:
        return True 
    elif len(data) == 1 and search != data[0]:
        return False 
    elif len(data) == 0:
        return False 

    medium = len(data) //2
    if search == data[medium]:
        return True
    else:
        if search > data[medium]:
            return binary(data[medium+1:],search)
        else:
            return binary(data[:medium],search)

def bfs(graph, start_node):
    visited = list() 
    need_visit = list() 

    need_visit.append(start_node)
    count = 0 

    while need_visit:
        count += 1
        node = need_visit.pop(0)
        if node not in visited:
            visited.append(node)
            need_visit.extend(graph[node])
    
    print(count)
    return visited


def dfs(graph,start_node):
    visited = list() #queue
    need_visit = list() #stack 

    need_visit.append(start_node)

    while need_visit:
        node = need_visit.pop()
        if node not in visited:
            visited.append(node)
            need_visit.extend(graph[node])
    
    return visited

def pre_order(tree):
    if tree == None:
        return 
    
    print(tree.data)

    pre_order(tree.left_node)
    pre_order(tree.right_node)

def in_order(tree):
    if tree == None:
        return 

    in_order(tree.left_node)

    print(tree.data)

    in_order(tree.right_node)

def post_order(node):
    if tree == None:
        return 
    
    post_order(tree.left_node)
    post_order(tree.right_node)

    print(tree.data)

adj_list = [[] for i in range(n+1)]
indegree = [0] * (n+1)
queue = []
result = []

def addEdge(x,y):
    adj_list[x].append(y) # x가 y로 이어진다.
    indegree[y] += 1

def Top_Sort():
    for i in range(1, n+1):
        if indegree[i] == 0:
            queue.append(i)
    
    for _ in range(n):
        if not queue:
            print("Cycle")
            return False 

        cur = queue.pop(0)
        result.append()

        for adj in adj_list[cur]:
            indegree[adj] -= 1 
            if indegree[adj] == 0:
                queue.append(adj)
    
    return True



import heapq

V, E = map(int, input().split())
K = int(input()) #시작점 K
weight = [INF]*(V+1) #가중치 테이블 , INF = 무한

graph = [[] for _ in range(V + 1)]

def Dijkstra(start):
    heap = []
    #가중치 테이블에서 시작 정점에 해당하는 가중치는 0으로 초기화
    weight[start] = 0
    heapq.heappush(heap,(0, start))

    #힙에 원소가 없을 때 까지 반복.
    while heap:
        wei, now = heapq.heappop(heap)

        #현재 테이블과 비교하여 불필요한(더 가중치가 큰) 튜플이면 무시.
        if weight[now] < wei:
            continue

        for w, next_node in graph[now]:
            #현재 정점 까지의 가중치 wei + 현재 정점에서 다음 정점(next_node)까지의 가중치 W
            # = 다음 노드까지의 가중치(next_wei)
            next_wei = w + wei
            #다음 노드까지의 가중치(next_wei)가 현재 기록된 값 보다 작으면 조건 성립.
            if next_wei < weight[next_node]:
                #계산했던 next_wei를 가중치 테이블에 업데이트.
                weight[next_node] = next_wei
                #다음 점 까지의 가증치와 다음 점에 대한 정보를 튜플로 묶어 최소 힙에 삽입.
                heapq.heappush(heap,(next_wei,next_node))

#초기화
for _ in range(E):
    u, v, w = map(int, input().split())
    #(가중치, 목적지 노드) 형태로 저장
    graph[u].append((w, v))

