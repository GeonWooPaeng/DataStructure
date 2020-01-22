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

def dijkstra(start):
    heap_data = []
    heapq.heappush(heap_data,(0,start))
    distance[start] = 0 #최단 거리 기록하는 배열
    while heap_data:
        dist,now = heapq.heappop(heap_data)
        if distance[now] < dist:
            continue 
        for i in adj[now]:
            cost = dist + i[1]
            if distance[i[0]] > cost:
                distance[i[0]] = cost 
                heapq.heappush(heap_data,(cost,i[0]))