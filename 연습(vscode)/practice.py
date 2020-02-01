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

def Dijkstra(start, end):
    heap = []
    #가중치 테이블에서 시작 정점에 해당하는 가중치는 0으로 초기화
    weight[start] = (0,start)
    heapq.heappush(heap,(0, start))

    #힙에 원소가 없을 때 까지 반복.
    while heap:
        wei, now = heapq.heappop(heap)

        #현재 테이블과 비교하여 불필요한(더 가중치가 큰) 튜플이면 무시.
        if weight[now][0] < wei:
            continue

        for w, next_node in graph[now]:
            #현재 정점 까지의 가중치 wei + 현재 정점에서 다음 정점(next_node)까지의 가중치 W
            # = 다음 노드까지의 가중치(next_wei)
            next_wei = w + wei
            #다음 노드까지의 가중치(next_wei)가 현재 기록된 값 보다 작으면 조건 성립.
            if next_wei < weight[next_node][0]:
                #계산했던 next_wei를 가중치 테이블에 업데이트.
                weight[next_node] = (next_wei,now)
                #다음 점 까지의 가증치와 다음 점에 대한 정보를 튜플로 묶어 최소 힙에 삽입.
                heapq.heappush(heap,(next_wei,next_node))

    path = end
    path_output = [] #stack
    while weight[path][1] != start:
        path_output.append(weight[path][1])
        path = weight[path][1]
    
    path_output.append(start)
    print(path_output)
    return weight

#초기화
for _ in range(E):
    u, v, w = map(int, input().split())
    #(가중치, 목적지 노드) 형태로 저장
    graph[u].append((w, v))


parent = dict() # 각각 node의 부모 node 저장
rank = dict() # 각각 node의 rank 값

def make_set(node):  #각각 원소 node의 초기화
    parent[node] = node #node가 1개(root가 자기자신)
    rank[node] = 0

def find(node):  #각 node의 root노드 찾기
    if parent[node] != node:
        parent[node] = find(parent[node])
    return parent[node]

def union(node_v,node_u): #노드 연결
    #union by rank
    #각각 root node 알기
    root1 = find(node_v)
    root2 = find(node_u)

    #rank 알아내기
    if rank[root1] > rank[root2]:
        parent[root2] = root1 
    else:
        parent[root1] = root2 

        if rank[root1] == rank[root2]:
            rank[root2] += 1  #rank 올려주기(아무거나 올려도 상관 없다.)

def kruskal(graph):
    mst = list() #cycle이 없으면 간선을 넣어 주는 곳, 이곳에 들어온 합은 신장트리 

    # 1. 초기화
    for node in graph['vertices']:
        make_set(node)

    # 2. 간선 weight 기반 sorting
    edges = graph['edges']
    edges.sort()

    # 3. 간선 연결(사이클 없는)
    for edge in edges: #간선 꺼내기
        wei, node_v , node_u = edge 

        #cycle 파악 후 합치기
        if parent[node_v] != parent[node_u]:  #같지 않으면 cycle 없다는 것
            union(node_v,node_u) #node 합치기
            mst.append(edge)

    return mst 



from collections import defaultdict
from heapq import*

def prim(start_node,edges):
    mst = list() #최소신장트리 간선 list
    
    adjacent_edges = defaultdict(list) #dictlist만들어 준다. key -> 각각node, value -> 간선 정보 list 
    #특정 node의 정보 각각 정리
    for weight,n1,n2 in edges:
        adjacent_edges[n1].append((weight,n1,n2))
        adjacent_edges[n2].append((weight,n2,n1))
    
    connected_nodes = set(start_node) #연결된 node 집합
    
    candidate_edge_list = adjacent_edges[start_node] #연결된 간선 list
    
    heapify(candidate_edge_list) #연결된 간선list heap구조로 바꿈 
    
    while candidate_edge_list: # 더이상 간선 list에 간선 없을 떄 까지 반복 
        weight,n1,n2 = heappop(candidate_edge_list) # 간선 list에서 간선 weight가 가장 작은 값 추출
        
        #인접 node가 연결된 node 잡합에 있는지 확인
        if n2 not in connected_nodes: #Cycle이 없으면
            connected_nodes.add(n2) #연결된 node집합에 넣는다.
            mst.append((weight,n1,n2))
            
            # 인접 node의 간선 정보를 간선list에 넣는곳
            for edge in adjacent_edges[n2]: #인접 node의 정보 가져오기
                if edge[2] not in connected_nodes: #인전 node가 연결된 node에 없을 시
                    heappush(candidate_edge_list,edge) #연결된 간선list에 간선 정보 넣는다.
    
    return mst
        

from heapdict import heapdict #안에 있는 최소 데이터가 root가 되게 한다.

def prim(graph,start):
    mst = list()  #최소 간선 list 저장하는 곳
    keys = heapdict() # node별 key값을 가진 (heap 구조)
    pi = dict() # node가 변했을 때 영향을 받은 간선을 저장하는 곳
    total_weigth = 0  # 전체 간선의 total을 저장하는 곳
    
    for node in graph.keys():
        keys[node] = float('inf') # 무한대로 만듬 
        pi[node] = None  
    
    # 현재 선택한 node
    keys[start] = 0  
    pi[start] = start 
    
    
    while keys:
        current_node, current_key = keys.popitem()
        mst.append([pi[current_node], current_node,current_key]) # 선택된 최소 간선 mst에 넣기
        total_weight += current_key
        
        for adjacent, weight in mygraph[current_node].items():
            if adjacent in keys and weight < keys[adjacent]: #인접된 node가 선택되었는지 확인, 값 비교하기 
                keys[adjacent] = weight 
                pi[adjacent] = current_node
    
    return mst, total_weight
   

def is_available(candidate, current_col):# 조건 만족하는 지 확인하는 함수(수직,대각선 check)
    current_row = len(candidate) #행 찾기
    for queen_row in range(current_row):
        if candidate[queen_row] == current_col or abs(candidate[queen_row]- current_col) == current_row - queen_row: 
            return False 
    return True

    
def DFS(N, current_row, current_candidate, final_result): # current_row - 다음행을 기억하기 위함, current_candidate - 지금 까지 배치된 queen의 위치 정보, final_result - 지금까지 배치 인자
    if current_row == N: #배치가 다 끝난 상태
        final_result.append(current_candidate[:])
        return 
    
    for candidate_col in range(N): #N개 열이 있고 앞에서 부터 인자를 check하겠다.
        if is_available(current_candidate, candidate_col): 
            current_candidate.append(candidate_col)
            DFS(N, current_row +1, current_candidate,final_result)
            
            #backtracking
            current_candidate.pop()

            
def solve_n_queens(N): # NxN은 체스판
    final_result = [] #배치도 저장
    DFS(N,0,[],final_result)
    return final_result