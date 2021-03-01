# 1697번: 숨바꼭질
#### 수빈이는 현재 점 N(0 ≤ N ≤ 100,000)에 있고, 동생은 점 K(0 ≤ K ≤ 100,000)에 있다. 수빈이는 걷거나 순간이동을 할 수 있다. 만약, 수빈이의 위치가 X일 때 걷는다면 1초 후에 X-1 또는 X+1로 이동하게 된다. 순간이동을 하는 경우에는 1초 후에 2*X의 위치로 이동하게 된다.
#### 수빈이와 동생의 위치가 주어졌을 때, 수빈이가 동생을 찾을 수 있는 가장 빠른 시간이 몇 초 후인지 구하는 프로그램을 작성하시오.



```python
## 1697번: 숨바꼭질
#### 수빈이는 현재 점 N(0 ≤ N ≤ 100,000)에 있고, 동생은 점 K(0 ≤ K ≤ 100,000)에 있다. 수빈이는 걷거나 순간이동을 할 수 있다. 만약, 수빈이의 위치가 X일 때 걷는다면 1초 후에 X-1 또는 X+1로 이동하게 된다. 순간이동을 하는 경우에는 1초 후에 2*X의 위치로 이동하게 된다.
#### 수빈이와 동생의 위치가 주어졌을 때, 수빈이가 동생을 찾을 수 있는 가장 빠른 시간이 몇 초 후인지 구하는 프로그램을 작성하시오.



from collections import deque

n, k = map(int, input().split())

MAX = 400000 # 계산을 통해 도출해야함(2십만 권고) # O(V+E) = 1십만 + 3십만 = 4십만 넣어봄

check = [False] * (MAX+1)
d = [-1]*(MAX+1)
q = deque()

q.append(n)
d[n] = 0 # 문제조건에서 출발점에선 0초이므로 0으로 초기화
check[n] = True

while q:
    now = q.popleft()

    for next in [now-1, now+1, 2*now]:
        if 0 <= next <= MAX and check[next] == False: # next가 아직 방문한 적 없는 위치이고 && 범위 안에 있다면,
            check[next] = True # (1)방문처리
            q.append(next) # (2) 방문할 큐에 추가
            d[next] = d[now] + 1 # (3) 최단거리 업데이트


print(d[k])

```

    5 17
    4


# 13913번: 숨바꼭질 4
#### 이동해야 하는 경로 추가 출력


```python
## 13913번: 숨바꼭질 4
#### 이동해야 하는 경로 추가 출력 (재귀함수로 경로 역추적)


from collections import deque
import sys


n, k = map(int, input().split())

MAX = 400000 # 계산을 통해 도출해야함(2십만 권고) # O(V+E) = 1십만 + 3십만 = 4십만 넣어봄
sys.setrecursionlimit(MAX)


check = [False] * (MAX+1)
d = [-1]*(MAX+1)
q = deque()
via = [-1] * (MAX+1) ## 경로를 담는 배열 추가 !

q.append(n)
d[n] = 0 # 문제조건에서 출발점에선 0초이므로 0으로 초기화
check[n] = True

while q:
    now = q.popleft()

    for next in [now-1, now+1, 2*now]:
        if 0 <= next <= MAX and check[next] == False: # next가 아직 방문한 적 없는 위치이고 && 범위 안에 있다면,
            check[next] = True # (1)방문처리
            q.append(next) # (2) 방문할 큐에 추가
            d[next] = d[now] + 1 # (3) 최단거리 업데이트

            via[next] = now ## (4) 경로 역추적 배열 추가 !


print(d[k]) # 1. 거리 출력

def go(n, k):  # 2. 경로 출력 (재귀함수 이용)
    if n != k:
        go(n, via[k])

    print(k, end=' ')


go(n,k)

```

    5 17
    4
    5 4 8 16 17 

# 14226번: 이모티콘
#### 3가지 연산(1.클립보드에 저장, 2.화면에 붙여넣기, 3.화면에 있는 이모티콘 1개 삭제)만 사용해, S개의 이모티콘을 화면에 만드는데 걸리는 시간의 최솟값을 구하는 프로그램을 작성하시오.
#### 모든 연산은 1초 (이미 화면에 이모티콘 1개를 입력한 상태)



```python
## 14226번: 이모티콘
#### 3가지 연산(1.클립보드에 저장, 2.화면에 붙여넣기, 3.화면에 있는 이모티콘 1개 삭제)만 사용해, S개의 이모티콘을 화면에 만드는데 걸리는 시간의 최솟값을 구하는 프로그램을 작성하시오.
#### 모든 연산은 1초 (이미 화면에 이모티콘 1개를 입력한 상태)


# 파란간선 BFS 문제 (2차원 dist배열 이용)
# (현재 화면에 있는 이모티콘 개수, 현재 클립보드에 있는 이모티콘 개수) = (s, c)

from collections import deque

n = int(input())

dist = [[-1]*(n+1) for _ in range(n+1)]
q = deque()
q.append((1, 0)) # 출발점
dist[1][0] = 0 # (s,c) = (1,0)에서 출발, 시작은 0초

while q:
    s, c = q.popleft() # 2차원

    if dist[s][s] == -1: # 연산 1: (s, c) -> (s, s)
        dist[s][s] = dist[s][c] + 1
        q.append((s,s))

    if s+c <= n and dist[s+c][c] == -1: # 연산 2: (s, c) -> (s+c, c)
        dist[s+c][c] = dist[s][c] + 1
        q.append((s+c,c))


    if s-1 >= 0 and dist[s-1][c] == -1: # 연산 3: (s, c) -> (s-1, c)
        dist[s-1][c] = dist[s][c] + 1
        q.append((s-1,c))

#print(dist)

# 답 출력
ans = -1
for i in range(n+1):
    if dist[n][i] != -1:
        if ans == -1 or ans > dist[n][i]:
            ans = dist[n][i] # 최소값으로 교체

print(ans)

```

    2
    [[1, 2, 5], [0, 1, 4], [-1, 2, 3]]
    2


# 13549번: 숨바꼭질 3
#### 수빈이는 동생과 숨바꼭질을 하고 있다. 수빈이는 현재 점 N(0 ≤ N ≤ 100,000)에 있고, 동생은 점 K(0 ≤ K ≤ 100,000)에 있다. 수빈이는 걷거나 순간이동을 할 수 있다. 만약, 수빈이의 위치가 X일 때 걷는다면 1초 후에 X-1 또는 X+1로 이동하게 된다. 순간이동을 하는 경우에는 0초 후에 2*X의 위치로 이동하게 된다.
#### 수빈이와 동생의 위치가 주어졌을 때, 수빈이가 동생을 찾을 수 있는 가장 빠른 시간이 몇 초 후인지 구하는 프로그램을 작성하시오.



```python
## 13549번: 숨바꼭질 3
#### 수빈이는 동생과 숨바꼭질을 하고 있다. 수빈이는 현재 점 N(0 ≤ N ≤ 100,000)에 있고, 동생은 점 K(0 ≤ K ≤ 100,000)에 있다. 수빈이는 걷거나 순간이동을 할 수 있다. 만약, 수빈이의 위치가 X일 때 걷는다면 1초 후에 X-1 또는 X+1로 이동하게 된다. 순간이동을 하는 경우에는 0초 후에 2*X의 위치로 이동하게 된다.
#### 수빈이와 동생의 위치가 주어졌을 때, 수빈이가 동생을 찾을 수 있는 가장 빠른 시간이 몇 초 후인지 구하는 프로그램을 작성하시오.

from collections import deque

MAX = 2000000

check = [False]*MAX
dist = [-1]*MAX

n, m = map(int, input().split())
check[n] = True # 출발점: 수빈이의 위치 n
dist[n] = 0 # 출발시각: 0초에서 시작

q = deque() # (1)현재 큐 선언
next_queue = deque() # (2)다음 큐 선언
q.append(n)


while q:
    now = q.popleft()

    # (1) 시간이 0초 걸리는 연산 -> "현재 큐"에 추가
    if now*2 < MAX and check[now*2] == False: # 연산을 통해 이동할 위치가 범위 안에 있고 & 아직 방문하지 않았다면,
        q.append(now*2)
        check[now*2] = True
        dist[now*2] = dist[now] # 현재시간 그대로


    # (2) 시간이 1초 걸리는 연산 -> "다음 큐"에 추가
    if now+1 < MAX and check[now+1] == False: # 연산을 통해 이동할 위치가 범위 안에 있고 & 아직 방문하지 않았다면,
        next_queue.append(now+1)
        check[now+1] = True
        dist[now+1] = dist[now] + 1 # 현재시간에 1초 더해서 업데이트

    if now-1 >= 0 and check[now-1] == False: # 연산을 통해 이동할 위치가 범위 안에 있고 & 아직 방문하지 않았다면,
        next_queue.append(now-1)
        check[now-1] = True
        dist[now-1] = dist[now] + 1 # 현재시간에 1초 더해서 업데이트


    ## 현재 큐가 남아있지 않다면 -> 다음 큐로 넘어가기
    if not q:
        q = next_queue # 다음 큐가 "현재 큐"가 되고,
        next_queue = deque() # 새로운 "다음 큐" 생성


print(dist[m])

```

    5 17
    2


- 덱(양방향 링크드리스트)를 활용한 풀이법


```python
# 덱 사용
# 연산 0초는 덱의 앞쪽으로 삽입, 연산 1초는 덱의 뒷쪽으로 삽입


from collections import deque

MAX = 2000000

check = [False]*MAX
dist = [-1]*MAX

n, m = map(int, input().split())

check[n] = True
dist[n] = 0

q = deque()
q.append(n)


while q:
    now = q.popleft()

    # (1) 시간이 0초 걸리는 연산 -> "현재 큐"에 추가
    if now*2 < MAX and check[now*2] == False: # 연산을 통해 이동할 위치가 범위 안에 있고 & 아직 방문하지 않았다면,
        q.appendleft(now*2) ## 덱의 앞에 삽입
        check[now*2] = True
        dist[now*2] = dist[now] # 현재시간 그대로


    # (2) 시간이 1초 걸리는 연산 -> "다음 큐"에 추가
    if now+1 < MAX and check[now+1] == False: # 연산을 통해 이동할 위치가 범위 안에 있고 & 아직 방문하지 않았다면,
        q.append(now+1) ## 덱의 뒤에 삽입
        check[now+1] = True
        dist[now+1] = dist[now] + 1 # 현재시간에 1초 더해서 업데이트

    if now-1 >= 0 and check[now-1] == False: # 연산을 통해 이동할 위치가 범위 안에 있고 & 아직 방문하지 않았다면,
        q.append(now-1) ## 덱의 뒤에 삽입
        check[now-1] = True
        dist[now-1] = dist[now] + 1 # 현재시간에 1초 더해서 업데이트


print(dist[m]) 
```

    5 17
    2


# 1261번: 알고스팟
#### 이동할 수 있는 방은 (x+1, y), (x, y+1), (x-1, y), (x, y-1) 이다. 단, 미로의 밖으로 이동 할 수는 없다.
#### 현재 (1, 1)에 있는 알고스팟 운영진이 (N, M)으로 이동하려면 벽을 최소 몇 개 부수어야 하는지 구하는 프로그램을 작성하시오.



```python
## 1261번: 알고스팟
#### 이동할 수 있는 방은 (x+1, y), (x, y+1), (x-1, y), (x, y-1) 이다. 단, 미로의 밖으로 이동 할 수는 없다.
#### 현재 (1, 1)에 있는 알고스팟 운영진이 (N, M)으로 이동하려면 벽을 최소 몇 개 부수어야 하는지 구하는 프로그램을 작성하시오.

## BFS 탐색을 벽을 부순 횟수에 따라서 나누어서 수행해야 한다 !
# 빈방: 0, 벽: 1
# 가중치 1) 빈방 -> 빈방 : 가중치 "0"
# 가중치 2) 빈방 -> 벽 : 가중치 "1"


from collections import deque


m, n = map(int, input().split()) # 가로, 세로

a = [list(map(int, list(input()))) for _ in range(n)]
#print(a)


dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

###check = [[False]*m for _ in range(n)]
dist = [[-1]*m for _ in range(n)]

###check[0] = True # 출발점
dist[0][0] = 0 # 벽 부수는 초기횟수: 0번

q = deque() # (1)현재 큐 선언
next_queue = deque() # (2)다음 큐 선언
q.append((0,0))



while q:
    x, y = q.popleft()

    for k in range(4):
        nx, ny = x+dx[k], y+dy[k]

        if 0<= nx < n and 0<= ny < m: ## 세로, 가로
            if dist[nx][ny] == -1: # next 좌표가 범위 내 있고 && 아직 방문한 적 없다면,
                if a[nx][ny] == 0: # (1)빈방으로 갈 때 (가중치 0)
                    q.append((nx, ny))
                    ###check[nx][ny] = True
                    dist[nx][ny] = dist[x][y] # 현재 횟수 그대로

                else: # (2)벽으로 갈 때 (가중치 1)
                    next_queue.append((nx, ny))
                    ###check[nx][ny] = True
                    dist[nx][ny] = dist[x][y] +1 # 현재 횟수에 1 추가


    ## 현재 큐가 남아있지 않다면 -> 다음 큐로 넘어가기
    if not q:
        q = next_queue # 다음 큐가 "현재 큐"가 되고,
        next_queue = deque() # 새로운 "다음 큐" 생성


print(dist[n-1][m-1])



```

    3 3
    011
    111
    110
    3


- 덱(양방향 링크드리스트)를 활용한 풀이법


```python
# 덱으로 구현

from collections import deque


m, n = map(int, input().split()) # 가로, 세로

a = [list(map(int, list(input()))) for _ in range(n)]
#print(a)


dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

###check = [[False]*m for _ in range(n)]
dist = [[-1]*m for _ in range(n)]

###check[0] = True # 출발점
dist[0][0] = 0 # 벽 부수는 초기횟수: 0번

q = deque() # 덱 선언
q.append((0,0))



while q:
    x, y = q.popleft()

    for k in range(4):
        nx, ny = x+dx[k], y+dy[k]

        if 0<= nx < n and 0<= ny < m: ## 세로, 가로
            if dist[nx][ny] == -1: # next 좌표가 범위 내 있고 && 아직 방문한 적 없다면,
                if a[nx][ny] == 0: # (1)빈방으로 갈 때 (가중치 0)
                    q.appendleft((nx, ny)) ## 덱 앞에 삽입
                    ###check[nx][ny] = True
                    dist[nx][ny] = dist[x][y] # 현재 횟수 그대로

                else: # (2)벽으로 갈 때 (가중치 1)
                    q.append((nx, ny)) ## 덱 뒤에 삽입
                    ###check[nx][ny] = True
                    dist[nx][ny] = dist[x][y] +1 # 현재 횟수에 1회 추가



print(dist[n-1][m-1])

```

    3 3
    011
    111
    110
    3

