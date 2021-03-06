# 12886번: 돌그룹
#### 돌 세개는 그룹으로 나누어져 있으며 각각의 그룹에는 돌이 A, B, C개가 있다. 강호는 모든 그룹에 있는 돌의 개수를 같게 만들려고 한다.
#### 강호는 돌을 단계별로 움직이며, 각 단계는 다음과 같이 이루어져 있다. 크기가 같지 않은 두 그룹을 고른다. 그 다음, 돌의 개수가 작은 쪽을 X, 큰 쪽을 Y라고 정한다. 그 다음, X에 있는 돌의 개수를 X+X개로, Y에 있는 돌의 개수를 Y-X개로 만든다.
#### A, B, C가 주어졌을 때, 강호가 돌을 같은 개수로 만들 수 있으면 1을, 아니면 0을 출력하는 프로그램을 작성하시오.





```python
## 12886번: 돌그룹
#### 돌 세개는 그룹으로 나누어져 있으며 각각의 그룹에는 돌이 A, B, C개가 있다. 강호는 모든 그룹에 있는 돌의 개수를 같게 만들려고 한다.
#### 강호는 돌을 단계별로 움직이며, 각 단계는 다음과 같이 이루어져 있다. 크기가 같지 않은 두 그룹을 고른다. 그 다음, 돌의 개수가 작은 쪽을 X, 큰 쪽을 Y라고 정한다. 그 다음, X에 있는 돌의 개수를 X+X개로, Y에 있는 돌의 개수를 Y-X개로 만든다.
#### A, B, C가 주어졌을 때, 강호가 돌을 같은 개수로 만들 수 있으면 1을, 아니면 0을 출력하는 프로그램을 작성하시오.



# 최소 횟수를 구하는 문제 -> 꼭 BFS
# 가능한지 여부 구하는 문제 -> BFS/DFS 모두 가능

import sys
sys.setrecursionlimit(1500*1500)

check = [[False]*1501 for _ in range(1501)]
x, y, z = map(int, input().split())

sum = x+y+z

def go(x, y):
    if check[x][y] == True: # 이미 방문한 적 있으면
        return # 함수 종료

    check[x][y] = True # 방문 표시
    
    a = [x, y, sum-x-y]
    for i in range(0, 3):
        for j in range(0, 3):
            if a[i] < a[j]:
                b = [x, y, sum-x-y] # b: 임시 저장 배열
                b[i] += a[i] # X+X
                b[j] -= a[i] # Y-X

                go(b[0], b[1])




# 답 출력
if sum % 3 != 0:
    print(0) # (조기종료)

else:
    go(x, y)

    if check[sum//3][sum//3] == True:
        print(1)

    else:
        print(0)
```

    10 15 35
    1
    

# 2206번: 벽 부수고 이동하기
#### N×M의 행렬로 표현되는 맵이 있다. 맵에서 0은 이동할 수 있는 곳을 나타내고, 1은 이동할 수 없는 벽이 있는 곳을 나타낸다. 당신은 (1, 1)에서 (N, M)의 위치까지 이동하려 하는데, 이때 최단 경로로 이동하려 한다. 최단경로는 맵에서 가장 적은 개수의 칸을 지나는 경로를 말하는데, 이때 시작하는 칸과 끝나는 칸도 포함해서 센다.
#### 만약에 이동하는 도중에 한 개의 벽을 부수고 이동하는 것이 좀 더 경로가 짧아진다면, **벽을 한 개 까지 부수고 이동하여도 된다.**
#### 한 칸에서 이동할 수 있는 칸은 상하좌우로 인접한 칸이다. 맵이 주어졌을 때, 최단 경로를 구해 내는 프로그램을 작성하시오.


```python
## 2206번: 벽 부수고 이동하기
#### N×M의 행렬로 표현되는 맵이 있다. 맵에서 0은 이동할 수 있는 곳을 나타내고, 1은 이동할 수 없는 벽이 있는 곳을 나타낸다. 당신은 (1, 1)에서 (N, M)의 위치까지 이동하려 하는데, 이때 최단 경로로 이동하려 한다. 최단경로는 맵에서 가장 적은 개수의 칸을 지나는 경로를 말하는데, 이때 시작하는 칸과 끝나는 칸도 포함해서 센다.
#### 만약에 이동하는 도중에 한 개의 벽을 부수고 이동하는 것이 좀 더 경로가 짧아진다면, "벽을 한 개 까지 부수고 이동하여도 된다."
#### 한 칸에서 이동할 수 있는 칸은 상하좌우로 인접한 칸이다. 맵이 주어졌을 때, 최단 경로를 구해 내는 프로그램을 작성하시오.


# 최단 경로 -> BFS

from collections import deque

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


# 문제 입력
n, m = map(int, input().split())
a = [list(map(int, input())) for _ in range(n)]


# 초기화 및 BFS
dist = [[[0]*2 for j in range(m)] for i in range(n)]
q = deque()
q.append((0, 0, 0)) # (0, 0)좌표부터 시작, 문제에서 시작점과 도착점은 모두 빈칸이라고 하였음.
dist[0][0][0] = 1 # 문제에서 최단경로에 시작하는 칸과 끝나는 칸도 포함해서 센다고 하였음.

while q:
    x, y, c = q.popleft()

    for k in range(4):
        nx = x+dx[k]
        ny = y+dy[k]

        if 0<=nx<n and 0<= ny <m:
            ## 경우 1) 빈칸 -> 빈칸
            if a[nx][ny] == 0 and dist[nx][ny][c] == 0: # 다음 인접 칸이 빈칸에 해당되며 & 아직 방문한 적 없을 때
                dist[nx][ny][c] = dist[x][y][c] +1 # (1)현재 최단 경로에 거리 +1
                q.append((nx, ny, c)) # (2)방문할 큐에 추가

            ## 경우 2) 빈칸 -> 벽
            if c == 0 and a[nx][ny] == 1 and dist[nx][ny][c+1] == 0: # 현재 벽을 부순 적 없었고&& 다음 인접칸이 벽인데 & 아직 방문한 적 없으면
                dist[nx][ny][c+1] = dist[x][y][c] +1 # (1)현재 최단 경로에 거리 +1
                q.append((nx, ny, c+1)) # (2)방문할 큐에 추가


# 답 출력
## 도착점에서 벽 0번 부신 경로와 1번 부신 경로 모두 존재하면 -> 둘 중 최소값 출력
## 하나만 존재하면 -> 그 값 출력
if dist[n-1][m-1][0] != 0 and dist[n-1][m-1][1] != 0:
    print(min(dist[n-1][m-1][0], dist[n-1][m-1][1]))

elif dist[n-1][m-1][0] != 0:
    print(dist[n-1][m-1][0])

elif dist[n-1][m-1][1] != 0:
    print(dist[n-1][m-1][1])

else: # 모두 불가능할 때는 -1 출력
    print(-1)
```

    6 4
    0100
    1110
    1000
    0000
    0111
    0000
    15
    

# 16946번: 벽 부수고 이동하기 4
#### N×M의 행렬로 표현되는 맵이 있다. 맵에서 0은 이동할 수 있는 곳을 나타내고, 1은 이동할 수 없는 벽이 있는 곳을 나타낸다. 한 칸에서 다른 칸으로 이동하려면, 두 칸이 인접해야 한다. 두 칸이 변을 공유할 때, 인접하다고 한다.
#### 각각의 벽에 대해서 다음을 구해보려고 한다. (한 칸에서 이동할 수 있는 칸은 상하좌우로 인접한 칸이다.)
#### 1. 벽을 부수고 이동할 수 있는 곳으로 변경한다.
#### 2. 그 위치에서 이동할 수 있는 칸의 개수를 세어본다.

#### 맵의 형태로 정답을 출력한다. 원래 빈 칸인 곳은 0을 출력하고, 벽인 곳은 이동할 수 있는 칸의 개수를 10으로 나눈 나머지를 출력한다.


```python
## 16946번: 벽 부수고 이동하기 4
#### N×M의 행렬로 표현되는 맵이 있다. 맵에서 0은 이동할 수 있는 곳을 나타내고, 1은 이동할 수 없는 벽이 있는 곳을 나타낸다. 한 칸에서 다른 칸으로 이동하려면, 두 칸이 인접해야 한다. 두 칸이 변을 공유할 때, 인접하다고 한다.
#### 각각의 벽에 대해서 다음을 구해보려고 한다. (한 칸에서 이동할 수 있는 칸은 상하좌우로 인접한 칸이다.)
#### 1. 벽을 부수고 이동할 수 있는 곳으로 변경한다.
#### 2. 그 위치에서 이동할 수 있는 칸의 개수를 세어본다.

#### 맵의 형태로 정답을 출력한다. 원래 빈 칸인 곳은 0을 출력하고, 벽인 곳은 이동할 수 있는 칸의 개수를 10으로 나눈 나머지를 출력한다.



# group[i][j] = (i,j) 좌표의 그룹번호 값
# group_size[k] = k번 그룹의 크기

# group_size의 크기 => 새로운 그룹의 번호


from collections import deque

dx = [0,0,1,-1]
dy = [1,-1,0,0]

n, m = map(int,input().split())
a = [list(map(int,list(input()))) for _ in range(n)]


check = [[False]*m for _ in range(n)] # 방문 여부 체크
group = [[-1]*m for _ in range(n)] # (i,j) 좌표의 그룹번호 값 저장
group_size = [] # group_size[k] = k번 그룹의 크기


def bfs(sx, sy):
    g = len(group_size) # 현재까지 group_size의 크기 => 새로운 그룹의 번호

    # 초기화
    q = deque()
    q.append((sx, sy))
    group[sx][sy] = g
    check[sx][sy] = True

    cnt = 1
    while q:
        x, y = q.popleft()
        for k in range(4):
            nx = x+dx[k]
            ny = y+dy[k]

            if 0<= nx <n and 0<= ny <m: # 다음 인접 좌표가 범위 내 있고
                if check[nx][ny] == False and a[nx][ny] == 0: # 아직 방문한 적 없고 && 빈칸에 해당되면 -> 그룹 매기기
                    check[nx][ny] = True
                    q.append((nx, ny))

                    group[nx][ny] = g ## 현재 그룹 번호 저장
                    cnt += 1 ## 그룹의 크기 1 늘리기


    group_size.append(cnt)



for i in range(n):
    for j in range(m):
        if a[i][j] == 0 and check[i][j] == False: # 빈칸인데 && 아직 방문한 적 없으면
            bfs(i, j) # BFS 함수 실행하여 그룹 매기기 수행



# 답 출력
for i in range(n):
    for j in range(m):
        if a[i][j] == 0:
            print(0, end='')

        else:
            near = set() # near: 현재 벽 부수고 이동할 수 있는 인접 그룹들의 모든 목록 (현재 벽 부수고 이동할 수 있는 인접 그룹 중 겹치는 그룹이 중복될 수 있으므로 set 사용 !)
            for k in range(4):
                nx = i+dx[k]
                ny = j+dy[k]
                if 0<= nx <n and 0<= ny <m:
                    if a[nx][ny] == 0:
                        near.add(group[nx][ny]) # 해당 그룹의 번호 near 목록에 add

            ans = 1 # 벽 부순 후 자기자신 1개부터 시작
            for g in near:
                ans += group_size[g] # g번 그룹에 속하는 빈칸들의 개수 크기 더하기
            print(ans % 10, end='')


    print() # 행마다 출력 줄바꿈
```

    4 5
    11001
    00111
    01010
    10101
    35003
    00632
    06040
    30403
    

# 14442번: 벽 부수고 이동하기 2
#### N×M의 행렬로 표현되는 맵이 있다. 맵에서 0은 이동할 수 있는 곳을 나타내고, 1은 이동할 수 없는 벽이 있는 곳을 나타낸다. 당신은 (1, 1)에서 (N, M)의 위치까지 이동하려 하는데, 이때 최단 경로로 이동하려 한다. 최단경로는 맵에서 가장 적은 개수의 칸을 지나는 경로를 말하는데, 이때 시작하는 칸과 끝나는 칸도 포함해서 센다.
#### 만약에 이동하는 도중에 벽을 부수고 이동하는 것이 좀 더 경로가 짧아진다면, "벽을 K개 까지 부수고 이동하여도 된다."
#### 한 칸에서 이동할 수 있는 칸은 상하좌우로 인접한 칸이다.
#### 맵이 주어졌을 때, 최단 경로를 구해 내는 프로그램을 작성하시오.


```python
## 14442번: 벽 부수고 이동하기 2
#### N×M의 행렬로 표현되는 맵이 있다. 맵에서 0은 이동할 수 있는 곳을 나타내고, 1은 이동할 수 없는 벽이 있는 곳을 나타낸다. 당신은 (1, 1)에서 (N, M)의 위치까지 이동하려 하는데, 이때 최단 경로로 이동하려 한다. 최단경로는 맵에서 가장 적은 개수의 칸을 지나는 경로를 말하는데, 이때 시작하는 칸과 끝나는 칸도 포함해서 센다.
#### 만약에 이동하는 도중에 벽을 부수고 이동하는 것이 좀 더 경로가 짧아진다면, "벽을 K개 까지 부수고 이동하여도 된다."
#### 한 칸에서 이동할 수 있는 칸은 상하좌우로 인접한 칸이다.
#### 맵이 주어졌을 때, 최단 경로를 구해 내는 프로그램을 작성하시오.



# 최단 경로 -> BFS

from collections import deque

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


# 문제 입력
n, m, K= map(int, input().split()) ## K 추가
a = [list(map(int, input())) for _ in range(n)]


# 초기화 및 BFS
dist = [[[0]*(K+1) for j in range(m)] for i in range(n)] ## 변경 ([0]: 안 부숨, [1], [2], ..., [K+1]: 1번~K번까지 부술 때)
q = deque()
q.append((0, 0, 0)) # (0, 0)좌표부터 시작, 문제에서 시작점과 도착점은 모두 빈칸이라고 하였음.
dist[0][0][0] = 1 # 문제에서 최단경로에 시작하는 칸과 끝나는 칸도 포함해서 센다고 하였음.


while q:
    x, y, c = q.popleft()

    for k in range(4):
        nx = x+dx[k]
        ny = y+dy[k]

        if 0<=nx<n and 0<= ny <m:
            ## 경우 1) 빈칸 -> 빈칸
            if a[nx][ny] == 0 and dist[nx][ny][c] == 0: # 다음 인접 칸이 빈칸에 해당되며 & 아직 방문한 적 없을 때
                dist[nx][ny][c] = dist[x][y][c] +1 # (1)현재 최단 경로에 거리 +1 (*벽 부순 횟수(c)는 그대로 넘겨받음 !)
                q.append((nx, ny, c)) # (2)방문할 큐에 추가

            ## 경우 2) 빈칸 -> 벽
            if c+1 < (K+1) and a[nx][ny] == 1 and dist[nx][ny][c+1] == 0: # 아직 벽을 K번 미만 부셨고&& 다음 인접칸이 벽인데 & 아직 방문한 적 없으면
                dist[nx][ny][c+1] = dist[x][y][c] +1 # (1)현재 최단 경로에 거리 +1 (*벽 부순 횟수(c)도 1 증가 !)
                q.append((nx, ny, c+1)) # (2)방문할 큐에 추가


# 답 출력
## 마지막 좌표(N, M)에서, 0번(안 부숨) ~ K+1번(K번째 부숨)까지 중 최소값 출력해야

ans = -1

for i in range(0, K+1):
    if dist[n-1][m-1][i] != 0: # i번 부순 경로가 존재하지 않는 경우는 pass
        if ans == -1:
            ans = dist[n-1][m-1][i]
        elif ans > dist[n-1][m-1][i]:
            ans = dist[n-1][m-1][i]

print(ans)

```

    6 4 1
    0100
    1110
    1000
    0000
    0111
    0000
    15
    


```python
#### 문제에서 주어진 범위: N(1 ≤ N ≤ 1,000), M(1 ≤ M ≤ 1,000), K(1 ≤ K ≤ 10)

## 범위 만큼 미리 정의해놓는 정답


# 최단 경로 -> BFS

from collections import deque
a = [[0]*1000 for _ in range(1000)]
d = [[[0]*11 for i in range(1000)] for j in range(1000)]
dx = [0,0,1,-1]
dy = [1,-1,0,0]
n,m,l = map(int,input().split())
a = []
for i in range(n):
    a.append(list(map(int,list(input()))))
q = deque()
d[0][0][0] = 1
q.append((0,0,0))
while q:
    x,y,z = q.popleft()
    for k in range(4):
        nx,ny = x+dx[k], y+dy[k]
        if nx < 0 or nx >= n or ny < 0 or ny >= m:
            continue
        if a[nx][ny] == 0 and d[nx][ny][z] == 0:
            d[nx][ny][z] = d[x][y][z] + 1
            q.append((nx,ny,z))
        if z+1 <= l and a[nx][ny] == 1 and d[nx][ny][z+1] == 0:
            d[nx][ny][z+1] = d[x][y][z] + 1
            q.append((nx,ny,z+1))
ans = -1
for i in range(l+1):
    if d[n-1][m-1][i] == 0:
        continue
    if ans == -1:
        ans = d[n-1][m-1][i]
    elif ans > d[n-1][m-1][i]:
        ans = d[n-1][m-1][i]
print(ans)
```

    6 4 1
    0100
    1110
    1000
    0000
    0111
    0000
    15
    

# 16933번: 벽 부수고 이동하기 3
#### 이동하지 않고 같은 칸에 머물러있는 경우도 가능하다. 이 경우도 방문한 칸의 개수가 하나 늘어나는 것으로 생각해야 한다.
#### 이번 문제에서는 낮과 밤이 번갈아가면서 등장한다. 가장 처음에 이동할 때는 낮이고, 한 번 이동할 때마다 낮과 밤이 바뀌게 된다. 이동하지 않고 같은 칸에 머무르는 경우에도 낮과 밤이 바뀌게 된다.
#### 만약에 이동하는 도중에 벽을 부수고 이동하는 것이 좀 더 경로가 짧아진다면, 벽을 K개 까지 부수고 이동하여도 된다. 단, 벽은 낮에만 부술 수 있다.


```python
## 16933번: 벽 부수고 이동하기 3
#### 이동하지 않고 같은 칸에 머물러있는 경우도 가능하다. 이 경우도 방문한 칸의 개수가 하나 늘어나는 것으로 생각해야 한다.
#### 이번 문제에서는 낮과 밤이 번갈아가면서 등장한다. 가장 처음에 이동할 때는 낮이고, 한 번 이동할 때마다 낮과 밤이 바뀌게 된다. 이동하지 않고 같은 칸에 머무르는 경우에도 낮과 밤이 바뀌게 된다.
#### 만약에 이동하는 도중에 벽을 부수고 이동하는 것이 좀 더 경로가 짧아진다면, 벽을 K개 까지 부수고 이동하여도 된다. 단, 벽은 낮에만 부술 수 있다.


# 파이썬만 시간초과


# 최단 경로 -> BFS

from collections import deque

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


# 문제 입력
n, m, K= map(int, input().split()) ## K 추가
a = [list(map(int, input())) for _ in range(n)]


# 초기화 및 BFS
dist = [[[[0]*2 for c in range(K+1)] for j in range(m)] for i in range(n)] ## 변경 ([0]: 안 부숨, [1], [2], ..., [K+1]: 1번~K번까지 부술 때)
q = deque()
q.append((0, 0, 0, 0)) # (0, 0)좌표부터 시작, 문제에서 시작점과 도착점은 모두 빈칸이라고 하였음. (낮부터 시작)
dist[0][0][0][0] = 1 # 문제에서 최단경로에 시작하는 칸과 끝나는 칸도 포함해서 센다고 하였음.


while q:
    x, y, c, night = q.popleft()

    for k in range(4):
        nx = x+dx[k]
        ny = y+dy[k]

        if 0<=nx<n and 0<= ny <m:
            ## 경우 1) 빈칸 -> 빈칸
            if a[nx][ny] == 0 and dist[nx][ny][c][1-night] == 0: # 다음 인접 칸이 빈칸에 해당되며 & 아직 방문한 적 없을 때
                dist[nx][ny][c][1-night] = dist[x][y][c][night] +1 # (1)현재 최단 경로에 거리 +1 (*벽 부순 횟수(c)는 그대로 넘겨받음 !)
                q.append((nx, ny, c, 1-night)) # (2)방문할 큐에 추가

            ## 경우 2) 빈칸 -> 벽 (*추가: 벽은 낮에만 부술 수 있다.)
            if c+1 < (K+1) and night == 0 and a[nx][ny] == 1 and dist[nx][ny][c+1][1-night] == 0: # 아직 벽을 K번 미만 부셨고&& 다음 인접칸이 벽인데 & 아직 방문한 적 없으면
                dist[nx][ny][c+1][1-night] = dist[x][y][c][night] +1 # (1)현재 최단 경로에 거리 +1 (*벽 부순 횟수(c)도 1 증가 !)
                q.append((nx, ny, c+1, 1-night)) # (2)방문할 큐에 추가

            ## 경우 3) 이동하지 않고 머무르기 (방문 칸의 개수 하나 늘어나면서 낮밤만 바뀜)
            if dist[x][y][c][1-night] == 0:
                dist[x][y][c][1-night] = dist[x][y][c][night] + 1
                q.append((x, y, c, 1-night))



# 답 출력
## 마지막 좌표(N, M)에서, 0번(안 부숨) ~ K+1번(K번째 부숨)까지 중 최소값 출력해야

ans = -1

for i in range(0, K+1):
    for night in range(0, 2):
        if dist[n-1][m-1][i][night] != 0: # i번 부순 경로가 존재하지 않는 경우는 pass
            if ans == -1:
                ans = dist[n-1][m-1][i][night]
            elif ans > dist[n-1][m-1][i][night]:
                ans = dist[n-1][m-1][i][night]

print(ans)
```

    1 4 1
    0100
    4
    
