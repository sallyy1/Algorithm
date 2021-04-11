# 4991번: 로봇 청소기
#### 칸은 깨끗한 칸과 더러운 칸으로 나누어져 있으며, 로봇 청소기는 더러운 칸을 방문해서 깨끗한 칸으로 바꿀 수 있다. (로봇 청소기는 가구가 놓여진 칸으로 이동할 수 없다. )
#### 로봇은 한 번 움직일 때, 인접한 칸으로 이동할 수 있다. 또, (로봇은 같은 칸을 여러 번 방문할 수 있다.)
#### 방의 정보가 주어졌을 때, 더러운 칸을 모두 깨끗한 칸으로 만드는데 필요한 이동 횟수의 최솟값을 구하는 프로그램을 작성하시오.



```python
## 4991번: 로봇 청소기
#### 칸은 깨끗한 칸과 더러운 칸으로 나누어져 있으며, 로봇 청소기는 더러운 칸을 방문해서 깨끗한 칸으로 바꿀 수 있다. (로봇 청소기는 가구가 놓여진 칸으로 이동할 수 없다. )
#### 로봇은 한 번 움직일 때, 인접한 칸으로 이동할 수 있다. 또, (로봇은 같은 칸을 여러 번 방문할 수 있다.)
#### 방의 정보가 주어졌을 때, 더러운 칸을 모두 깨끗한 칸으로 만드는데 필요한 이동 횟수의 최솟값을 구하는 프로그램을 작성하시오.



# 내가 시도한 풀이 (실패, 무한루프)
from collections import deque

dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]


## BFS 탐색
def bfs(a, w, h, sx, sy):
    d = [[0]*w for _ in range(h)]
    ##d[sx][sy] = 0 # 시작시 0초

    q = deque()
    q.append((sx, sy))

    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]

            if 0<=nx<h and 0<=ny<w:
                if a[nx][ny] == 'x':
                    continue


                if a[nx][ny] == '*':
                    a[nx][ny] = '.'
                    if d[nx][ny] == 0:
                        d[nx][ny] = d[x][y]+1
                    else:
                        d[nx][ny] = min(d[nx][ny], d[x][y]+1)
                    q.appendleft((nx, ny))


                elif a[nx][ny] == '.':
                    q.append((nx, ny))
                    if d[nx][ny] == 0:
                        d[nx][ny] = d[x][y]+1
                    else:
                        d[nx][ny] = min(d[nx][ny], d[x][y]+1)


    return a, d


## 더러운 칸을 모두 깨끗한 칸으로 바꾸었는지 체크 함수
def check(a, d):
    for i in range(h):
        for j in range(w):
            if a[i][j] =='*' and d[i][j] == 0: ## 만약, 방문할 수 없는 더러운 칸이 존재하는 경우에는 -1을 출력한다.
                return -1


    return


# 입력
while True:
    w, h = map(int, input().split())

    if w==0 and h==0:
        break # (문제 종료)


    a = [list(input()) for _ in range(h)]


    # 로봇 청소기의 시작 위치 찾기
    for i in range(h):
        for j in range(w):
            if a[i][j] == 'o':
                sx = i
                sy = j

                a[i][j] = '.'
                break

    a, d = bfs(a, w, h, sx, sy)

    if check(a, d) == -1:
        print(-1)
    else:
        print(max([max(row) for row in d]))


```


```python
for row in a:
    print(''.join(row))
```

    .......
    .......
    .......
    .......
    .......



```python
print(max([max(row) for row in d]))
```

    0



```python
for row in d:
    print(row)
```

    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]



```python
# 정답
# 순열 (next permutation) 이용

from collections import deque


dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]



def next_permutation(a):
    i = len(a)-1
    while i > 0 and a[i-1] >= a[i]:
        i -= 1
    if i <= 0:
        return False
    j = len(a)-1
    while a[j] <= a[i-1]:
        j -= 1

    a[i-1],a[j] = a[j],a[i-1]

    j = len(a)-1
    while i < j:
        a[i],a[j] = a[j],a[i]
        i += 1
        j -= 1

    return True



## BFS 탐색
def bfs(a, sx, sy):
    n = len(a)
    m = len(a[0])

    dist = [[-1]*m for _ in range(n)]
    dist[sx][sy] = 0

    q = deque()
    q.append((sx,sy))
    
    while q:
        x,y = q.popleft()
        for k in range(4):
            nx,ny = x+dx[k], y+dy[k]

            if 0 <= nx < n and 0 <= ny < m:
                if dist[nx][ny] == -1 and a[nx][ny] != 'x': # 아직 방문한 적 없고 && 벽이 아니면
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx,ny))

                    
    return dist




# 입력
while True:
    w, h = map(int, input().split())

    if w==0 and h==0:
        break # (문제 종료)


    a = [input() for _ in range(h)]

    b = [(0,0)] ## 특별히 방문해야 할 리스트 (로봇 청소기 시작 위치 && 더러운 곳(10개 이하)의 위치)

    # 로봇 청소기의 시작 위치 및 더러운 곳 위치 찾기
    for i in range(h):
        for j in range(w):
            if a[i][j] == 'o':
                b[0] = (i,j)

            elif a[i][j] == '*':
                b.append((i,j))


    l = len(b)
    d = [[0]*l for _ in range(l)]

    ok = True


    for i in range(l):
        dist = bfs(a,b[i][0], b[i][1])

        for j in range(l):
            d[i][j] = dist[b[j][0]][b[j][1]]

            if d[i][j] == -1:
                ok = False



    if not ok:
        print(-1)
        continue # 다음 입력으로 넘어감


    p = [i+1 for i in range(l-1)] ## i+1 : 시작 위치 빼고 1번 ~ L 번 먼지
    ans = -1
    while True:
        now = d[0][p[0]]

        for i in range(l-2):
            now += d[p[i]][p[i+1]]



        if ans == -1 or ans > now:
            ans = now

        if not next_permutation(p):
            break


    print(ans)

```

    7 5
    .......
    .o...*.
    .......
    .*...*.
    .......
    8
    0 0


# 1600번: 말이 되고픈 원숭이
#### https://www.acmicpc.net/problem/1600


```python
## 1600번: 말이 되고픈 원숭이
#### https://www.acmicpc.net/problem/1600



# 내가 시도한 풀이 (실패)


from collections import deque
import sys


# 말의 이동방식 (8방향)
dx = [-1, -2, -2, -1, 1, 2, 2, 1]
dy = [-2, -1, 1, 2, 2, 1, -1, -2]

# 원숭이의 이동방식 (4방향)
dx2 = [-1, 1, 0, 0]
dy2 = [0, 0, -1, 1]


# 입력
k = int(input())
w, h = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(h)]


## BFS 탐색
dist = [[-1]*w for _ in range(h)]
dist[0][0] = 0

q = deque()
q.append((0, 0))

cnt = 0
while q:
    x, y = q.popleft()
    

    if cnt <= k:
        for k in range(8):
            nx, ny = x+dx[k], y+dy[k]

            if 0<=nx<h and 0<=ny<w and dist[nx][ny]==-1 and a[nx][ny] != 1:
                if nx==h-1 and ny==w-1:
                    print(cnt+1)
                    sys.exit(0)

                dist[nx][ny] = d[x][y] +1
                q.append((nx, ny))
                cnt += 1
                

    else:
        for k in range(4):
            nx, ny = x+dx2[k], y+dy2[k]

            if 0<=nx<h and 0<=ny<w and dist[nx][ny]==-1 and a[nx][ny] != 1:
                if nx==h-1 and ny==w-1:
                    print(cnt+1)
                    sys.exit(0)

                dist[nx][ny] = d[x][y] +1
                q.append((nx, ny))
                cnt += 1


if dist[h-1][w-1]==-1:
    print(-1)
else:
    print(dist[h-1][w-1])

```

    1
    4 4
    0 0 0 0
    1 0 0 0
    0 0 1 0 
    0 1 0 0
    3



    An exception has occurred, use %tb to see the full traceback.


    SystemExit: 0



    /usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2890: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)



```python
for row in dist:
    print(row)
```

    [0, -1, -1, -1]
    [-1, -1, 1, -1]
    [-1, 1, -1, -1]
    [-1, -1, -1, -1]



```python
print(cnt)
```

    2



```python
# 정답


from collections import deque

# 말의 이동방식 (8방향) + 원숭이의 이동방식 (4방향)
dx = [0,0,1,-1,-2,-1,1,2,2,1,-1,-2]
dy = [1,-1,0,0,1,2,2,1,-1,-2,-2,-1]

cost = [0,0,0,0,1,1,1,1,1,1,1,1]



d = [[[-1]*31 for i in range(200)] for j in range(200)] #  K는 0이상 30이하의 정수이다.

# 입력
l = int(input())
m,n = map(int,input().split())

a = []
for i in range(n):
    a.append(list(map(int,input().split())))


## BFS 탐색
q = deque()
q.append((0,0,0))
d[0][0][0] = 0

while q:
    x,y,c = q.popleft()

    for k in range(12):
        nx = x+dx[k]
        ny = y+dy[k]
        nc = c+cost[k]

        if 0 <= nx < n and 0 <= ny < m:
            if a[nx][ny] == 1:
                continue

            if nc <= l:
                if d[nx][ny][nc] == -1:
                    d[nx][ny][nc] = d[x][y][c] + 1
                    q.append((nx,ny,nc)) # (x좌표, y좌표, 현재까지 말 이동 횟수)



ans = -1
for i in range(l+1):
    if d[n-1][m-1][i] == -1:
        continue

    if ans == -1 or ans > d[n-1][m-1][i]:
        ans = d[n-1][m-1][i]

print(ans)

```

    1
    4 4
    0 0 0 0
    1 0 0 0
    0 0 1 0
    0 1 0 0
    4



```python
a
```




    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]




```python
for row in dist:
    print(row)
```

    [[0, 3, 4, 6, 7, 10, 14, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [2, 3, 3, 5, 6, 9, 13, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [4, 2, 2, 4, 8, 11, 15, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [6, 2, 3, 4, 7, 11, 17, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 2, 3, 4, 5, 8, 12, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 1, 3, 3, 7, 10, 14, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 3, 2, 4, 6, 10, 16, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    [[3, 2, 2, 4, 4, 7, 10, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [2, 1, 3, 3, 5, 7, 11, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [4, 2, 3, 3, 5, 9, 15, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    [[4, 2, 3, 4, 5, 6, 9, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [3, 2, 3, 3, 5, 6, 11, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 3, 2, 4, 4, 8, 16, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]



```python
# 정답

https://www.acmicpc.net/source/share/dc94ab7715214d80bfa87f66434faad3
```

    1
    4 4
    0 0 0 0
    1 0 0 0
    0 0 1 0
    0 1 0 0
    2


# 17086번: 아기 상어 2
#### N×M 크기의 공간에 아기 상어 여러 마리가 있다. 공간은 1×1 크기의 정사각형 칸으로 나누어져 있다. 한 칸에는 아기 상어가 최대 1마리 존재한다.
#### 어떤 칸의 안전 거리는 그 칸과 가장 거리가 가까운 아기 상어와의 거리이다. 두 칸의 거리는 하나의 칸에서 다른 칸으로 가기 위해서 지나야 하는 칸의 수이고, 이동은 인접한 8방향(대각선 포함)이 가능하다.
#### 안전 거리가 가장 큰 어떤 칸의 안전 거리의 최댓값 출력. (0은 빈 칸, 1은 아기 상어가 있는 칸)


```python
## 17086번: 아기 상어 2
#### N×M 크기의 공간에 아기 상어 여러 마리가 있다. 공간은 1×1 크기의 정사각형 칸으로 나누어져 있다. 한 칸에는 아기 상어가 최대 1마리 존재한다.
#### 어떤 칸의 안전 거리는 그 칸과 가장 거리가 가까운 아기 상어와의 거리이다. 두 칸의 거리는 하나의 칸에서 다른 칸으로 가기 위해서 지나야 하는 칸의 수이고, 이동은 인접한 8방향(대각선 포함)이 가능하다.
#### 안전 거리가 가장 큰 어떤 칸의 안전 거리의 최댓값 출력. (0은 빈 칸, 1은 아기 상어가 있는 칸)

from collections import deque


# 8방향
dx = [-1,-1,-1,0,1,1,1,0]
dy = [-1,0,1,1,1,0,-1,-1]



# BFS 탐색
def bfs(a, sx, sy):
    d = [[-1]*m for _ in range(n)]
    d[sx][sy] = 0

    q = deque()
    q.append((sx,sy))

    while q:
        x, y = q.popleft()

        for k in range(8):
            nx, ny = x+dx[k], y+dy[k]

            #### 범위 체크 및 방문 여부 체크 ####
            if 0<=nx<n and 0<=ny<m and d[nx][ny]==-1:
                if a[nx][ny] == 1: ## 어떤 칸의 안전 거리는 그 칸과 가장 거리가 가까운 아기 상어와의 거리이다. -> 아기상어(1) 만나면 그만 탐색
                    return d[x][y] + 1

                else: ## 빈칸(0) 만나면 계속 진행
                    d[nx][ny] = d[x][y] + 1
                    q.append((nx,ny))



# 입력
n, m = map(int, input().split())
a = [list(map(int, input().split()))  for _ in range(n)]


# 답 구하기
ans = 0

for i in range(n):
    for j in range(m):
        if a[i][j] == 0:
            distance = bfs(a, i, j)

            if ans < distance:
                ans = distance


print(ans)
```

    5 4
    0 0 1 0
    0 0 0 0
    1 0 0 0 
    0 0 0 0
    0 0 0 1
    2


# 2234번: 성곽
#### 성의 지도를 입력받아서 다음을 계산하는 프로그램을 작성하시오.
#### 1. 이 성에 있는 방의 개수
#### 2. 가장 넓은 방의 넓이
#### 3. 하나의 벽을 제거하여 얻을 수 있는 가장 넓은 방의 크기

#### 벽에 대한 정보는 한 정수로 주어지는데, 서쪽에 벽이 있을 때는 1을, 북쪽에 벽이 있을 때는 2를, 동쪽에 벽이 있을 때는 4를, 남쪽에 벽이 있을 때는 8을 더한 값이 주어진다. (참고로 이진수의 각 비트를 생각하면 쉽다. 따라서 이 값은 0부터 15까지의 범위 안에 있다.)



```python
## 2234번: 성곽
#### 성의 지도를 입력받아서 다음을 계산하는 프로그램을 작성하시오.
#### 1. 이 성에 있는 방의 개수
#### 2. 가장 넓은 방의 넓이
#### 3. 하나의 벽을 제거하여 얻을 수 있는 가장 넓은 방의 크기

#### 벽에 대한 정보는 한 정수로 주어지는데, 서쪽에 벽이 있을 때는 1을, 북쪽에 벽이 있을 때는 2를, 동쪽에 벽이 있을 때는 4를, 남쪽에 벽이 있을 때는 8을 더한 값이 주어진다. (참고로 이진수의 각 비트를 생각하면 쉽다. 따라서 이 값은 0부터 15까지의 범위 안에 있다.)


from collections import deque

dx = [0, -1, 0, 1] ## 문제에서 주어진 1,2,3,4 순서대로 !
dy = [-1, 0, 1, 0]


# 입력
m, n = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]


d = [[0]*m for _ in range(n)] ## 방 번호 저장 및 방문 여부 파악할 거리 배열

def bfs(x, y, rooms): ## rooms: 방 번호
    q = deque()
    q.append((x,y))

    d[x][y] = rooms
    cnt = 0 ## cnt: 방의 크기 계산

    while q:
        x, y = q.popleft()
        cnt += 1

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]
            if nx < 0 or nx >= n or ny < 0 or ny >= m: ## 범위
                continue

            if d[nx][ny] != 0: ## 방문 여부
                continue

            if (a[x][y] & (1<<k)) > 0: ## 벽이 있다면
                continue

            q.append((nx,ny))
            d[nx][ny] = rooms

    return cnt ## 방의 크기 리턴



# 출력
#### 1. 이 성에 있는 방의 개수
rooms = 0 ## rooms: 방 번호 업데이트
room = [0] ## room: 1번 ~ (??)번까지의 방 크기 저장할 배열 리스트

for i in range(n):
    for j in range(m):
        if d[i][j] == 0:
            rooms += 1
            room.append((bfs(i, j, rooms)))
print(rooms)

### 2. 가장 넓은 방의 넓이
ans = 0
for i in range(1, rooms+1):
    if ans < room[i]:
        ans = room[i]
print(ans)


#### 3. 하나의 벽을 제거하여 얻을 수 있는 가장 넓은 방의 크기
ans = 0
for i in range(n):
    for j in range(m):
        x,y = i,j
        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]
            if nx < 0 or nx >= n or ny < 0 or ny >= m: ## 범위
                continue

            if (a[x][y] & (1<<k)) > 0: ## 벽이 있다면
                # 벽 부시기 시도
                if d[nx][ny] == d[x][y]: # (벽을 사이에 둔 두 칸이 같은 번호의 방이 아닌지 체크)
                    continue
                # 인접한 방 번호끼리 붙였을 때 합쳐지는 방의 크기 계산
                if ans < room[d[x][y]] + room[d[nx][ny]]:
                    ans = room[d[x][y]] + room[d[nx][ny]]
print(ans)
```

    7 4
    11 6 11 6 3 10 6
    7 9 6 13 5 15 5
    1 10 12 7 13 7 5
    13 11 10 8 10 12 13
    5
    9
    16


# 12906번: 새로운 하노이 탑
#### 막대 A, 막대 B, 막대 C에 놓여져 있는 원판의 상태가 주어졌을 때, 게임의 목표를 달성하는데 필요한 움직임의 최소 횟수를 구하는 프로그램을 작성하시오.
#### 게임의 목표는 막대 A에는 원판 A만, 막대 B는 원판 B만, 막대 C는 원판 C만 놓여져 있어야 한다.
#### https://www.acmicpc.net/problem/12906


```python
## 12906번: 새로운 하노이 탑
#### 막대 A, 막대 B, 막대 C에 놓여져 있는 원판의 상태가 주어졌을 때, 게임의 목표를 달성하는데 필요한 움직임의 최소 횟수를 구하는 프로그램을 작성하시오.
#### 게임의 목표는 막대 A에는 원판 A만, 막대 B는 원판 B만, 막대 C는 원판 C만 놓여져 있어야 한다.
#### https://www.acmicpc.net/problem/12906



from collections import deque

# 입력
s = []
for i in range(3):
    temp = input().split()
    cnt = int(temp[0])

    if cnt > 0:
        s.append(temp[1])
    else:
        s.append('')


## 각 알파벳의 개수를 카운트 (게임의 목표는 막대 A에는 원판 A만, 막대 B는 원판 B만, 막대 C는 원판 C만 놓여져 있어야 한다.)
cnt = [0, 0, 0]
for i in range(3):
    for ch in s[i]:
        cnt[ord(ch)-ord('A')] += 1


## BFS
d = dict() # 거리 배열 -> 튜플 형태의 정점 정보 담기 위해 딕셔너리 이용
d[tuple(s)] = 0 # 시행횟수는 0번 부터 시작

q = deque()
q.append(tuple(s))



while q:
    x = q.popleft()

    for i in range(3):
        for j in range(3):
            if i==j:
                continue

            if len(x[i]) == 0:
                continue


            y = list(x[:]) # 리스트 형 변환 (복사)
            y[j] = y[j] + x[i][-1]
            y[i] = y[i][:-1]

            y = tuple(y) # 다시 튜플로 변환

            if y not in d:
                d[y] = d[x] + 1
                q.append(y)


## 답 출력
ans = ['', '', ''] # 최종 구해야 하는 도착 정점 튜플 찾기 (게임의 목표는 막대 A에는 원판 A만, 막대 B는 원판 B만, 막대 C는 원판 C만 놓여져 있어야 한다.)
for i in range(3):
    for _ in range(cnt[i]):
        ans[i] += chr(ord('A') + i)


print(d[tuple(ans)])

```

    1 B
    1 C
    1 A
    5



```python
d
```




    {('', '', 'ABC'): 2,
     ('', '', 'ACB'): 2,
     ('', '', 'BAC'): 4,
     ('', '', 'BCA'): 5,
     ('', '', 'CAB'): 4,
     ('', '', 'CBA'): 5,
     ('', 'A', 'BC'): 5,
     ('', 'A', 'CB'): 4,
     ('', 'AB', 'C'): 4,
     ('', 'ABC', ''): 5,
     ('', 'AC', 'B'): 4,
     ('', 'ACB', ''): 4,
     ('', 'B', 'AC'): 2,
     ('', 'B', 'CA'): 4,
     ('', 'BA', 'C'): 5,
     ('', 'BAC', ''): 5,
     ('', 'BC', 'A'): 3,
     ('', 'BCA', ''): 4,
     ('', 'C', 'AB'): 1,
     ('', 'C', 'BA'): 3,
     ('', 'CA', 'B'): 2,
     ('', 'CAB', ''): 2,
     ('', 'CB', 'A'): 1,
     ('', 'CBA', ''): 2,
     ('A', '', 'BC'): 4,
     ('A', '', 'CB'): 5,
     ('A', 'B', 'C'): 5,
     ('A', 'BC', ''): 4,
     ('A', 'C', 'B'): 3,
     ('A', 'CB', ''): 2,
     ('AB', '', 'C'): 4,
     ('AB', 'C', ''): 3,
     ('ABC', '', ''): 4,
     ('AC', '', 'B'): 4,
     ('AC', 'B', ''): 5,
     ('ACB', '', ''): 5,
     ('B', '', 'AC'): 1,
     ('B', '', 'CA'): 3,
     ('B', 'A', 'C'): 3,
     ('B', 'AC', ''): 3,
     ('B', 'C', 'A'): 0,
     ('B', 'CA', ''): 1,
     ('BA', '', 'C'): 2,
     ('BA', 'C', ''): 1,
     ('BAC', '', ''): 2,
     ('BC', '', 'A'): 1,
     ('BC', 'A', ''): 2,
     ('BCA', '', ''): 2,
     ('C', '', 'AB'): 2,
     ('C', '', 'BA'): 4,
     ('C', 'A', 'B'): 5,
     ('C', 'AB', ''): 5,
     ('C', 'B', 'A'): 3,
     ('C', 'BA', ''): 4,
     ('CA', '', 'B'): 5,
     ('CA', 'B', ''): 4,
     ('CAB', '', ''): 5,
     ('CB', '', 'A'): 3,
     ('CB', 'A', ''): 4,
     ('CBA', '', ''): 4}



- 예제 1


```python
s = ['A', 'AA', 'AA']
tuple(s) # 튜플로 형변환
```




    ('A', 'AA', 'AA')




```python
cnt = [0, 0, 0] # 입력으로 주어진 각 알파벳의 개수 카운트
for i in range(3):
    for ch in s[i]:
        cnt[ord(ch) - ord('A')] += 1

cnt
```




    [5, 0, 0]




```python
# 최종 구해야 하는 도착 정점 튜플
ans = ['', '', '']

for i in range(3):
    for _ in range(cnt[i]):
        ans[i] += chr(ord('A')+i)

print(tuple(ans))
```

    ('AAAAA', '', '')


- 예제 2


```python
s = ['B', 'C', 'A']
cnt = [0, 0, 0]
for i in range(3):
    for ch in s[i]:
        cnt[ord(ch)-ord('A')] += 1

cnt        
```




    [1, 1, 1]




```python
cnt = [1, 1, 1]

ans = ['', '', '']
for i in range(3):
    for j in range(cnt[i]):
        print(i, j)
        print(chr(ord('A') + i))
        ans[i] += chr(ord('A') + i)
        print(ans)
```

    0 0
    A
    ['A', '', '']
    1 0
    B
    ['A', 'B', '']
    2 0
    C
    ['A', 'B', 'C']


#### # 꼭 튜플이여야 하는가? Yes !
- TypeError: unhashable type: 'list'
- 딕셔너리에서 key값은 유일하고 변하지 않는 값(숫자형, 문자형, 튜플)이 와야하고, Value는 모든 자료형을 가질 수 있다.


----------------------------------------------------------------------
- key값은 유일성,불변성의 특징을 갖기에, 따라서 리스트는 key값으로 올 수 없다.
- 리스트와 달리 튜플자료형은 불변하는 성질이 있으므로 key로 사용이 가능하다.

(https://yganalyst.github.io/data_handling/Py_study3/)

#### 튜플 다루기
- 삭제(del) 및 특정 요소값 변경 불가능
- 인덱싱/슬라이싱, 더하기/곱하기(확장), 길이 구하기(len) 가능


(https://wikidocs.net/15)


```python
# 꼭 튜플이여야 하는가? Yes
# TypeError: unhashable type: 'list'
# key값은 유일하고 변하지 않는 값(숫자형, 문자형, 튜플)이 와야하고, Value는 모든 자료형을 가질 수 있다.




from collections import deque

# 입력
s = []
for i in range(3):
    temp = input().split()
    cnt = int(temp[0])

    if cnt > 0:
        s.append(temp[1])
    else:
        s.append('')


## 각 알파벳의 개수를 카운트 (게임의 목표는 막대 A에는 원판 A만, 막대 B는 원판 B만, 막대 C는 원판 C만 놓여져 있어야 한다.)
cnt = [0, 0, 0]
for i in range(3):
    for ch in s[i]:
        cnt[ord(ch)-ord('A')] += 1


## BFS
d = dict() # 거리 배열 -> 튜플 형태의 정점 정보 담기 위해 딕셔너리 이용
d[s] = 0 # 시행횟수는 0번 부터 시작

q = deque()
q.append(s)



while q:
    x = q.popleft()

    for i in range(3):
        for j in range(3):
            if i==j:
                continue

            if len(x[i]) == 0:
                continue


            #y = list(x[:]) # 리스트 형 변환 (복사)
            y[j] = y[j] + x[i][-1]
            y[i] = y[i][:-1]

            #y = tuple(y) # 다시 튜플로 변환

            if y not in d:
                d[y] = d[x] + 1
                q.append(y)


## 답 출력
ans = ['', '', ''] # 최종 구해야 하는 도착 정점 튜플 찾기 (게임의 목표는 막대 A에는 원판 A만, 막대 B는 원판 B만, 막대 C는 원판 C만 놓여져 있어야 한다.)
for i in range(3):
    for _ in range(cnt[i]):
        ans[i] += chr(ord('A') + i)


print(d[ans])

```

    1 B
    1 C
    1 A



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-26-387a918f7c38> in <module>()
         29 ## BFS
         30 d = dict() # 거리 배열 -> 튜플 형태의 정점 정보 담기 위해 딕셔너리 이용
    ---> 31 d[s] = 0 # 시행횟수는 0번 부터 시작
         32 
         33 q = deque()


    TypeError: unhashable type: 'list'


# 9376번: 탈옥
#### 두 죄수를 탈옥시키기 위해서 열어야 하는 문의 개수를 구하는 프로그램을 작성하시오.
#### https://www.acmicpc.net/problem/9376


```python
## 9376번: 탈옥
#### 두 죄수를 탈옥시키기 위해서 열어야 하는 문의 개수를 구하는 프로그램을 작성하시오.
#### https://www.acmicpc.net/problem/9376



# 내가 시도한 풀이
from collections import deque

dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]





def dfs(x, y, dist, a):
    global check
    global ans

    for k in range(4):
        nx, ny = x+dx[k], y+dy[k]

        if 0<=nx<h and 0<=ny<w:
            if check[nx][ny] == False:
                if a[nx][ny] == '*':
                    continue
                elif a[nx][ny] == '#':
                    dist += 1
                    ##a[nx][ny].replace('.')
                    a[nx] = a[nx][:ny] + '.' + a[nx][ny+1:]
                    dfs(nx, ny, dist, a)
                    
                    a[nx] = a[nx][:ny] + '#' + a[nx][ny+1:]
                    dfs(nx, ny, dist-1, a)
            

            else:
                if ans == -1 or ans > dist:
                    ans = dist


                return ans
                




t = int(input())
for _ in range(t):
    h, w = map(int, input().split())

    a = [input() for _ in range(h)]


    x1=y1=x2=y2=-1
    for i in range(h):
        for j in range(w):
            if a[i][j]=='$':
                if x1==-1:
                    x1,y1 = i,j
                else:
                    x2,y2 = i,j


    dist = 0

    
    check = [[False]*w for _ in range(h)]
    check[x1][x2] = True

    ##if a[x][y] == '$':
    ##a[x][y] = '.'
    a[x1] = a[x1][:y1] + '.' + a[x1][y1+1:]


    ans = -1
    dist += dfs(x1, y1, 0, a)


    ans = -1
    dist += dfs(x2, y2, 0, a)



    print(ans)




```

    1
    5 9
    ****#****
    *..#.#..*
    ****.****
    *$#.#.#$*
    *********



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-29-552d0562833a> in <module>()
         74 
         75     ans = -1
    ---> 76     dist += dfs(x1, y1, 0)
         77 
         78 


    TypeError: dfs() missing 1 required positional argument: 'a'



```python
ans = -1
dist += dfs(x1, y1, 0, a)


ans = -1
dist += dfs(x2, y2, 0, a)



print(ans)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-30-fdff77e89bc3> in <module>()
          1 ans = -1
    ----> 2 dist += dfs(x1, y1, 0, a)
          3 
          4 
          5 ans = -1


    TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'



```python
ans = dfs(x1, y1, 0)

print(ans)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-28-11c22850d1fb> in <module>()
    ----> 1 ans = dfs(x1, y1, 0)
          2 
          3 print(ans)


    <ipython-input-25-d18985061bc5> in dfs(x, y, dist)
         28                     ##a[nx][ny].replace('.')
         29                     a[nx] = a[nx][:ny] + '.' + a[nx][ny+1:]
    ---> 30                     dfs(nx, ny, dist)
         31 
         32                     a[nx] = a[nx][:ny] + '#' + a[nx][ny+1:]


    <ipython-input-25-d18985061bc5> in dfs(x, y, dist)
         35 
         36             else:
    ---> 37                 if ans == -1 or ans > dist:
         38                     ans = dist
         39 


    TypeError: '>' not supported between instances of 'NoneType' and 'int'



```python
for row in a:
    print(row)
```

    ****.****
    *.......*
    ****.****
    *$.....$*
    *********



```python
a = '#'

a[0].replace('#', '.')
```




    '.'




```python
a
```




    '#'




```python
a[0]
```




    '#'




```python
# 정답

# 죄수 1, 죄수 2, 도착점 세가지에서 BFS 탐색한 결과(최소 시행횟수)를 이용해 답 구할 수 있다.


from collections import deque

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(a, x, y):
    n = len(a)
    m = len(a[0])

    dist = [[-1]*m for _ in range(n)]
    dist[x][y] = 0

    q = deque()
    q.append((x,y))


    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]

            if 0<=nx<n and 0<=ny<m and dist[nx][ny]==-1:
                if a[nx][ny] != '*': # 갈 수 없는 벽이 아닐 때만 탐색

                    ## (경우 1) 문일 때
                    if a[nx][ny] == '#':
                        dist[nx][ny] = dist[x][y]+1 # 시행횟수 1 추가
                        q.append((nx,ny))
                        # 큐의 뒤에 삽입 ! (append)
                        # 가중치 1이므로, BFS 시 우선순위 낮음 (구하려는 최소 시행횟수에 1번 추가 됨)

                    ## (경우 2) 빈칸일 때
                    else:
                        dist[nx][ny] = dist[x][y] # 그대로 이어 받음
                        q.appendleft((nx,ny)) ### appendleft !!
                        # 큐의 앞에 삽입 ! (appendlefr)
                        # 가중치 0이므로, BFS 시 우선순위 높음 (구하려는 최소 시행횟수에 0번 추가됨)

    return dist





t = int(input())
for _ in range(t):
    # 입력
    h, w = map(int, input().split())
    a = ['.' + input() + '.'  for _ in range(h)]

    h += 2
    w += 2
    a = ['.'*w] + a + ['.'*w]

    # 답 구하기
    d0 = bfs(a, 0, 0) ## 도착점은 (0,0) 으로 설정
    x1 = y1 = x2 = y2 = -1
    for i in range(h):
        for j in range(w):
            if a[i][j] == '$':
                if x1 == -1:
                    x1, y1 = i, j
                elif x2 == -1:
                    x2, y2 = i, j

    d1 = bfs(a, x1, y1)
    d2 = bfs(a, x2, y2)

    ans = h*w ## 최대값 초기화

    for i in range(h):
        for j in range(w):
            if a[i][j] == '*': # 벽
                continue

            if d0[i][j] == -1 or d1[i][j] == -1 or d2[i][j] == -1: ## 세가지 BFS탐색 결과에서 모두 방문한 정점이어야, 만나는 중간지점에 해당될 수 있음
                continue

            # 답 출력
            cur = d0[i][j] + d1[i][j] + d2[i][j]
            if a[i][j] == '#': # 문에서 셋이 만날 땐 구한 열어야 하는 문의 개수에서 겹치는 2번 빼야 함 
                cur -= 2



            ans = min(ans, cur)


    print(ans)

```

    3
    5 9
    ****#****
    *..#.#..*
    ****.****
    *$#.#.#$*
    *********
    4
    5 11
    *#*********
    *$*...*...*
    *$*.*.*.*.*
    *...*...*.*
    *********.*
    0
    9 9
    *#**#**#*
    *#**#**#*
    *#**#**#*
    *#**.**#*
    *#*#.#*#*
    *$##*##$*
    *#*****#*
    *.#.#.#.*
    *********
    9

