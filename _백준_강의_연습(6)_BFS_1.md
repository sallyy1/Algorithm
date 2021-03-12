# 16928번: 뱀과 사다리 게임
#### 게임판의 상태가 주어졌을 때, 1번 칸에서 시작해서 100번 칸에 도착하기 위해 주사위를 굴려야 하는 횟수의 최솟값을 구해보자.
#### 플레이어는 주사위를 굴려 나온 수만큼 이동해야 한다. (그때 도착한 칸이 사다리면, 사다리를 타고 위로 올라가고, 뱀이면, 뱀을 따라서 내려가게 된다.)


```python
## 16928번: 뱀과 사다리 게임
#### 게임판의 상태가 주어졌을 때, 1번 칸에서 시작해서 100번 칸에 도착하기 위해 주사위를 굴려야 하는 횟수의 최솟값을 구해보자.
#### 플레이어는 주사위를 굴려 나온 수만큼 이동해야 한다. (그때 도착한 칸이 사다리면, 사다리를 타고 위로 올라가고, 뱀이면, 뱀을 따라서 내려가게 된다.)


# dist[v] : v에 도착하는 최소 주사위 던짐 횟수
# next[x] : 주사위를 통해 도착한 칸이 x일 때, 이동해야 하는 칸
# 1)일반칸: x -> x
# 2)사다리: x -> y
# 3)뱀: x -> y


from collections import deque

n, m = map(int, input().split())

after = list(range(101)) ## next[x] 배열 만들어줌
dist = [-1]*101

for _ in range(n): # 사다리 이동 정보
    x, y = map(int, input().split())
    after[x] = y

for _ in range(m): # 뱀 이동 정보
    u, v = map(int, input().split())
    after[u] = v



# 시작점 초기화
dist[1] = 0
q = deque()
q.append(1)


while q:
    x = q.popleft()

    for i in range(1, 6+1): # 주사위 1~6번 칸까지 이동 시도
        y = x+i

        if y <= 100: # (예외처리)보드판 범위 내에 해당되는지 체크
            y = after[y] ## 실질적으로 이동하는 칸은 1)주사위 -> 뱀 or 사다리가 아닌, 2) 이전 칸 -> 바로 뱀 or 사다리 만큼
            ## 따라서 주사위만큼 이동한 중간 칸인 x+i는 방문 여부 처리해주면 안 됨

            if dist[y] == -1: # 아직 방문 안한 칸이었다면,
                dist[y] = dist[x] + 1 # (1)거리 1 추가
                q.append(y) # (2)방문할 큐에 추가


print(dist[100])
```

    3 7
    32 62
    42 68
    12 98
    95 13
    97 25
    93 37
    79 27
    75 19
    49 47
    67 17
    3



```python
## after[x] = x or y

# 1)일반칸: x -> x (그대로)
# 2)사다리: x -> y (문제 입력으로 주어진 더 큰 칸으로 이동)
# 3)뱀: x -> y (문제 입력으로 주어진 더 작은 칸으로 이동)


for index, after in enumerate(after):
    print(index, after) # 인덱스, 이동후 칸 번호
```

    0 0
    1 1
    2 2
    3 3
    4 4
    5 5
    6 6
    7 7
    8 8
    9 9
    10 10
    11 11
    12 98
    13 13
    14 14
    15 15
    16 16
    17 17
    18 18
    19 19
    20 20
    21 21
    22 22
    23 23
    24 24
    25 25
    26 26
    27 27
    28 28
    29 29
    30 30
    31 31
    32 62
    33 33
    34 34
    35 35
    36 36
    37 37
    38 38
    39 39
    40 40
    41 41
    42 68
    43 43
    44 44
    45 45
    46 46
    47 47
    48 48
    49 47
    50 50
    51 51
    52 52
    53 53
    54 54
    55 55
    56 56
    57 57
    58 58
    59 59
    60 60
    61 61
    62 62
    63 63
    64 64
    65 65
    66 66
    67 17
    68 68
    69 69
    70 70
    71 71
    72 72
    73 73
    74 74
    75 19
    76 76
    77 77
    78 78
    79 27
    80 80
    81 81
    82 82
    83 83
    84 84
    85 85
    86 86
    87 87
    88 88
    89 89
    90 90
    91 91
    92 92
    93 37
    94 94
    95 13
    96 96
    97 25
    98 98
    99 99
    100 100


# 16948번: 데스 나이트
#### 데스 나이트가 있는 곳이 (r, c)라면, (r-2, c-1), (r-2, c+1), (r, c-2), (r, c+2), (r+2, c-1), (r+2, c+1)로 이동할 수 있다.
#### 크기가 N×N인 체스판과 두 칸 (r1, c1), (r2, c2)가 주어진다. 데스 나이트가 (r1, c1)에서 (r2, c2)로 이동하는 최소 이동 횟수를 구해보자. 체스판의 행과 열은 0번부터 시작한다.



```python
## 16948번: 데스 나이트
#### 데스 나이트가 있는 곳이 (r, c)라면, (r-2, c-1), (r-2, c+1), (r, c-2), (r, c+2), (r+2, c-1), (r+2, c+1)로 이동할 수 있다.
#### 크기가 N×N인 체스판과 두 칸 (r1, c1), (r2, c2)가 주어진다. 데스 나이트가 (r1, c1)에서 (r2, c2)로 이동하는 최소 이동 횟수를 구해보자. 체스판의 행과 열은 0번부터 시작한다.



# 내가 풀은 답

from collections import deque

n = int(input())
r1, c1, r2, c2 = map(int, input().split())

dist = [[-1]*n for _ in range(n)]
dx = [-2, -2, 0, 0, 2, 2]
dy = [-1, 1, -2, 2, -1, 1]

# 출발점 초기화
dist[r1][c1] = 0 # 시작은 0회
q = deque()
q.append((r1, c1))


while q:
    x, y = q.popleft()

    ## 종료조건
    if x == r2 and y == c2:
        print(dist[x][y])
        break



    for k in range(6):
        nx, ny = x+dx[k], y+dy[k]

        if 0<= nx <n and 0<= ny <n:
            if dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))

# 문제 조건: 이동할 수 없는 경우에는 -1을 출력한다.
else:
    print(-1)
```

    6
    5 1 0 5
    -1



```python
# 정답


from collections import deque
dx = [-2,-2,0,0,2,2]
dy = [-1,1,-2,2,-1,1]
dist = [[-1]*200 for _ in range(200)]
n = int(input())
sx,sy,ex,ey = map(int,input().split())
q = deque()
q.append((sx,sy))
dist[sx][sy] = 0
while q:
    x,y = q.popleft()
    for k in range(6):
        nx,ny = x+dx[k],y+dy[k]
        if 0 <= nx < n and 0 <= ny < n:
            if dist[nx][ny] == -1:
                q.append((nx,ny))
                dist[nx][ny] = dist[x][y] + 1
print(dist[ex][ey])
```

# 14502번: 연구소
#### 크기가 N×M인 직사각형 연구소는 빈 칸, 벽으로 이루어져 있으며, 벽은 칸 하나를 가득 차지한다. 일부 칸은 바이러스가 존재하며, 이 바이러스는 상하좌우로 인접한 빈 칸으로 모두 퍼져나갈 수 있다.
#### 벽 3개를 새로 세울 때, 바이러스가 퍼질 수 없는 안전 영역 크기의 최댓값을 구하는 프로그램을 작성하시오. (0은 빈 칸, 1은 벽, 2는 바이러스가 있는 곳이다.)



```python
## 14502번: 연구소
#### 크기가 N×M인 직사각형 연구소는 빈 칸, 벽으로 이루어져 있으며, 벽은 칸 하나를 가득 차지한다. 일부 칸은 바이러스가 존재하며, 이 바이러스는 상하좌우로 인접한 빈 칸으로 모두 퍼져나갈 수 있다.
#### 벽 3개를 새로 세울 때, 바이러스가 퍼질 수 없는 안전 영역 크기의 최댓값을 구하는 프로그램을 작성하시오. (0은 빈 칸, 1은 벽, 2는 바이러스가 있는 곳이다.)


# 1. 벽을 3개 세움
# 2. 바이러스가 퍼질 수 없는 곳의 크기 구하기
## 모든 경우의 수 구해야 하므로 '브루트 포스'문제


from collections import deque

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]


def bfs(a):
    n = len(a)
    m = len(a[0])

    b = [[0]*m for _ in range(n)]

    # 초기화
    q = deque()
    for i in range(n):
        for j in range(m):
            b[i][j] = a[i][j] # b배열에 복사하고
            if b[i][j] == 2: # 바이러스(2) 칸이라면 방문할 큐에 담기
                q.append((i, j))

    # BFS 탐색
    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]
            if 0<=nx<n and 0<=ny<m: # 범위 내 있고
                if b[nx][ny] == 0: # 아직 바이러스가 안 퍼진(0) 칸이라면,
                    b[nx][ny] = 2 # (1)바이러스를 퍼뜨리고
                    q.append((nx, ny)) # (2)방문할 큐에 추가



    # 바이러스를 모두 퍼뜨렸는데도 아직 빈칸(0)인 안전 영역의 크기 계산
    cnt = 0
    for i in range(n):
        for j in range(m):
            if b[i][j] == 0:
                cnt += 1

    return cnt


# 문제 입력 및 답 출력
n, m = map(int, input().split())
a = [list(map(int, input().split()))for _ in range(n)]


ans = 0
for x1 in range(n):
    for y1 in range(m):
        if a[x1][y1] != 0:
            continue # 빈칸(0)이 아니면 pass

        for x2 in range(n):
            for y2 in range(m):
                if a[x2][y2] != 0:
                    continue # 빈칸(0)이 아니면 pass

                for x3 in range(n):
                    for y3 in range(m):
                        if a[x3][y3] != 0:
                            continue # 빈칸(0)이 아니면 pass


                        # 벽 새로 세우기로 선택한 세 좌표 중 같은 좌표가 있다면 pass
                        if x1 == x2 and y1 == y2:
                            continue
                        if x2 == x3 and y2 == y3:
                            continue
                        if x3 == x1 and y3 == y1:
                            continue


                        # 1)벽(1) 새로 세우고
                        a[x1][y1] = 1
                        a[x2][y2] = 1
                        a[x3][y3] = 1


                        # 2)bfs 하여 안전 영역의 크기 구한 후
                        cur = bfs(a)

                        if ans < cur:
                            ans = cur # 최대값 교체

                        # 3)원래 빈 벽(0)으로 복구
                        a[x1][y1] = 0
                        a[x2][y2] = 0
                        a[x3][y3] = 0


print(ans)
```
