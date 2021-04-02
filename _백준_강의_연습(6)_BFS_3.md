# 16236번: 아기상어
#### https://www.acmicpc.net/problem/16236


```python
## 16236번: 아기상어
#### https://www.acmicpc.net/problem/16236



# A[r][c] = 0이면 물고기 없음, 0보다 크면 해당 물고기의 크기 (물고기는 한 칸에는 최대 1마리 존재한다.)


from collections import deque

dx = [0, 0, -1, 1]
dy = [1, -1, 0, 0]



# 더 이상 먹을 수 있는 물고기가 공간에 있는지 / 없는지 탐색하는 BFS (있으면, 가장 위&왼쪽 위치 리턴 // 없다면, None 리턴 -> 엄마상어에게 도움 요청)
#### 아기 상어는 자신의 크기보다 큰 물고기가 있는 칸은 지나갈 수 없고, 나머지 칸은 모두 지나갈 수 있다. 
#### 아기 상어는 자신의 크기보다 작은 물고기만 먹을 수 있다. 따라서, 크기가 같은 물고기는 먹을 수 없지만, 그 물고기가 있는 칸은 지나갈 수 있다.
def bfs(a, x, y, size):
    ans = []
    n = len(a)

    q = deque()
    q.append((x, y))

    d = [[-1]*n for _ in range(n)]
    d[x][y] = 0 # 0초에서 시작



    while q:
        x, y = q.popleft()

        for k in range(4):
            nx = x+dx[k]
            ny = y+dy[k]

            if 0<=nx<n and 0<= ny <n and d[nx][ny] == -1:
                ok = False # 이동할 수 있는지 여부
                eat = False # 먹을 수 있는지 여부

                
                if a[nx][ny]==0: ## 빈칸은 당연히 지나갈 수 있음 && 먹을 건 없으니 pass
                    ok = True
                    #eat == False #(그대로)

                #if size < a[nx][ny]:
                    #ok = False # (그대로)

                elif size >= a[nx][ny]:
                    ok = True
                    if size > a[nx][ny]:
                        eat = True
                    #elif size == a[nx][ny]:
                        #eat = False # (그대로)
                


                if ok == True:
                    q.append((nx, ny))
                    d[nx][ny] = d[x][y] + 1 # 현재 최단 거리에 +1 업데이트

                    if eat == True:
                        ans.append((d[nx][ny], nx, ny)) # sort를 위함 (거리가 가까운 물고기가 많다면, 가장 위에 있는 물고기, 그러한 물고기가 여러마리라면, 가장 왼쪽에 있는 물고기를 먹는다.)

    
    if not ans:
        return None

    ans.sort()
    return ans[0]

              
                






# 입력
n = int(input())
a = [list(map(int, input().split())) for _ in range(n)]

x, y = 0, 0
for i in range(n):
    for j in range(n):
        if a[i][j] == 9: # 아기상어의 시작 위치 찾기
            x,y = i,j
            a[i][j] = 0


# 답 구하기
ans = 0
size = 2
exp = 0

while True:
    p = bfs(a, x, y, size)

    if p is None:
        break # (종료 조건) 엄마 상어에게 도움 요청

    dist, nx, ny = p
    a[nx][ny] = 0 # 먹을 수 있으므로 먹음
    ans += dist

    # (경험치 조건) 아기 상어는 자신의 크기와 같은 수의 물고기를 먹을 때 마다 크기가 1 증가한다.
    exp += 1
    if size == exp:
        size += 1
        exp = 0 # 리셋

    
    x, y = nx, ny




print(ans)

```

    3
    0 0 1
    0 0 0
    0 9 0
    3



```python
p = None

p is None
```




    True




```python
p == None
```




    True




```python
ans = []

if not ans:
    print('yes')
```

    yes



```python
if ans == []:
    print('yes')
```

    yes


# 6087번: 레이저 통신
#### https://www.acmicpc.net/problem/6087


```python
## 6087번: 레이저 통신
#### https://www.acmicpc.net/problem/6087


# 필요한 거울의 개수 = 직선의 개수 -1

from collections import deque

dx =[0, 0, -1, 1]
dy =[-1, 1, 0, 0]

w, h = map(int, input().split())
a = [list(input()) for _ in range(h)]

# 시작 좌표와 도착 좌표 찾기
sx=sy=ex=ey= -1
for i in range(h):
    for j in range(w):
        if a[i][j] == 'C':
            if sx == -1:
                sx, sy = i, j

            elif ex == -1:
                ex, ey = i, j
              

            
# BFS 탐색
dist = [[-1]*w for _ in range(h)]
dist[sx][sy] = 0

q = deque()
q.append((sx, sy))

while q:
    x, y = q.popleft()

    for k in range(4):
        nx, ny = x+dx[k], y+dy[k]
        
        while 0<=nx<h and 0<=ny<w:
            if a[nx][ny] == '*': # 벽을 만나면
                break

            if dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1  # 직선의 개수를 1 추가
                q.append((nx, ny))

            # 벽을 만나거나 범위를 벗어나기 전까지
            # 현재 k방향으로 연속되는 점 계속 탐색
            nx += dx[k]
            ny += dy[k]

print(dist[ex][ey] - 1)
```

    7 8
    .......
    ......C
    ......*
    *****.*
    ....*..
    ....*..
    .C..*..
    .......
    3


# 1963: 소수 경로
#### https://www.acmicpc.net/problem/1963


```python
## 1963: 소수 경로
#### https://www.acmicpc.net/problem/1963



from collections import deque

# change 함수
def change(num, index, digit):
    if index==0 and digit==0:
        return -1 # 첫번째 자릿수는 0으로 변경 불가능

    s = list(str(num))
    s[index] = chr(digit+ord('0'))

    return int(''.join(s))



prime = [False] * 10001

# 소수 찾기
prime = [False] * 10001 # 에라토스테네스의 체 이용
prime[0] = prime[1] = True # (예외 처리) 소수가 아님 -> 지움 처리

for i in range(2, 10001):
    if prime[i] == False: ## 지워지지 않음 (소수 X)
        for j in range(i*i, 10001, i):
            prime[j] = True ## 지워짐 (소수 O)


# 지워지지 X -> 소수 True // 지워짐 O -> 소수 False 로 변환
for i in range(10001):
    prime[i] = not prime[i]



t = int(input())
for _ in range(t):
    n, m = map(int, input().split())

    c = [False] * 10001 # 방문 여부 배열
    c[n] = True

    dist = [0] * 10001 # 최단 거리 배열
    dist[n] = 0

    q = deque() # 방문할 큐
    q.append(n)

    

    while q:
        now = q.popleft()
        
        for i in range(0, 4): # 4자리 수의 각 인덱스마다
            for j in range(0, 10): # 0~9 숫자로 change 함수 돌려보기
                next = change(now, i, j)

                if next != -1: # 변경 불가능한 경우 빼고
                    if prime[next] == True and c[next] == False: # 다음 이동할 수가 소수이고 && 아직 방문한 적 없는 수라면
                        dist[next] = dist[now] + 1 ## 거리 1 증가
                        q.append(next) ## 방문할 큐에 추가
                        c[next] = True ## 방문 표시
    

    
    if c[m] == False:
        print('Impossible')

    else:
        print(dist[m])

```

    3
    1033 8179
    6
    1373 8017
    7
    1033 1033
    0



```python
ord('0')
```




    48




```python
chr(1)
```




    '\x01'




```python
chr(1 + ord('0'))
```




    '1'




```python
chr(9)
```




    '\t'




```python
chr(1) + chr(ord('0'))
```




    '\x010'



## 에라토스테네스의 체
- 에라토스테네스의 체를 사용한 경우,
- 어떤 수 N이 소수인지 아닌지 판별하기 위해 루트 N 방법을 사용할 필요가 없다.

- 에라토스테네스의 체의 결과에서 지워지지 않았으면 소수, 아니면 소수가 아니기 때문이다.

# 10026: 적록색약
#### https://www.acmicpc.net/problem/10026
#### 상하좌우로 인접해 있는 경우에 두 글자는 같은 구역에 속한다.
#### 그림이 입력으로 주어졌을 때, 적록색약인 사람이 봤을 때와 아닌 사람이 봤을 때 구역의 수를 구하는 프로그램을 작성하시오.



```python
## 10026: 적록색약
#### https://www.acmicpc.net/problem/10026
#### 상하좌우로 인접해 있는 경우에 두 글자는 같은 구역에 속한다.
#### 그림이 입력으로 주어졌을 때, 적록색약인 사람이 봤을 때와 아닌 사람이 봤을 때 구역의 수를 구하는 프로그램을 작성하시오.



from collections import deque



# 적록색약 여부에 따라 인접한 두 칸이 같은 구역에 속하는지/아닌지 구하는 함수
def can(blind, fro, to):
    if fro == to:
        return True

    if blind==True:
        if fro=='R' and to=='G':
            return True
        elif fro=='G' and to=='R':
            return True


    return False



# 입력
n = int(input())
a = [input() for _ in range(n)]


# BFS 탐색
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

def go(a, blind):
    n = len(a)
    check = [[False]*n for _ in range(n)]

    ans = 0
    for i in range(n):
        for j in range(n):
            if check[i][j] == True:
                continue # 이미 방문했다면 pass

            ## 다음 구역으로 넘어감 (답 +1)
            ans += 1
            q = deque()
            q.append((i, j))
            check[i][j] = True

            while q:
                x, y = q.popleft()

                for k in range(4):
                    nx, ny = x+dx[k], y+dy[k]

                    if 0<=nx<n and 0<=ny<n:
                        if check[nx][ny] == True:
                            continue

                        else: # 아직 방문한 적 없을 때만
                            if can(blind, a[x][y], a[nx][ny])==True:
                                ## 같은 구역 내 다음 이동 진행
                                check[nx][ny]=True
                                q.append((nx, ny))

    return ans


# 답 출력
print(str(go(a, False)) + ' ' + str(go(a,True)))
```

    5
    RRRBB
    GGBBB
    BBBRR
    BBBRR
    RRRRR
    4 3


# 14395번: 4연산
#### https://www.acmicpc.net/problem/14395


```python
## 14395번: 4연산
#### https://www.acmicpc.net/problem/14395

from collections import deque

limit = 1000000000 # 10**9  (1 ≤ s, t ≤ 109)

s, t = map(int, input().split())

check = set()
check.add(s)

q = deque()
q.append((s, ''))

while q:
    x, ops = q.popleft()

    # 종료 조건
    if x == t:
        if len(ops) == 0: ## s와 t가 같은 경우에는 0을 출력
            ops = '0'

        print(ops)
        exit()

    # 다음연산이 범위 내 가능하고 && 아직 방문한 적 없다면
    if 0<= x*x <= limit and x*x not in check:
        q.append((x*x, ops+'*'))
        check.add(x*x)

    if 0<= x+x <= limit and x+x not in check:
        q.append((x+x, ops+'+'))
        check.add(x+x)

    if 0<= x-x <= limit and x-x not in check:
        q.append((x-x, ops+'-'))
        check.add(x-x)

    if x != 0 and 0<= x//x <= limit and x//x not in check:
        q.append((x//x, ops+'/'))
        check.add(x//x)


print(-1) ## 바꿀 수 없는 경우에는 -1을 출력
```

    7 256
    /+***
    -1


# 5014번: 스타트링크
#### https://www.acmicpc.net/problem/5014



```python
## 5014번: 스타트링크
#### https://www.acmicpc.net/problem/5014

from collections import deque

f, s,g, u,d = map(int, input().split())


# 3가지 선언 및 초기화
check = [False] * (f+1)
check[s] = True

dist = [0] * (f+1)
dist[s] = 0

q = deque()
q.append(s)



while q:
    now = q.popleft()

    if now+u <= f and check[now+u]==False:
        check[now+u] = True
        dist[now+u] = dist[now] + 1
        q.append(now+u)


    if 1 <= now-d and check[now-d]==False:
        check[now-d] = True
        dist[now-d] = dist[now] + 1
        q.append(now-d)




# 답 출력
if check[g]==True:
    print(dist[g])

else: ## (문제 조건) 만약, 엘리베이터로 이동할 수 없을 때는 "use the stairs"를 출력한다.
    print("use the stairs")
```

    10 1 10 2 1
    6


# 연구소 시리즈

## - 연구소 1 (14502번)
- 일부 칸은 바이러스가 존재하며, 이 바이러스는 상하좌우로 인접한 빈 칸으로 모두 퍼져나갈 수 있다. 새로 세울 수 있는 벽의 개수는 3개이며, 꼭 3개를 세워야 한다.
- 벽을 3개 세운 뒤, 바이러스가 퍼질 수 없는 곳을 안전 영역이라고 한다.
- 연구소의 지도가 주어졌을 때 얻을 수 있는 안전 영역 크기의 최댓값을 구하는 프로그램을 작성하시오.
https://www.acmicpc.net/problem/14502


## - 연구소 2 (17141번)
- 일부 빈 칸은 바이러스를 놓을 수 있는 칸이다. 바이러스는 상하좌우로 인접한 모든 빈 칸으로 동시에 복제되며, 1초가 걸린다.
- 연구소의 상태가 주어졌을 때, 모든 빈 칸에 바이러스를 퍼뜨리는 최소 시간을 구해보자.
https://www.acmicpc.net/problem/17141


## - 연구소 3
- 바이러스는 활성 상태와 비활성 상태가 있다. 가장 처음에 모든 바이러스는 비활성 상태이고, 활성 상태인 바이러스는 상하좌우로 인접한 모든 빈 칸으로 동시에 복제되며, 1초가 걸린다.
- 승원이는 연구소의 바이러스 M개를 활성 상태로 변경하려고 한다. 활성 바이러스가 비활성 바이러스가 있는 칸으로 가면 비활성 바이러스가 활성으로 변한다.
- 연구소의 상태가 주어졌을 때, 모든 빈 칸에 바이러스를 퍼뜨리는 최소 시간을 구해보자.
https://www.acmicpc.net/problem/17142




```python
## 연구소 1 (14502번)
#### 일부 칸은 바이러스가 존재하며, 이 바이러스는 상하좌우로 인접한 빈 칸으로 모두 퍼져나갈 수 있다. 새로 세울 수 있는 벽의 개수는 3개이며, 꼭 3개를 세워야 한다.
#### 벽을 3개 세운 뒤, 바이러스가 퍼질 수 없는 곳을 안전 영역이라고 한다.
#### 연구소의 지도가 주어졌을 때 얻을 수 있는 안전 영역 크기의 최댓값을 구하는 프로그램을 작성하시오.

#### https://www.acmicpc.net/problem/14502


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

    4 6
    0 0 0 0 0 0 
    1 0 0 0 0 2
    1 1 1 0 0 2
    0 0 0 0 0 2
    9



```python
## - 연구소 2 (17141번)

#### 일부 빈 칸은 바이러스를 놓을 수 있는 칸이다. 바이러스는 상하좌우로 인접한 모든 빈 칸으로 동시에 복제되며, 1초가 걸린다.
#### 연구소의 상태가 주어졌을 때, 모든 빈 칸에 바이러스를 퍼뜨리는 최소 시간을 구해보자.
#### https://www.acmicpc.net/problem/17141


from collections import deque

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]



# 문제 입력 및 답 출력
n, virus = map(int, input().split())
a = [list(map(int, input().split()))for _ in range(n)]

candi = [] # 일부 빈 칸은 바이러스를 놓을 수 있는 칸이다. (놓을 수 있는 바이러스의 개수는 입력받는 virus 값)

for i in range(n):
    for j in range(n):
        if a[i][j] == 2:
            candi.append((i, j))
            a[i][j] = 0 ## 일반 빈칸으로 변경됨 !




## 재귀 - <선택> 문제로 구현
# candi 후보 중 주어진 virus만큼 선택해 바이러스 놓고 BFS 수행

def go(index, cnt):
    # 종료 조건
    if index == len(candi):
        if cnt == virus:
            bfs()

    # 다음 재귀 수행 (선택)
    else:
        # 경우 1) 선택 O
        x, y = candi[index]
        a[x][y] = 3
        go(index+1, cnt+1)

        # 복원 !!
        a[x][y] = 0

        # 경우 2) 선택 X
        go(index+1, cnt)




## 모든 빈 칸에 바이러스를 퍼뜨리는 최소 시간을 구하기 위한 BFS 탐색
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

ans = -1

def bfs():
    dist = [[-1]*n for _ in range(n)]

    q = deque()
    for i in range(n):
        for j in range(n):
            if a[i][j] == 3: ## 새로 바이러스를 놓은 virus개 칸에 대하여
                q.append((i, j)) # 방문할 큐에 추가
                dist[i][j] = 0 # 0초에서 시작


    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]

            if 0<=nx<n and 0<=ny<n:
                if a[nx][ny] == 1: # 벽
                    continue

                if dist[nx][ny] == -1: # 아직 방문 안함
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))


    ## 문제 답 계산
    # 연구소의 모든 빈 칸에 바이러스가 있게 되는 최소 시간을 출력한다. 바이러스를 어떻게 놓아도 모든 빈 칸에 바이러스를 퍼뜨릴 수 없는 경우에는 -1을 출력한다.
    cur = 0
    for i in range(n):
        for j in range(n):
            if a[i][j] == 0:
                if dist[i][j] == -1:
                    return -1
                    #return ## ans=-1 로 초기화했으므로 그냥 return으로 종료만 해도 가능


                if cur < dist[i][j]:
                    cur = dist[i][j] # 해당 경우의 최대 시간


    # 가장 최소 시간 구하기
    global ans
    if ans == -1 or ans > cur:
        ans = cur
        



          
go(0, 0)
print(ans)
```

    5 1
    2 2 2 1 1
    2 1 1 1 1
    2 1 1 1 1
    2 1 1 1 1
    2 2 2 1 1
    4



```python
print(go(0, 0))
```

    None



```python
ans
```




    [3, -1, 2, 1, 2, 3, 4]




```python
dist = [
[0, 1, 2, 3, -1, -1, 2],
[5, -1, 3, 2, 3, 4, 5]]
```


```python
max(dist)
```




    [5, -1, 3, 2, 3, 4, 5]




```python
## 연구소 3

#### 바이러스는 활성 상태와 비활성 상태가 있다. 가장 처음에 모든 바이러스는 비활성 상태이고, 활성 상태인 바이러스는 상하좌우로 인접한 모든 빈 칸으로 동시에 복제되며, 1초가 걸린다.
#### 승원이는 연구소의 바이러스 M개를 활성 상태로 변경하려고 한다. 활성 바이러스가 비활성 바이러스가 있는 칸으로 가면 비활성 바이러스가 활성으로 변한다.
#### 연구소의 상태가 주어졌을 때, 모든 빈 칸에 바이러스를 퍼뜨리는 최소 시간을 구해보자.
#### https://www.acmicpc.net/problem/17142




from collections import deque

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]



# 문제 입력 및 답 출력
n, virus = map(int, input().split())
a = [list(map(int, input().split()))for _ in range(n)]

candi = [] # 일부 빈 칸은 바이러스를 놓을 수 있는 칸이다. (놓을 수 있는 바이러스의 개수는 입력받는 virus 값)

for i in range(n):
    for j in range(n):
        if a[i][j] == 2:
            candi.append((i, j))
            ## 그대로 2번 '비활성 바이러스'로 남겨 놓음 #a[i][j] = 0 ## 일반 빈칸으로 변경됨 !




## 재귀 - <선택> 문제로 구현
# candi 후보 중 주어진 virus만큼 선택해 바이러스 놓고 BFS 수행

def go(index, cnt):
    # 종료 조건
    if index == len(candi):
        if cnt == virus:
            bfs()

    # 다음 재귀 수행 (선택)
    else:
        # 경우 1) 선택 O
        x, y = candi[index]
        a[x][y] = 3 ## 바이러스 -> 활성 바이러스(3번)로 변경
        go(index+1, cnt+1)

        # 복원 !!
        a[x][y] = 2 ## 원래 기본 비활성 바이러스(2번)로 다시 복원

        # 경우 2) 선택 X
        go(index+1, cnt)




## 모든 빈 칸에 바이러스를 퍼뜨리는 최소 시간을 구하기 위한 BFS 탐색
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

ans = -1

def bfs():
    dist = [[-1]*n for _ in range(n)]

    q = deque()
    for i in range(n):
        for j in range(n):
            if a[i][j] == 3: ## 새로 바이러스를 놓은 virus개 칸에 대하여
                q.append((i, j)) # 방문할 큐에 추가
                dist[i][j] = 0 # 0초에서 시작


    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]

            if 0<=nx<n and 0<=ny<n:
                if a[nx][ny] == 1: # 벽
                    continue

                if dist[nx][ny] == -1: # 아직 방문 안함
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))


    ## 문제 답 계산
    # 연구소의 모든 빈 칸에 바이러스가 있게 되는 최소 시간을 출력한다. 바이러스를 어떻게 놓아도 모든 빈 칸에 바이러스를 퍼뜨릴 수 없는 경우에는 -1을 출력한다.
    cur = 0
    for i in range(n):
        for j in range(n):
            ## 연구소의 상태가 주어졌을 때, "모든 빈 칸"에 바이러스를 퍼뜨리는 최소 시간을 구해보자.
            if a[i][j] == 0: ## 빈칸일 때
                if dist[i][j] == -1:
                    return -1 ## (조기 종료)
                    #return ## ans=-1 로 초기화했으므로 그냥 return으로 종료만 해도 가능

                if cur < dist[i][j]:
                    cur = dist[i][j] ## 해당 경우의 최대 시간 비교


    # 가장 최소 시간 구하기
    global ans
    if ans == -1 or ans > cur:
        ans = cur
        



          
go(0, 0)
print(ans)
```

    5 1
    2 1 1 1 1
    2 2 2 1 1 
    2 2 2 1 1 
    2 2 2 1 1 
    2 1 1 1 1 
    0

