- 브루트포스 **순서** 문제 : O(N!)
- 브루트포스 **선택** 문제 : O(2**n)


```python
# 오름차순 선택 연습 (재귀 x, for문)

N , M = 5, 3

for i in range(1, N+1-2):
    for j in range(i+1, N+1-1):
        for k in range(j+1, N+1):
            print(str(i)+str(j)+str(k) + '\n')
```

    123
    
    124
    
    125
    
    134
    
    135
    
    145
    
    234
    
    235
    
    245
    
    345
    


# 15649번: N과 M (1)
#### 1부터 N까지 자연수 중에서 중복 없이 M개를 고른 수열


```python
## 15649번: N과 M (1)
#### 1부터 N까지 자연수 중에서 중복 없이 M개를 고른 수열


## (순서) 문제
import sys

n, m = map(int, input().split())

c = [False]*(n+1)
a = [0] * m # m자릿수만큼 담길 수열 (= 출력해야 하는 내용)

def go(index, n, m):
    if index == m: # 종료조건 (0 - M-1 번 index를 출력해야 M개가 되므로)
        print(' '.join(map(str, a)))
        #sys.stdout.write(' '.join(map(str, a)) + '\n')
        return

    for i in range(1, n+1):
        if c[i] == True: # 앞에서 사용한 수라면,
            continue # pass

        c[i] = True # 현재 인덱스 값 사용했으니까 '사용함'으로 바꿔줌
        a[index] = i

        go(index+1, n, m) # 다음 재귀 진행
        c[i] = False # 다시 '사용안함'으로 되돌려놓기 (index번째 위치에 숫자 i가 온 이후의 모든 일들은 go() 함수에서 처리가 이미 끝났기에 => 현재 i가 아닌 다른 i가 와야 함)

go(0, n, m)
```

    4 2
    1 2
    1 3
    1 4
    2 1
    2 3
    2 4
    3 1
    3 2
    3 4
    4 1
    4 2
    4 3



```python
' '.join(map(str, a))
```




    '3'




```python
import sys

sys.stdin.readline().rstrip
```




    <function str.rstrip>



# 15650번: N과 M (2)
#### 1부터 N까지 자연수 중에서 중복 없이 M개를 고른 수열
#### 고른 수열은 오름차순이어야 한다.


```python
## 15650번: N과 M (2)
#### 1부터 N까지 자연수 중에서 중복 없이 M개를 고른 수열
#### 고른 수열은 오름차순이어야 한다.


## (순서) 문제 

# 재귀함수에 'start' 인자 추가됨
import sys

n, m = map(int, input().split())

c = [False]*(n+1)
a = [0] * m # m자릿수만큼 담길 수열 (= 출력해야 하는 내용)

def go(index, start, n, m):
    if index == m: # 종료조건 (0 - M-1 번 index를 출력해야 M개가 되므로)
        ##print(' '.join(map(str, a)))
        sys.stdout.write(' '.join(map(str, a)) + '\n')
        return

    for i in range(start, n+1): # i의 시작은 start ~
        if c[i] == True: # 앞에서 사용한 수라면,
            continue # pass

        c[i] = True # 현재 인덱스 값 사용했으니까 '사용함'으로 바꿔줌
        a[index] = i

        go(index+1, i+1, n, m) # 다음 재귀 진행
        c[i] = False # 다시 '사용안함'으로 되돌려놓기 (index번째 위치에 숫자 i가 온 이후의 모든 일들은 go() 함수에서 처리가 이미 끝났기에 => 현재 i가 아닌 다른 i가 와야 함)

go(0, 1, n, m)
```

    4 2
    1 2
    1 3
    1 4
    2 3
    2 4
    3 4



```python
# 재귀함수에 'start' 인자 추가됨
# 'start'가 있으므로 앞에서 사용한 수를 담고 있는 배열 c[True/False]는 제거 가능

## (순서) 문제 


import sys

n, m = map(int, input().split())

##c = [False]*(n+1)
a = [0] * m # m자릿수만큼 담길 수열 (= 출력해야 하는 내용)

def go(index, start, n, m):
    if index == m: # 종료조건 (0 - M-1 번 index를 출력해야 M개가 되므로)
        print(' '.join(map(str, a)))
        #sys.stdout.write(' '.join(map(str, a)) + '\n')
        return

    for i in range(start, n+1): # i의 시작은 start ~
        ##if c[i] == True: # 앞에서 사용한 수라면,
          ##continue # pass

        ##c[i] = True # 현재 인덱스 값 사용했으니까 '사용함'으로 바꿔줌
        a[index] = i

        go(index+1, i+1, n, m) # 다음 재귀 진행
        ##c[i] = False # 다시 '사용안함'으로 되돌려놓기 (index번째 위치에 숫자 i가 온 이후의 모든 일들은 go() 함수에서 처리가 이미 끝났기에 => 현재 i가 아닌 다른 i가 와야 함)

go(0, 1, n, m)
```

    4 2
    1 2
    1 3
    1 4
    2 3
    2 4
    3 4



```python
## (선택) 문제로 접근

import sys

n, m = map(int, input().split())

a = [0] * m # m자릿수만큼 담길 수열 (= 출력해야 하는 내용)

## index : 수 index
## selected: 선택한 수의 개수

def go(index, selected, n, m):
    if selected == m: # 종료조건 (m개의 수를 이미 모두 다 선택했을 시)
        sys.stdout.write(' '.join(map(str, a)) + '\n')
        return

    if (index > n): return # 종료조건 (아직 m개를 못 골랐는데 더 이상 선택할 수 없는 경우)


    # 1) 해당 수 index를 선택할 경우
    a[selected] = index
    go(index+1, selected+1, n, m) # 다음 재귀 진행

    # 2) 해당 수 index를 선택하지 않을 경우
    a[selected] = 0
    go(index+1, selected, n, m) # 다음 재귀 진행


go(1, 0, n, m)
```

    4 2
    1 2
    1 3
    1 4
    2 3
    2 4
    3 4


# 15651번: N과 M (3)
#### 1부터 N까지 자연수 중에서 M개를 고른 수열
#### 같은 수를 여러 번 골라도 된다. (중복 허용)


```python
## 15651번: N과 M (3)
#### 1부터 N까지 자연수 중에서 M개를 고른 수열
#### 같은 수를 여러 번 골라도 된다. (중복 허용)


## (순서) 문제
import sys

n, m = map(int, input().split())

##c = [False]*(n+1)
a = [0] * m # m자릿수만큼 담길 수열 (= 출력해야 하는 내용)

def go(index, n, m):
    if index == m: # 종료조건 (0 - M-1 번 index를 출력해야 M개가 되므로)
        print(' '.join(map(str, a)))
        #sys.stdout.write(' '.join(map(str, a)) + '\n')
        return

    for i in range(1, n+1):
        ##if c[i] == True: # 앞에서 사용한 수라면,
        ##  continue # pass

        ##c[i] = True # 현재 인덱스 값 사용했으니까 '사용함'으로 바꿔줌
        a[index] = i

        go(index+1, n, m) # 다음 재귀 진행
        ##c[i] = False # 다시 '사용안함'으로 되돌려놓기 (index번째 위치에 숫자 i가 온 이후의 모든 일들은 go() 함수에서 처리가 이미 끝났기에 => 현재 i가 아닌 다른 i가 와야 함)

go(0, n, m)
```

    4 2
    1 1
    1 2
    1 3
    1 4
    2 1
    2 2
    2 3
    2 4
    3 1
    3 2
    3 3
    3 4
    4 1
    4 2
    4 3
    4 4


# 15651번: N과 M (4)
#### 1부터 N까지 자연수 중에서 M개를 고른 수열
#### 같은 수를 여러 번 골라도 된다. (중복 허용)
#### 고른 수열은 비내림차순이어야 한다. (길이가 K인 수열 A가 A1 ≤ A2 ≤ ... ≤ AK-1 ≤ AK를 만족하면, 비내림차순이라고 한다.)



```python
## 15651번: N과 M (4)
#### 1부터 N까지 자연수 중에서 M개를 고른 수열
#### 같은 수를 여러 번 골라도 된다. (중복 허용)
#### 고른 수열은 비내림차순이어야 한다. (길이가 K인 수열 A가 A1 ≤ A2 ≤ ... ≤ AK-1 ≤ AK를 만족하면, 비내림차순이라고 한다.)

## start를 (i+1 ~ N) 말고 (i ~ N) 으로 변경해줌 !

## (순서) 문제 

import sys

n, m = map(int, input().split())

##c = [False]*(n+1)
a = [0] * m # m자릿수만큼 담길 수열 (= 출력해야 하는 내용)

def go(index, start, n, m):
    if index == m: # 종료조건 (0 - M-1 번 index를 출력해야 M개가 되므로)
        print(' '.join(map(str, a)))
        #sys.stdout.write(' '.join(map(str, a)) + '\n')
        return

    for i in range(start, n+1): # i의 시작은 start ~
        ##if c[i] == True: # 앞에서 사용한 수라면,
          ##continue # pass

        ##c[i] = True # 현재 인덱스 값 사용했으니까 '사용함'으로 바꿔줌
        a[index] = i

        go(index+1, i, n, m) # 다음 재귀 진행
        ##c[i] = False # 다시 '사용안함'으로 되돌려놓기 (index번째 위치에 숫자 i가 온 이후의 모든 일들은 go() 함수에서 처리가 이미 끝났기에 => 현재 i가 아닌 다른 i가 와야 함)

go(0, 1, n, m)
```

    3 3
    1 1 1
    1 1 2
    1 1 3
    1 2 2
    1 2 3
    1 3 3
    2 2 2
    2 2 3
    2 3 3
    3 3 3



```python
## (선택) 문제로 접근

import sys

n,m = map(int,input().split())
cnt = [0]*(n+1)

def go(index, selected, n, m):
    if selected == m:
        for i in range(1, n+1):
            for j in range(cnt[i]):
                sys.stdout.write(str(i)+' ')
        sys.stdout.write('\n')
        return
    if index > n:
        return
    for i in range(m-selected, 0, -1):
        cnt[index] = i
        go(index+1, selected+i, n, m)
    cnt[index] = 0
    go(index+1, selected, n, m)

go(1,0,n,m)
```

    4 2
    1 1 
    1 2 
    1 3 
    1 4 
    2 2 
    2 3 
    2 4 
    3 3 
    3 4 
    4 4 


# 18290번: NM과 K (1)
#### 크기가 N×M인 격자판의 각 칸에 정수가 하나씩 들어있다. 이 격자판에서 칸 K개를 선택할 것이고, 선택한 칸에 들어있는 수를 모두 더한 값의 최댓값을 구하려고 한다. 단, 선택한 두 칸이 인접하면 안된다. r행 c열에 있는 칸을 (r, c)라고 했을 때, (r-1, c), (r+1, c), (r, c-1), (r, c+1)에 있는 칸이 인접한 칸이다.
#### 선택한 칸(K개)에 들어있는 수를 모두 더한 값의 최댓값을 출력한다.


```python
## 18290번: NM과 K (1)
#### 크기가 N×M인 격자판의 각 칸에 정수가 하나씩 들어있다. 이 격자판에서 칸 K개를 선택할 것이고, 선택한 칸에 들어있는 수를 모두 더한 값의 최댓값을 구하려고 한다. 단, 선택한 두 칸이 인접하면 안된다. r행 c열에 있는 칸을 (r, c)라고 했을 때, (r-1, c), (r+1, c), (r, c-1), (r, c+1)에 있는 칸이 인접한 칸이다.
#### 선택한 칸(K개)에 들어있는 수를 모두 더한 값의 최댓값을 출력한다.

# 중복 선택을 피하기 위해 '오름차순(비내림차순)' 방법을 활용
# (start - N) 개념


n, m, k = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]

c = [[False]*m for _ in range(n)] # 방문(선택) 여부 체크하는 배열
ans = -2147483647

dx = [0, 0, 1, -1]
dy = [1, -1, 0,0]

def go(px, py, cnt, s): # (px, py) : 이전에 선택한 칸의 행과 열 기억
    if cnt == k:
        global ans
        if ans < s: # 최댓값 교체
            ans = s
        
        return


    for x in range(px, n): # start - N 개념
        for y in range(py if x==px else 0, m): # start - N 개념
            if c[x][y] == True: # 이미 선택했던 칸이라면,
                continue # pass

            ok = True
            for i in range(4):
                nx, ny = x+dx[i], y+dy[i]
                if 0<= nx <n and 0<= ny <m:
                    if c[nx][ny] == True: # 이미 선택했던 칸이라면,
                        ok = False # 선택 불가능 표시


            if ok == True: # 선택이 가능한 상황이라면,
                c[x][y] = True # 방문처리

                go(x, y, cnt+1, s+a[x][y]) # (x, y)에서 출발?
                c[x][y] = False # (x, y) 방문 여부 다시 되돌려놓기



go(0, 0, 0, 0)
print(ans)

```

    5 5 3
    1 9 8 -2 0
    -1 9 8 -3 0
    -5 1 9 -1 0
    0 0 0 9 8
    9 9 9 0 0
    27


- 1차원이라고 가정한 풀이법 (row-major order)
- (r,c) = r*M(열의 개수) + c 로 나타낼 수 있음을 이용하여 구현


```python
# (r,c) = r*M(열의 개수) + c

n, m, k = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]

c = [[False]*m for _ in range(n)] # 방문(선택) 여부 체크하는 배열
ans = -2147483647

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def go(prev, cnt, s):
    if cnt == k:
        global ans
        if ans < s: # 최댓값 교체
            ans = s
        
        return


    for j in range(prev+1, n*m): # start - N 개념 (오름차순)
        x = j // m
        y = j % m

        if c[x][y] == True: # 이미 선택했던 칸이라면,
            continue # pass


        ok = True
        for i in range(4):
            nx, ny = x+dx[i], y+dy[i]
            if 0<= nx <n and 0<= ny <m:
                if c[nx][ny] == True: # 이미 선택했던 칸이라면,
                    ok = False # 선택 불가능 표시
                    break


        if ok == True: # 선택이 가능한 상황이라면,
            c[x][y] = True # 방문처리

            go(x*m + y, cnt+1, s+a[x][y]) # x*m + y 번째 1차원 칸에서 출발?
            c[x][y] = False # (x, y) 방문 여부 다시 되돌려놓기



go(-1, 0, 0)
print(ans)
```

    2 2 2
    5 4
    4 5
    10

