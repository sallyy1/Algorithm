## 4577번: 소코반
## https://www.acmicpc.net/problem/4577




```
소코반은 1982년에 일본에서 만들어진 게임으로, 일본어로 창고지기라는 뜻이다.
이 게임은 캐릭터를 이용해 창고 안에 있는 박스를 모두 목표점으로 옮기는 게임이다.
목표점의 수와 박스의 수는 같다.
플레이어는 화살표(위, 아래, 왼쪽, 오른쪽)를 이용해 캐릭터를 아래와 같은 규칙으로 조정할 수 있다.


입력으로 주어지는 모든 데이터는 항상 캐릭터가 한 명이고, 박스의 수와 목표점의 수는 같다.
또, 목표점 위에 올라가 있지 않은 박스는 적어도 한 개 이며, 가장 바깥쪽 칸은 항상 벽이다.
```




```python
def is_complete(a):
  goal = 0
  box = 0

  # 모든 목표점 위에 박스가 올려져 있는 'B' 상태만 있어야 함
  for row in a:
    for elem in row:
      if elem == '+':
        goal += 1

      elif elem == 'b':
        box += 1

  return goal == 0 and box == 0 # True



# 게임 여러번 실행
tc = 1
while True:
  n, m = map(int, input().split())

  if n == m == 0:
    break # 게임 종료 시그널


  # 입력 지도 배열
  a = [list(input()) for row in range(n)]

  x = 0
  y = 0
  for i in range(n):
    for j in range(m):
      if a[i][j] in 'wW':
        x, y = i, j # 캐릭터의 시작 위치 찾기
        break




  # 입력 이동 키
  s = input() # ULRURDDDUULLDDD

  for char in s:
    # 이동 방향/크기 찾기
    dx = 0
    dy = 0

    if char == 'U': dx = -1
    if char == 'D': dx = 1
    if char == 'L': dy = -1
    if char == 'R': dy = 1

    nx = x+dx
    ny = y+dy


    # 이동 수행
    ## 1) 바로 이동 가능한 경우들
    if a[nx][ny] == '#': # (1)벽
      pass

    elif a[nx][ny] == '.': # (2)빈 공간
      a[nx][ny] = 'w'

      if a[x][y] == 'w':
        a[x][y] = '.'
      elif a[x][y] == 'W':
        a[x][y] = '+'

      x = nx
      y = ny      



    elif a[nx][ny] == '+': # (3)비어있는 목표점
      a[nx][ny] = 'W'

      if a[x][y] == 'w':
        a[x][y] = '.'
      elif a[x][y] == 'W':
        a[x][y] = '+'

      x = nx
      y = ny




    ## 2) 박스를 밀어야 하는 경우들
    ## (x,y) -> (nx,ny) -> (nx2,ny2)
    elif a[nx][ny] in 'bB': # (4)박스
      nx2 = nx + dx
      ny2 = ny + dy
      ok = True

      if a[nx2][ny2] in '#bB':
        ok = False # 못 미는 경우 제외

      if ok:
        if a[nx2][ny2] == '.':
          a[nx2][ny2] = 'b'
        elif a[nx2][ny2] == '+':
          a[nx2][ny2] = 'B'


        if a[nx][ny] == 'b':
          a[nx][ny] = 'w'
        elif a[nx][ny] == 'B':
          a[nx][ny] = 'W'


        if a[x][y] == 'w':
          a[x][y] = '.'
        elif a[x][y] == 'W':
          a[x][y] = '+'


        x = nx
        y = ny



    if is_complete(a) == True: # (게임이 끝난 뒤 입력되는 키는 무시한다.)
      break

  # 출력
  print('Game %d: %s' %(tc, 'complete' if is_complete(a) else 'incomplete'))

  for row in a:
    print(''.join(row))

  tc += 1
    
```

    8 9
    #########
    #...#...#
    #..bb.b.#
    #...#w#.#
    #...#b#.#
    #...++++#
    #...#..##
    #########
    ULRURDDDUULLDDD
    Game 1: imcomplete
    #########
    #...#...#
    #..bb...#
    #...#.#.#
    #...#.#.#
    #...+W+B#
    #...#b.##
    #########
    0 0


## 2064번: IP 주소
## https://www.acmicpc.net/problem/2064




```
예를 들어 네트워크 주소가 194.85.160.176이고, 네트워크 마스크가 255.255.255.248인 경우를 생각해 보자.
이 경우, 이 네트워크에는 194.85.160.176부터 194.85.160.183까지의 8개의 IP 주소가 포함된다.

어떤 네트워크에 속해있는 IP 주소들이 주어졌을 때, 네트워크 주소와 네트워크 마스크를 구해내는 프로그램을 작성하시오.
답이 여러 개인 경우에는 가장 크기가 작은(포함되는 IP 주소가 가장 적은, 즉 m이 최소인) 네트워크를 구하도록 한다.
```




```python
s = '194.85.160.177'

a = s.split('.') # ['194', '85', '160', '177']
a = [bin(int(x))[2:].zfill(8) for x in a]

a = ''.join(a)


aa = [''.join(a[8*i : 8*i+8]) for i in range(4)]
aa = [int(x, 2) for x in aa]

aa = '.'.join(map(str, aa))

print(a, aa)
```

    11000010010101011010000010110001 194.85.160.177



```python
# 입력
n = int(input())
ips = [input() for _ in range(n)]


def convert_to_bin(s):
  a = s.split('.') # ['194', '85', '160', '177']
    # x = '85'
    # int(x) = 85
    # bin(int(x)) = '0b1010101'
    # bin(int(x))[2:] = '1010101'
    # bin(int(x))[2:].zfill(8) = '01010101'

  a = [bin(int(x))[2:].zfill(8) for x in a]
  return ''.join(a)



def convert_to_ip(a):
  aa = [''.join(a[8*i : 8*i+8]) for i in range(4)]
  aa = [int(x,2) for x in aa]

  return '.'.join(map(str, aa))


def longest_common_prefix(ips):
  p = -1

  for idx in range(32):
    if len(set(ip[idx] for ip in ips)) == 1:
      p = idx

    else:
      break

  return p




# 출력
ips = [convert_to_bin(ip) for ip in ips]

p_idx = longest_common_prefix(ips)

network_address = [ips[0][i] if i <= p_idx else '0' for i in range(32)]
network_mask = ['1' if i <= p_idx else '0' for i in range(32)]

network_address = convert_to_ip(network_address)
network_mask = convert_to_ip(network_mask)

print(network_address)
print(network_mask)
```


```python
'194.85.160.177'.split('.')
```




    ['194', '85', '160', '177']




```python
3
194.85.160.177
194.85.160.183
194.85.160.178
```

## 3107번: IPv6
## https://www.acmicpc.net/problem/3107




```
IPv6은 길이가 128비트인 차세대 인터넷 프로토콜이다.

IPv6의 주소는 32자리의 16진수를 4자리씩 끊어 나타낸다. 이때, 각 그룹은 콜론 (:)으로 구분해서 나타낸다.

예를 들면, 다음과 같다.
2001:0db8:85a3:0000:0000:8a2e:0370:7334


규칙 1. 각 그룹의 앞자리의 0의 전체 또는 일부를 생략 할 수 있다. 위의 IPv6을 축약하면, 다음과 같다
2001:db8:85a3:0:00:8a2e:370:7334

규칙 2. 만약 0으로만 이루어져 있는 그룹이 있을 경우 그 중 한 개 이상 연속된 그룹을 하나 골라 콜론 2개(::)로 바꿀 수 있다.
2001:db8:85a3::8a2e:370:7334
2번째 규칙은 모호함을 방지하기 위해서 오직 한 번만 사용할 수 있다.

올바른 축약형 IPv6주소가 주어졌을 때, 이를 원래 IPv6 (32자리의 16진수)로 복원하는 프로그램을 작성하시오.
```




```python
ipv6 = input()


parts = ipv6.split(':')

if parts.count("") > 1: # '::1'인 경우 또는 '1::'인 경우
  parts.remove("") # 1개 제거

# 예 2
# ::1
# ["", "", "1"]
# ["", "1"] -> 2개

if len(parts) < 8:
  idx = parts.index("")
  parts[idx:idx+1] = ["0"] * (8-len(parts)+1)


answer = [p.zfill(4) for p in parts]
print(':'.join(answer))
```

    ::1
    0000:0000:0000:0000:0000:0000:0000:0001

