# Section04-3
# 파이썬 데이터 타입(자료형)
# 리스트, 튜플

# 리스트(순서o, 중복o, 수정o, 삭제o)

# 선언 -> 대괄호
a = []
b = list() # 명시적 방법
c = [1,2, 3, 4]
d = [10, 100, 'Pen', 'Banana', 'Orange'] # 타 언어와 달리 데이터타입이 달라도 같은 문자열 list 에 담을 수 있음(유연)
e = [10, 100, ['Pen', 'Banana', 'Orange']] # list of list 도 가능함(유연)

# 인덱싱
print(d[3])
print(d[-2])
print(d[0] + d[1])
print(e[2][1])
print(e[-1][-2])

# 슬라이싱
print(d[0:3])
print(d[0:1])
print(d[0:2])
print(e[2][1:3])

# 연산
print(c + d) # 리스트 확장
print(c * 3) # 리스트 길이 3배 늘어남
print(str(c[0])+'hi')

# 리스트 수정, 삭제
c[0] = 77
print(c)

c[1:2] = [100, 1000, 10000]
print(c)
c[1] = ['a', 'b', 'c'] # 중첩 (nasted)
print(c)


del c[1]
print(c)

del c[-1]
print(c)

print()
print()
print()

# 리스트 함수
y = [5, 2, 3, 1, 4]
print(y)

y.append(6) # append() 함수
print(y)

y.sort() # sort() 정렬 함수
print(y)

y.reverse() # reverse() 함수
print(y)

y.insert(2, 7) # insert() 함수
print(y)

y.remove(2) # remove() 함수 -> 삭제를 원하는 값을 입력
print(y)

# 차이)) del y[1] -> 삭제를 원하는 index를 입력

y.remove(7)
print(y)

y.pop() # pop() 함수 -> 맨 마지막에 있는 원소를 꺼내고 없애는 함수
# 주의)) pop() 을 쓰다보면 언젠가는 에러가 난다!! (값이 없기 때문)
print(y) # LIFO

ex = [88, 77]
y.extend(ex) # extent() 함수 -> 연장(끝 부분에 원소자체를 추가함)
print(y)

#y.append(ex) # 차이)) 끝 부분에 list로 추가 됨
#print(y)

# 삭제 : del, remove, pop

# 튜플 (순서o, 중복o, 수정x, 삭제x) -> 프로그램 상 중요한 key 값/데이터 (예: 계좌번호)
a = () # 괄호
b = (1,) # ,로 끝냄
c = (1, 2, 3, 4)
d = (10, 100, ('a', 'b', 'c')) # 튜플 안에 튜플 선언

print(c[2])
print(c[3])
print(d[2][1]) # 리스트와 인덱싱은 같음

print(d[2:])
print(d[2][0:2]) # 슬라이싱도 가능

print(c + d)
print(c * 3) # 확장도 가능

print()
print()

# 튜플 함수
z = (5, 2, 1, 3, 1)

print(z)
print(3 in z)
print(z.index(5)) # index() 함수
print(z.count(1)) # count() 함수