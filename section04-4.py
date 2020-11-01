# Section04-4
# 파이썬 데이터 타입(자료형)
# 딕셔너리, 집합 자료형

# 딕셔너리(Dictionary): 순서x, 중복x, 수정o, 삭제o

# key, value (Json) -> MongoDB
# 선언
a = {'Name' : 'Kim', 'Phone' : '010-7777-7777', 'Birth' : '870214'}
b = {0: 'Hello Python', 1 : 'Hello Coding'} # key는 숫자도 가능하나, 주로 사용되진 않음 (주로 의미있는 값이 사용됨)
c = {'arr' : [1, 2, 3, 4, 5]}

print(type(a))

# 출력
print(a['Name']) # 조회는 Key 로 조회
print(a.get('Name')) # get() 메소드 추천(안전!)
print(a.get('address')) # address가 없어도 None이라고 뜨지 오류가 뜨지 않음
print(c['arr'][1:3])

# 딕셔너리 추가
a['address'] = 'Seoul'
print(a)
a['rank'] = [1, 3, 4]
a['rank2'] = (1, 2, 3)
print(a)

# keys, values, item
print(a.keys()) # keys() 함수 -> key 만 list 형태로 가져옴
print(list(a.keys())) # 형 변환

temp = list(a.keys())
print(temp[1:3])

print(a.values()) # values() 함수 -> value 만 list 형태로 가져옴
print(list(a.values()))

print(a.items()) # items() 함수 -> 전체 를 list 내 튜플 형태로 보여줌
print(list(a.items()))

print(2 in b)
print('name' in a)
print('name2' not in a)


# 집합(sets) : 순서x, 중복x
a = set()
b = set([1, 2, 3, 4])
c = set([1, 4, 5, 6, 6, 7]) # 중복을 허용하지 않음 !

print(type(a))
print(c)

t = tuple(b) # 튜플로 형 변환
print(t)

l = list(b) # 리스트로 형 변환
print(l)

s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])

print(s1.intersection(s2)) # intersection(), & -> 교집합 구하기
print(s1 & s2)

print(s1 | s2) # union(), | -> 합집합 구하기
print(s1.union(s2))

print(s1 - s2) # difference(), - -> 합집합 구하기
print(s1.difference(s2))

# 추가 & 제거
s3 = set([7, 8, 10, 15])

s3.add(18) # add() 함수
s3.add(7)

print(s3)

s3.remove(15) # remove() 함수
print(s3)

print(type(s3))

