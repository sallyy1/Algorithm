# 데이터 타입
'''
int : 정수
float : 실수
complex : 복소수
bool : 불린
str : 문자열(시퀀스)
list : 리스트(시퀀스)
tuple : 튜플(시퀀스)
set : 집합
dict : 사전

bytearray
byte
frozenset
'''

v_str1 = "Niceman"
v_bool = True
v_str2 = "Goodboy"
v_float = 10.3
v_int = 7
v_dict = {
    "name" : "Kim",
    "age" : 25
}
v_list = [3, 5, 7] # 리스트 = 배열
v_tuple = 3, 5, 7
v_set = {7, 8, 9}

print(type(v_str1))
print(type(v_str2))
print(type(v_tuple))
print(type(v_set))
print(type(v_float))
print(type(v_int))
print(type(v_list))
print(type(v_bool))

i1 = 39
i2 = 939
big_int = 999999999999999999999999999999999999999999999999999999
big_int = 777777777777777777777777777777777777777777777777777777
f1 = 1.234
f2 = 3.784
f3 = .5
f4 = 10.

print(i1 * i2)
print(big_int * big_int)
print(f1 ** f2)
print(f3 + i2)

result = f3 + i2
print(result, type(result))

a = 5.
b = 4
c = 10

print(type(a), type(b))
result2 = a + b
print(result2)

# 형 변환
# int, float, complex(복소수)

print(int(result2))
print(float(c))
print(complex(3))
print(int(True))
print(int(False))
print(int('3'))
print(complex(False))


# 단항 연산자
y = 100
y += 100 # *=
print(y)


# 수치 연산 함수
# https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex

print(abs(-7)) # 절댓값
n, m = divmod(100, 8) # a를 b로 나눈 몫 & 나머지
print(n, m)

import math

print(math.ceil(5.1)) # 5.1보다 가장 가까운 큰 정수
print(math.floor(3.874)) # 3.874보다 가장 가까운 작은 정수

"""
+ 
- 
* 
/ 
// : 몫 
% : 나머지
abs(x) 
int(x) 
float(x) 
complex(x)
pow(x, y) 
x ** y : 제곱
....
"""
