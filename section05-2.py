# Section05-2
# 파이썬 흐름제어(반복문)
# 반복문 실습

# 코딩의 핵심 -> 조건 해결 중요

# 기본 반복문 : for, while

v1 = 1
while v1 < 11:
    print("v1 is :", v1)
    v1 += 1

for v2 in range(10): # 0 부터 시작해서 10 미만 값인, 9 까지 반복 !
    print("v2 is : ", v2)

for v3 in range(1, 11): # 시작값 ~ 끝값-1 까지 반복
    print("v3 is : ", v3)


# 1 ~ 100 합

sum1 = 0
cnt1 = 1

while cnt1 <= 100:
    sum1 += cnt1
    cnt1 += 1

print('1 ~ 100 : ', sum1)

print('1 ~ 100 : ', sum(range(1, 101))) # 방법 2) range(시작값, 끝값, (증감 단위 값)) 함수 -> 효율적 방법 !

print('1 ~ 100 : ', sum(range(1, 101, 2))) # 2 씩 건너 뛰어라 ! -> 짝수의 합 구할 수 있음


# 시퀀스(순서가 있는) 자료형 반복
# 문자열, 리스트, 튜플,(순서o) // 집합, 사전 (순서는 x지만 반복은 가능)
# iterable 리턴 함수 : range, reversed, enumerate, filter, map, zip

names = ["Kim", "Park", "Cho", "Choi", "You"]

for v in names:
    print("You are : ", v)



word = "dreams" # 문자열은 immutable(한번 할당이 되면 수정이 불가능) -> 이유: 순서가 있고 & 한 단어 한단어가 다 자기 공간에 맞게 할당되어 있기 때문에

for s in word:
    print("Word : ", s)


my_info = {
    "name" : "Kim",
    "age" : 33,
    "city" : "Seoul"
}

# 기본 값은 키 !
for key in my_info:
    print("my_info", key)

# 값
for key in my_info.values():
    print("my_info", key)

# 키
for key in my_info.keys():
    print("my_info", key)

# 키 and 값
for k, v in my_info.items():
    print("my_info", k, v)


# 대문자 -> 소문자, 소문자 -> 대문자 바꾸는 예제
name = "KennRY"

for n in name:
    if n.isupper():
        print(n.lower())
    else:
        print(n.upper())


# break
numbers = [14, 3, 4, 7, 10, 24, 17, 2, 37, 15, 34, 36, 38]


# for - else 구문(반복문이 정상적으로 수행된 경우, else 블럭 수행 !)
for num in numbers:
    if num == 33:
        print("found : 33!")
        break
    else:
        print("not found : 33!")

else: # break 거치치 않게 모든 경우의 반복문을 다 수행했을 때
    print("Not found 33......")



# continue
lt = ["1", 2, 5, True, 4.3, complex(4)]

for v in lt:
    if type(v) is float:
        print("찾았다!")


for v in lt:
    if type(v) is float:
        continue # 4.3은 건너뛰고 나머지 값들의 type만 출력하는 프로그램
    print("타입 :", type(v))


name3 = "Niceman"
print(list(reversed(name3)))
print(tuple(reversed(name3)))
