# Section05-3
# 파이썬 흐름제어(제어문)
# 제어문 관련 퀴즈(정답은 영상)

# 1 ~ 5 문제 if 구문 사용
# 1. 아래 딕셔너리에서 '가을'에 해당하는 과일을 출력하세요.
q1 =  {"봄": "딸기", "여름": "토마토", "가을": "사과"}

print("1. ")

for a in q1.keys(): # 그냥 q1: 으로만 반복시켜도 됨 (기본이 key로 반복하기 때문에)
    if a == '가을':
        print(q1[a])
        break

for k, v in q1.items(): # 2)) items() -> key 와 value 를 반복시키고
    if a == '가을':
        print(v) # value 만 출력해도 됨
        break

# 2. 아래 딕셔너리에서 '사과'가 포함되었는지 확인하세요.
q2 =  {"봄": "딸기", "여름": "토마토", "가을": "사과"}

print("\n2. ")

for b in q2.values():
    if b == '사과':
        print("포함되었습니다.")
        break

else: # for - else는 break 와 함께 사용 !
    print("Not found")

# 2)) items() 사용
for k, v in q2.items():
    if v == '사과':
        print(k, v)
        break
else:
    print("사과 없음")


# 3. 다음 점수 구간에 맞게 학점을 출력하세요.
# 81 ~ 100 : A학점
# 61 ~ 80 :  B학점
# 41 ~ 60 :  C학점
# 21 ~ 40 :  D학점
#  0 ~ 20 :  E학점

score = 77

print("\n3. ")

if score >= 81:
    print("A학점")
elif score >= 61:
    print("B학점")
elif score >= 41:
    print("C학점")
elif score >= 21:
    print("D학점")
else:
    print("E학점")


# 4. 다음 세 개의 숫자 중 가장 큰수를 출력하세요.(if문 사용) : 12, 6, 18
arr = [12, 6, 18]
big = arr[0]

for n in arr:
    if(n > big):
        big = n
print("\n4. ", big)


# 2))
a, b, c = 12, 6, 18

best = a

if b > a:
    best = b
if c > b:
    best = c
print('best : ', best)


# 5. 다음 주민등록 번호에서 7자리 숫자를 사용해서 남자, 여자를 판별하세요. (1,3 : 남자, 2,4 : 여자)
s = '891022-2473837'

sex = int(s[-7]) # 문자열이므로 int 로 형 변환 해야 !!

if(sex == 1 or sex == 3):
    print("\n5. 남자 입니다")
elif(sex == 2 or sex == 4):
    print("\n5. 여자 입니다")


# 6 ~ 10 반복문 사용(while 또는 for)

# 6. 다음 리스트 중에서 '정' 글자를 제외하고 출력하세요.
q3 = ["갑", "을", "병", "정"]

print("\n6. ")

for v in q3:
    if v == "정":
        continue # '정' 이면 건너뛰고,
    else:
        print("타입 :", v) # 나머지 값들은 출력


# 7. 1부터 100까지 자연수 중 '홀수'만 한 라인으로 출력 하세요.
print("\n7. ")

for n in range(1,101, 2):
    print(n, end = ' ') # print() 함수의 end인자 활용
print('\n')

# 2))
for n in range(1, 101):
    if n % 2 != 0:
        print(n, end = ',')

# 8. 아래 리스트 항목 중에서 5글자 이상의 단어만 출력하세요.
q4 = ["nice", "study", "python", "anaconda", "!"]

for s in q4:
    if len(s) >= 5:
        print("8. ", s)
print()


# 9. 아래 리스트 항목 중에서 소문자만 출력하세요.
q5 = ["A", "b", "c", "D", "e", "F", "G", "h"]

for l in q5:
    if l.islower():
        print("9. ", l)
print()


# 10. 아래 리스트 항목 중에서 소문자는 대문자로 대문자는 소문자로 출력하세요.
q6 = ["A", "b", "c", "D", "e", "F", "G", "h"]

for z in q6:
    if z.islower():
        print("10. ", z.upper())
    else:
        print("10. ", z.lower())


# 리스트 컴프리헨션 (list comprehension)
numbers = []

for n in range(1,101):
    numbers.append(n)
print(numbers)



numbers2 = [x for x in range(1, 101)] # 더 직관적이고 빠르다 ! (내부적으로 순회하는 방법)
print(numbers2)


# 6번 변형
q5 = [x for x in q3 if x != '정']
print(q5)

# 7번 변형
q6 = [x for x in range(1,101) if x %2 != 0]
print(q6)

# 구조
x = [x for x in range(1,100) if 조건문]


