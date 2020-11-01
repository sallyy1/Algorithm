# Section04-5
# 파이썬 데이터 타입(자료형)
# 딕셔너리, 집합 자료형
# 데이터 타입 관련 퀴즈(정답은 영상)

# 1. 아래 문자열의 길이를 구해보세요.
q1 = "dk2jd923i1jdk2jd93jfd92jd918943jfd8923"
print(len(q1)) # len() 함수 !
print("1. ", len(q1))

# 2. print 함수를 사용해서 아래와 같이 출력해보세요.
#    apple;orange;banana;lemon
print('apple;orange;banana;lemon')
print("2. ", """apple;orange;banana;lemon""")

# 3. 화면에 * 기호 100개를 표시하세요.
print('*' * 100) # 문자열도 연산이 가능하다 !
print("3. ", '*' * 100)

# 4. 문자열 "30" 을 각각 정수형, 실수형, 복소수형, 문자형으로 변환해보세요.
str = "30"
print(int(str))
print(float(str)) # 실수형
print(complex(str)) # 복소수
str2 = int(str)
print(repr(str2)) # 문자형 -> repr(숫자(int형)) 함수 ??

# 5. 다음 문자열 "Niceman" 에서 "man" 문자열만 추출해보세요.
str5 = "Niceman"
print(str5[-3:])
print("5. ", str5[4:7]) # 2) 7 까지 출력해야 n까지 나옴

str5_idx = str5.index("man") # 3) index() 함수 -> 내가 찾고자 하는 요소의 시작 인덱스를 return 해주는 함수 !
print("5. ", str5[str5_idx : str5_idx+3])

# 6. 다음 문자열을 거꾸로 출력해보세요. : "Strawberry"
str6 = "Strawberry"
print(str6[::-1]) # 전체 범위의 문자를 -1씩 점프하면서 출력
print(list(reversed(str6))) # 2) reversed() 함수는 꼭 list() 형 변환을 해주어야 함! -> 근데, 각 요소값을 하나씩 출력해줌.

# 7. 다음 문자열에서 '-'를 제거 후 출력하세요. : "010-7777-9999"
str7 = "010-7777-9999"
print(str7.replace("-", ""))# 제거 -> replace("old 값", "new 값") 함수를 이용 !

print("7. ", str7[0:3]+str7[4:8]+str7[9:13]) # 2) 슬라이싱 이용

import re # 3) 정규표현식 이용
print("7. ", re.sub('[^0-9]', '', str7)) # re.sub() 함수 -> 0부터 9가 아니면(^) => ''으로 대체한다 !


# 8. 다음 문자열(URL)에서 "http://" 부분을 제거 후 출력하세요. : "http://daum.net"
str8 = "http://daum.net"
print(str8.replace("http://", ""))

print("8. ", str8[7:]) # 2) 슬라이싱 이용

# 9. 다음 문자열을 모두 대문자, 소문자로 각각 출력해보세요. : "NiceMan"
str9 = "NiceMan"
print(str9.upper()) # upper() 함수
print(str9.lower()) # lower() 함수

# 10. 다음 문자열을 슬라이싱을 이용해서 "cde"만 출력하세요. : "abcdefghijklmn"
str10 = "abcdefghijklmn"
print(str10[2:5])

# 11. 다음 리스트에서 "Apple" 항목만 삭제하세요. : ["Banana", "Apple", "Orange"] (???)
list11 = ["Banana", "Apple", "Orange"]
# del list11[1] # 인덱스 입력
# print(list11)

list11.remove("Apple") # 2) remove() 함수 이용 -> 값 입력
print("11. ", list11) 
# TypeError: descriptor 'remove' for 'list' objects doesn't apply to a 'str' object


# 12. 다음 튜플을 리스트로 변환하세요. : (1,2,3,4)
t12 = (1,2,3,4)  # 튜플 -> 수정x, 삭제x
print(list(t12))

# 13. 다음 항목을 딕셔너리(dict)으로 선언해보세요. : <성인 - 100000 , 청소년 - 70000 , 아동 - 30000>
dic13 = {'성인' : 100000, '청소년' : 7000, '아동' : 3000}

# 선언 방법 2)
dic2 = {}
dic2['성인'] = 100000
dic2['청소년'] = 7000
dic2['아동'] = 3000

print("13. ", dic2)

# 14. 13번 에서 선언한 dict 항목에 <소아 - 0> 항목을 추가해보세요.
dic13['소아'] = 0

# 15. 13번에서 선언한 딕셔너리(dict)에서 Key 항목만 출력해보세요.
print(dic13.keys())
print(list(dic13.keys())) # list 로 형 변환 해야 함

# 16. 13번에서 선언한 딕셔너리(dict)에서 value 항목만 출력해보세요.
print(dic13.values())
print(list(dic13.values())) # list 로 형 변환 해야 함

print(tuple(dic13.values())) # tuple 로도 물론 형 변환 할 수 있음 !

# *** 결과 값만 정확하게 출력되면 됩니다. ^^* 고생하셨습니다. ***
