# section04-2
# 문자열, 문자열연산, 슬라이싱

str1 = "I am a boy"
str2 = 'Niceman'
str3 = ''
str4 = str()

print(len(str1), len(str2), len(str3), len(str4))

# 이스케이프
escape_str1 = "Do you have a \"big collection\""
print(escape_str1)

escape_str2 = "Tab\tTab\t"
print(escape_str2)

# Raw String (중요)
raw_s1 = r'C:\Programs\Test\Bin'
print(raw_s1)
raw_s2 = r"\\a\\a"
print(raw_s2)

# 멀티라인
multi = \
"""
문자열 
멀티라인 
테스트
"""
print(multi)

# 문자열 연산
str_o1 = '*'
str_o2 = 'abc'
str_o3 = 'def'
str_o4 = "Niceman"

print(str_o1 * 100)
print(str_o2 + str_o3)
print(str_o1 * 3)
print('a' in str_o4)
print('f' in str_o4)
print('z' not in str_o4)

# 문자열 형 변환
print(str(77) + 'a')
print(str(10.4))

# 문자열 함수
# 참고 : https://www.w3schools.com/python/python_ref_string.asp

# a = 'niceman'
# b = 'orange'

# print(a.islower())
# print(b.endswith('e')) # 마지막 글자가 해당 값?
# print(a.capitalize()) # 첫 글자만 대문자로 바꿈
# print(a.replace('nice', 'good'))# 특정 위치만 원하는 값으로 대체하기
# print(reversed(b))
# print(list(reversed(b))) # 역순으로 변환하는 함수 (꼭, list로 형 변환을 해야 함!!!)


# 문자열은 한번 선언하면 바꿀 수 없다.(immutable(수정불가능한 자료형))
# 슬라이싱 처리
a = 'niceman'
b = 'orange'

print(a[0:3])
print(a[0:4])
print(a[0:len(a)])
print(a[:4])

print(b[0:4:2]) # 해당 갯수만큼 점프(생략)하면서 슬라이싱
print(b[1:-2]) # 맨 마지막 값이 '-1' (순회) /// 정방향은 '0' 부터 시작
print(b[::-1]) # reverse와 같은 효과
