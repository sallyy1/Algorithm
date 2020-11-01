# Section02-1
# 파이썬 기초 코딩
# print 구문의 이해

# 기본출력
print('Hello Python!')
print("Hello Python!")
print("""Hello Python!""")
print('''Hello Python!''')

print() # 기본으로 줄바꿈이 한번 됨.

# Separator 옵션 사용
print('T', 'E', 'S', 'T', sep='') # 글자 사이를 ''(공백) 으로 붙여서 연결해줌
print('2019', '02', '19', sep='-') # 글자 사이를 '-'로 이어서 연결해줌
print('niceman', 'google.com', sep='@')

# end 옵션 사용
print('Welcome To', end='') # 끝을 공백으로 처리하겠다는 옵션 -> 줄바꿈이 되지 않고 끝이 다음문장과 이어짐
print(' the black parade', end='')
print(' piano notes')

# format 사용 [], {}, ()
print('{} and {}'.format('You', 'Me'))
print("{0} and {1} and {0}".format('You', 'Me'))
print("{a} are {b}".format(a='You', b='Me'))

print("%s's favorite number is %d" %('Eunki', 7)) # %s : 문자, %d : 정수, %f : 실수


print("Test1: %5d, Price: %4.2f" %(776, 6534.123)) # 5 자릿수의 정수
print("Test1: {0: 5d}, Price: {1: 4.2f}".format(776, 6534.123)) # 중괄호{} 로 묶을 때는 % 안 써도 됨

print("Test1: {a: 5d}, Price: {b: 4.2f}".format(a=776, b=6534.123))

'''
참고 : Escape 코드

\n : 개행
\t : 탭
\\ : 문자
\' : 문자
\" : 문자
\r : 캐리지 리턴
\f : 폼피드
\a : 벨 소리
\b : 백 스페이스
\000 : 널 문자
'''

print("'you'")
print('\'you\'')
print('"you"')
print("""'you'""")
print('\\you\\\n\n\n')
print('test')
print('\t\t\ttest')