# Section 06
# 파이썬 함수식 및 람다(lambda)

# 함수 정의 방법
# def 함수명(parameter):
#   code

# 함수 호출 -> function_name()
# 함수명(parameter)

# 함수 선언 위치 중요

# 예제 1 (리턴 값이 없는 함수)
def hello(world):
    print("Hello", world)

hello("Python!")
hello(7777)

# 예제 2 (리턴 값이 있는 함수)
def hello_return(world):
    val = "Hello " + str(world) # 문자 + 숫자 는 오류가 뜨므로, str()로 형변환 해줌
    return val

str = hello_return("Python!!!!!")
print(str)

# 예제 3 (다중리턴)
def func_mul(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300
    return y1, y2, y3

val1, val2, val3 = func_mul(100)
print(val1, val2, val3)


# 예제 4 (데이터 타입 반환)
def func_mul2(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300
    return [y1, y2, y3] # 리스트로 반환

lt = func_mul2(100)
print(lt, type(lt))


def func_mul2(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300
    return (y1, y2, y3) # 튜플로 반환

tp = func_mul2(100)
print(tp, type(tp))


def func_mul3(x):
    y1 = x * 2
    y2 = x * 4
    y3 = x * 6
    return {'ret1': y1, 'ret2': y2, 'ret3': y3} # 딕셔너리 리턴


dic = func_mul3(8)

print(type(dic), dic, dic.get('ret3'), dic.items(), dic.keys(), dic.values())


# 예제 4
# *args, *kwargs

# args
# 매개변수명 자유롭게 변경 가능
def args_func(*args): # * 1개 -> 가변함수 -> 튜플로 반환해줌
    #print(type(args))
    for t in args: # 튜플이므로 for문 가능
        print(t)

args_func('Kim')
args_func('Kim', 'Park')
args_func('Kim', 'Park', 'Lee')


def args_func(*args):
    #print(type(args))
    for i, v in enumerate(args): # enumerate() -> 인덱스를 만들어주는 함수
        print(i, v)

args_func('Kim')
args_func('Kim', 'Park')
args_func('Kim', 'Park', 'Lee')


def args_func(*args):
    #print(type(args))
    for i, v in enumerate(range(100)): # enumerate() -> 인덱스를 만들어주는 함수
        print(i, v)


# kwargs (키워드 파라미터(인자))
# 매개변수명 자유롭게 변경 가능
def kwargs_func(**kwargs): # ** 2개 -> 딕셔너리 형태로 인자를 받음
    print(kwargs)

kwargs_func(name1 = 'Kim', name2 = 'Park', name3 = 'Lee')


def kwargs_func(**kwargs):
    for k, v in kwargs.items():
        print(k, v)

kwargs_func(name1 = 'Kim', name2 = 'Park', name3 = 'Lee')


# 전체 혼합
def example_mul(arg1, arg2, *args, **kwargs):
    print(arg1, arg2, args, kwargs)

example_mul(10, 20)
example_mul(10, 20, 'Park', 'Kim')
example_mul(10, 20, 'Park', 'Kim', age1=24, age2=35)


# 중첩함수(클로저)
# 파이썬 -> 데코레이터 클로저
def nested_func(num):
    def func_in_func(num):
        print('>>>', num) # 최종 실행문
    print("in func")
    func_in_func(num + 10000)

nested_func(10000)

# 실행불가
# func_in_func(1)



# 예제6
# Hint
def tot_length1(word: str, num: int) -> int:
    return len(word) * num


print('hint exam1 : ', tot_length1("i love you", 10))


def tot_length2(word: str, num: int) -> None:
    print('hint exam2 : ', len(word) * num)


tot_length2("niceman", 10)


# 람다식 예제
# 메모리 절약, 가독성 향상, 코드 간결
# 함수는 객체 생성 -> 리소스(메모리) 할당
# 람다는 즉시 실행 함수(Heap 초기화) -> 메모리 초기화

# 예제7
# def mul_10(num):
#     return num * 10

# def mul_10_one(num): return num * 10
#
# lambda x: x * 10

# 일반적 함수 -> 변수 할당
def mul_10(num):
    return num * 10


mul_func = mul_10

print(mul_func(5))
print(mul_func(6))

# 람다 함수 -> 할당
lambda_mul_func = lambda x: x * 10


def func_final(x, y, func): # parameter에 함수를 넣는 것도 가능 (?)
    print(x * y * func(10))


func_final(10, 10, lambda_mul_func)


