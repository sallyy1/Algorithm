# 백준 7490번: 0 만들기

- 주어진 N의 범위가 매우 한정적이므로, **완전탐색** 가능 !

*각 테스트 케이스엔 자연수 N이 주어진다(3 <= N <= 9).*

- 자연수 N 일 때, 연산자 리스트의 조합은 3*(N-1)


```python
# eval() 함수 : 문자열 형태의 표현식 계산

print(eval("1+2-3"))
print(eval("12+3+4-5-6-7"))
```

    0
    1



```python
# 정답

import copy # copy.deepcopy()

# 연산자 경우의 수 만드는 재귀(총 3^(n-1))
def recursive(array, n): # 연산자 리스트 만드는 함수 (길이: n)
    # 종료 조건
    if len(array) == n:
        operators_list.append(copy.deepcopy(array)) # 깊은 복사
        return
    
    # 1) 공백 연산자
    array.append(' ')
    recursive(array, n)
    array.pop() # 다시 빈 리스트로 만들기
    
    # 2) 덧셈 연산자
    array.append('+')
    recursive(array, n)
    array.pop() # 다시 빈 리스트로 만들기
    
    # 3) 뺄셈 연산자
    array.append('-')
    recursive(array, n)
    array.pop() # 다시 빈 리스트로 만들기
    
    

# Test Case    
test_case = int(input())

for _ in range(test_case):
    operators_list = []
    n = int(input())
    
    recursive([], n-1)
    
    
    integers = [i for i in range(1, n+1)]
    

    for operators in operators_list:
        string = ""
        
        # "1+2-3" 형식 string 만들기
        for i in range(n-1):
            string += str(integers[i]) + operators[i]
        string += str(integers[-1])
        
        # 판별
        if eval(string.replace(" ", "")) == 0: # 공백 띄어쓰기 제거한 string 문자열의 계산이 0에 해당되는가?
            print(string)
    print()
```

    2
    3
    1+2-3
    
    7
    1+2-3+4-5-6+7
    1+2-3-4+5+6-7
    1-2 3+4+5+6+7
    1-2 3-4 5+6 7
    1-2+3+4-5+6-7
    1-2-3-4-5+6+7
    



```python

```
