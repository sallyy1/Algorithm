#
if int i < 10
print(i)

# 6번
a = ['파', '이', '썬', '썬', '썬', '썬', '즐', '즐', '즐', '거', '운']
last = None
for elem in a:
    if elem == last:
        continue
    print(elem, end = '')
    last = elem


# 7번
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = lambda x: x * x
c = list()
for elem in a:
    c.append(b(elem))
print(c)



# 8번
#c


# 9번
# 교집합 -> intersection


# 10번
#d