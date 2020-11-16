from pythonping import ping


a=(ping('168.20.0.1'))
a = list(a)
a = map(str,a)
# print(list(a))
# for i in a:
#     print((i))

if any('Request timed out' in s for s in a):
    print('hello')