import random as r

s = 216*[1] + (1391-216)*[0]

for _ in range(1000000):
    r.shuffle(s)
    if sum(s[:667]) < 75:
        print "hi"
