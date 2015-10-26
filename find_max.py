
from collections import defaultdict as dd

content = [x for x in open('partial_test.txt').readlines()]

longest = dd(int)

for line in content:
	split = line.split(' ')

	longest[len(split)] += 1


print longest


f = open('partial_test_19.txt', 'w')

for line in content:
	split = line.split(' ')
	if len(split) == 19:
		f.write(line)
