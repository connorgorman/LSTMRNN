

filename = 'ptb.test.txt.sorted.integerized'


content = [x for x in open(filename).readlines() ]


f = open('ptb.test.sort.int.extended.txt', 'w')

if True:
	for i in range(0, len(content), 4):
		line0 = content[i]
		line1 = content[i+1]
		f.write(line0)
		f.write(line1)


