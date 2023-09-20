def rec1(x):
	rec2(x + 1)

def rec2(y):
	rec1(y + 2)

z = rec1(12)