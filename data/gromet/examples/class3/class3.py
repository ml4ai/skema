# Example with chain of method calls

class Foo:
    def __init__(self, a):
        self.a = a

    def add1(self):
        self.a = self.a + 1
        return self
    
    def sub1(self):
        self.a = self.a - 1
        return self


foo = Foo()

foo.add1().add1().sub1()
