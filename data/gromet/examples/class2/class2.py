class MyClass1:
    def __init__(self, a: int):
        self.a = a

    def get_a(self):
        return self.a

class MyClass2(MyClass1):
    def __init__(self, b: int):
        self.b = b
        super().__init__(b + 1)

    def get_b(self):
        return self.b


mc = MyClass2(2)
x = mc.get_a()
y = mc.get_b()