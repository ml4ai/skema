
class MyClass:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b
        self.c = a + b

    def get_c(self):
        return self.c


mc = MyClass(2, 3)
x = mc.get_c()


# What this requires:
#   Defining the new type with fields
#   Defining constructor (magic method) ==> function that returns an instance of he object
#   Defining a class method ==> function that takes the object as an argument

# Next steps (in order of importance):
#   Inheritance
#   GroMEt primitive functions that work with typing, e.g.,
#       isinstance(mc, 'MyClass')
#       type(mc) == 'MyClass'
#   Pythonic "magic" methods: e.g., implementing __eq__, etc...
