def bar():  # defines module.bar()
    print("module.bar()")

def foo(): # defines module.foo()
    bar()  # calling module.bar()
    def bar(): # defining module.foo.bar()
        print("module.foo.bar()")
    bar() # calling module.foo.bar()

foo()
bar() # calls module.bar()
