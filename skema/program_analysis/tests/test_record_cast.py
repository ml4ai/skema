# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Attribute,
    Var,
    Name,
    FunctionDef,
    Call,
    CASTLiteralValue,
    RecordDef,
    Operator,
    ModelReturn
)

def class1():
    return """
class MyClass:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b
        self.c = a + b

    def get_c(self):
        return self.c

mc = MyClass(2, 3)
x = mc.get_c()
    """

def class2():
    return """
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

    """

def class3():
    return """
class Foo:
    def __init__(self, a):
        self.a = a

    def add1(self):
        self.a = self.a + 1

    def sub1(self):
        self.a = self.a - 1

foo = Foo()

foo.add1().add1().sub1()
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_class1():
    fun_cast = generate_cast(class1())
    record_def_node = fun_cast.nodes[0].body[0]
    assert isinstance(record_def_node, RecordDef)
    assert record_def_node.name == "MyClass"
    assert len(record_def_node.bases) == 0
    assert len(record_def_node.fields) == 3
    fields = record_def_node.fields
    assert isinstance(fields[0], Name)
    assert fields[0].name == "a"
    assert fields[0].id == 3

    assert isinstance(fields[1], Name)
    assert fields[1].name == "b"
    assert fields[1].id == 4

    assert isinstance(fields[2], Name)
    assert fields[2].name == "c"
    assert fields[2].id == 5

    assert len(record_def_node.funcs) == 2
    init_func = record_def_node.funcs[0]
    assert isinstance(init_func, FunctionDef)
    assert isinstance(init_func.name, Name)
    assert init_func.name.name == "__init__"
    assert init_func.name.id == 1

    func_args = init_func.func_args
    assert len(func_args) == 3
    assert isinstance(func_args[0], Var)
    assert func_args[0].val.name == "self"
    assert func_args[0].val.id == 2

    assert isinstance(func_args[1], Var)
    assert func_args[1].val.name == "a"
    assert func_args[1].val.id == 3
    
    assert isinstance(func_args[2], Var)
    assert func_args[2].val.name == "b"
    assert func_args[2].val.id == 4
    
    func_body = init_func.body
    assert len(func_body) == 3
    assert isinstance(func_body[0], Assignment)
    assert isinstance(func_body[0].left, Attribute)
    assert func_body[0].left.value.name == "self"        
    assert func_body[0].left.attr.name == "a"

    assert isinstance(func_body[0].right, Name)
    assert func_body[0].right.name == "a"

    assert isinstance(func_body[1], Assignment)
    assert isinstance(func_body[1].left, Attribute)
    assert func_body[1].left.value.name == "self"        
    assert func_body[1].left.attr.name == "b"

    assert isinstance(func_body[1].right, Name)
    assert func_body[1].right.name == "b"

    assert isinstance(func_body[2], Assignment)
    assert isinstance(func_body[2].left, Attribute)
    assert func_body[2].left.value.name == "self"        
    assert func_body[2].left.attr.name == "c"

    assert isinstance(func_body[2].right, Operator)
    assert len(func_body[2].right.operands) == 2
    assert func_body[2].right.op == "ast.Add" 
    assert func_body[2].right.operands[0].name == "a"
    assert func_body[2].right.operands[1].name == "b"

    get_func = record_def_node.funcs[1]
    assert isinstance(get_func, FunctionDef)
    assert get_func.name.name == "get_c"
    func_args = get_func.func_args

    assert len(func_args) == 1
    assert func_args[0].val.name == "self"

    func_body = get_func.body
    assert len(func_body) == 1
    assert isinstance(func_body[0], ModelReturn)
    assert isinstance(func_body[0].value, Attribute)
    assert func_body[0].value.value.name == "self"
    assert func_body[0].value.attr.name == "c"

    #######################################################
    asg_node = fun_cast.nodes[0].body[1]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert asg_node.left.val.name == "mc"
    assert asg_node.left.val.id == 9

    func_call_node = asg_node.right
    assert isinstance(func_call_node, Call)
    assert func_call_node.func.name == "MyClass"

    assert len(func_call_node.arguments) == 2
    assert isinstance(func_call_node.arguments[0], CASTLiteralValue)
    assert func_call_node.arguments[0].value_type == "Integer"
    assert func_call_node.arguments[0].value == "2"
    
    assert isinstance(func_call_node.arguments[1], CASTLiteralValue)
    assert func_call_node.arguments[1].value_type == "Integer"
    assert func_call_node.arguments[1].value == "3"

    #######################################################
    asg_node = fun_cast.nodes[0].body[2]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert asg_node.left.val.name == "x"
    assert asg_node.left.val.id == 10

    assert isinstance(asg_node.right, Call)
    assert isinstance(asg_node.right.func, Attribute)
    assert asg_node.right.func.value.name == "mc" 
    assert asg_node.right.func.attr.name == "get_c"

def test_class2():
    fun_cast = generate_cast(class2())
    record_def_node = fun_cast.nodes[0].body[0]
    assert isinstance(record_def_node, RecordDef)
    assert record_def_node.name == "MyClass1"
    assert len(record_def_node.bases) == 0
    assert len(record_def_node.fields) == 1
    fields = record_def_node.fields
    assert isinstance(fields[0], Name)
    assert fields[0].name == "a"
    assert fields[0].id == 3

    assert len(record_def_node.funcs) == 2
    init_func = record_def_node.funcs[0]
    assert isinstance(init_func, FunctionDef)
    assert isinstance(init_func.name, Name)
    assert init_func.name.name == "__init__"
    assert init_func.name.id == 1

    func_args = init_func.func_args
    assert len(func_args) == 2
    assert isinstance(func_args[0], Var)
    assert func_args[0].val.name == "self"
    assert func_args[0].val.id == 2

    assert isinstance(func_args[1], Var)
    assert func_args[1].val.name == "a"
    assert func_args[1].val.id == 3
    
    func_body = init_func.body
    assert len(func_body) == 1
    assert isinstance(func_body[0], Assignment)
    assert isinstance(func_body[0].left, Attribute)
    assert func_body[0].left.value.name == "self"        
    assert func_body[0].left.attr.name == "a"

    assert isinstance(func_body[0].right, Name)
    assert func_body[0].right.name == "a"

    get_func = record_def_node.funcs[1]
    assert isinstance(get_func, FunctionDef)
    assert get_func.name.name == "get_a"
    func_args = get_func.func_args

    assert len(func_args) == 1
    assert func_args[0].val.name == "self"

    func_body = get_func.body
    assert len(func_body) == 1
    assert isinstance(func_body[0], ModelReturn)
    assert isinstance(func_body[0].value, Attribute)
    assert func_body[0].value.value.name == "self"
    assert func_body[0].value.attr.name == "a"

    #######################################################
    record_def_node = fun_cast.nodes[0].body[1]
    assert isinstance(record_def_node, RecordDef)
    assert record_def_node.name == "MyClass2"
    assert len(record_def_node.bases) == 0
    assert len(record_def_node.fields) == 1
    fields = record_def_node.fields
    assert isinstance(fields[0], Name)
    assert fields[0].name == "b"
    assert fields[0].id == 9

    assert len(record_def_node.funcs) == 2
    init_func = record_def_node.funcs[0]
    assert isinstance(init_func, FunctionDef)
    assert isinstance(init_func.name, Name)
    assert init_func.name.name == "__init__"
    assert init_func.name.id == 1

    func_args = init_func.func_args
    assert len(func_args) == 2
    assert isinstance(func_args[0], Var)
    assert func_args[0].val.name == "self"
    assert func_args[0].val.id == 8

    assert isinstance(func_args[1], Var)
    assert func_args[1].val.name == "b"
    assert func_args[1].val.id == 9
    
    func_body = init_func.body
    assert len(func_body) == 2
    assert isinstance(func_body[0], Assignment)
    assert isinstance(func_body[0].left, Attribute)
    assert func_body[0].left.value.name == "self"        
    assert func_body[0].left.attr.name == "b"

    assert isinstance(func_body[0].right, Name)
    assert func_body[0].right.name == "b"

    assert isinstance(func_body[1], Call)
    assert isinstance(func_body[1].func, Attribute)
    assert isinstance(func_body[1].func.value, Call)     
    assert func_body[1].func.value.func.name == "super"
    assert func_body[1].func.value.func.id == 10
    assert func_body[1].func.attr.name == "__init__"

    func_args = func_body[1].arguments
    assert isinstance(func_args[0], Operator)
    assert func_args[0].op == "ast.Add"

    assert len(func_args[0].operands) == 2
    assert func_args[0].operands[0].name == "b"

    assert isinstance(func_args[0].operands[1], CASTLiteralValue)
    assert func_args[0].operands[1].value == "1"

    get_func = record_def_node.funcs[1]
    assert isinstance(get_func, FunctionDef)
    assert get_func.name.name == "get_b"
    func_args = get_func.func_args

    assert len(func_args) == 1
    assert func_args[0].val.name == "self"

    func_body = get_func.body
    assert len(func_body) == 1
    assert isinstance(func_body[0], ModelReturn)
    assert isinstance(func_body[0].value, Attribute)
    assert func_body[0].value.value.name == "self"
    assert func_body[0].value.attr.name == "b"

    # #######################################################
    asg_node = fun_cast.nodes[0].body[2]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert asg_node.left.val.name == "mc"
    assert asg_node.left.val.id == 14

    assert isinstance(asg_node.right, Call)
    init_func = asg_node.right
    assert init_func.func.name == "MyClass2"
    assert len(init_func.arguments) == 1
    assert isinstance(init_func.arguments[0], CASTLiteralValue)

    asg_node = fun_cast.nodes[0].body[3]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert asg_node.left.val.name == "x"

    assert isinstance(asg_node.right, Call)
    assert isinstance(asg_node.right.func, Attribute)
    assert asg_node.right.func.attr.name == "get_a"
    assert asg_node.right.func.attr.id == 4
    assert asg_node.right.func.value.name == "mc"
    assert asg_node.right.func.value.id == 14


    asg_node = fun_cast.nodes[0].body[4]
    assert isinstance(asg_node, Assignment)
    assert isinstance(asg_node.left, Var)
    assert asg_node.left.val.name == "y"
    assert asg_node.left.val.id == 16

    assert isinstance(asg_node.right, Call)
    assert isinstance(asg_node.right.func, Attribute)
    assert asg_node.right.func.attr.name == "get_b"
    assert asg_node.right.func.attr.id == 11
    assert asg_node.right.func.value.name == "mc"
    assert asg_node.right.func.value.id == 14

def test_class3():
    class_cast = generate_cast(class3())
    record_def_node = class_cast.nodes[0].body[0]
    assert isinstance(record_def_node, RecordDef)
    assert record_def_node.name == "Foo"
    assert len(record_def_node.bases) == 0
    assert len(record_def_node.fields) == 1
    fields = record_def_node.fields
    assert isinstance(fields[0], Name)
    assert fields[0].name == "a"
    assert fields[0].id == 3

    assert len(record_def_node.funcs) == 3
    init_func = record_def_node.funcs[0]
    assert isinstance(init_func, FunctionDef)
    assert isinstance(init_func.name, Name)
    assert init_func.name.name == "__init__"
    assert init_func.name.id == 1

    func_args = init_func.func_args
    assert len(func_args) == 2
    assert isinstance(func_args[0], Var)
    assert func_args[0].val.name == "self"
    assert func_args[0].val.id == 2

    assert isinstance(func_args[1], Var)
    assert func_args[1].val.name == "a"
    assert func_args[1].val.id == 3
    
    func_body = init_func.body
    assert len(func_body) == 1
    assert isinstance(func_body[0], Assignment)
    assert isinstance(func_body[0].left, Attribute)
    assert func_body[0].left.value.name == "self"        
    assert func_body[0].left.attr.name == "a"

    assert isinstance(func_body[0].right, Name)
    assert func_body[0].right.name == "a"

    add_func = record_def_node.funcs[1]
    assert isinstance(add_func, FunctionDef)
    assert add_func.name.name == "add1"
    assert add_func.name.id == 4
    func_args = add_func.func_args

    assert len(func_args) == 1
    assert func_args[0].val.name == "self"

    func_body = add_func.body
    asg_stmt = func_body[0]
    assert isinstance(asg_stmt, Assignment)
    asg_left = asg_stmt.left
    assert isinstance(asg_left, Attribute)
    assert isinstance(asg_left.attr, Name)
    assert asg_left.attr.name == "a"
    assert isinstance(asg_left.value, Name)
    assert asg_left.value.name == "self"

    asg_right = asg_stmt.right
    assert isinstance(asg_right, Operator)
    assert asg_right.op == "ast.Add"
    assert isinstance(asg_right.operands[0], Attribute)
    assert isinstance(asg_right.operands[0].attr, Name)
    assert asg_right.operands[0].attr.name == "a"
    assert isinstance(asg_right.operands[0].value, Name)
    assert asg_right.operands[0].value.name == "self"

    assert isinstance(asg_right.operands[1], CASTLiteralValue)
                         
    sub_func = record_def_node.funcs[2]
    assert isinstance(sub_func, FunctionDef)
    assert sub_func.name.name == "sub1"
    assert sub_func.name.id == 7
    func_args = sub_func.func_args

    assert len(func_args) == 1
    assert func_args[0].val.name == "self"

    func_body = sub_func.body
    asg_stmt = func_body[0]
    assert isinstance(asg_stmt, Assignment)
    asg_left = asg_stmt.left
    assert isinstance(asg_left, Attribute)
    assert isinstance(asg_left.attr, Name)
    assert asg_left.attr.name == "a"
    assert isinstance(asg_left.value, Name)
    assert asg_left.value.name == "self"

    asg_right = asg_stmt.right
    assert isinstance(asg_right, Operator)
    assert asg_right.op == "ast.Sub"
    assert isinstance(asg_right.operands[0], Attribute)
    assert isinstance(asg_right.operands[0].attr, Name)
    assert asg_right.operands[0].attr.name == "a"
    assert isinstance(asg_right.operands[0].value, Name)
    assert asg_right.operands[0].value.name == "self"

    assert isinstance(asg_right.operands[1], CASTLiteralValue)
                         
    assignment_node = class_cast.nodes[0].body[1]
    assert isinstance(assignment_node, Assignment)
    assert isinstance(assignment_node.left, Var)
    assert isinstance(assignment_node.left.val, Name)
    assert assignment_node.left.val.name == "foo"
    assert assignment_node.left.val.id == 10

    assert isinstance(assignment_node.right, Call)
    assert assignment_node.right.func.name == "Foo"

    call_node = class_cast.nodes[0].body[2]
    assert isinstance(call_node, Call)
    assert isinstance(call_node.func, Attribute)
    assert isinstance(call_node.func.value, Call)
    assert call_node.func.attr.name == "sub1"
    assert call_node.func.attr.id == 7
    assert isinstance(call_node.func.value.func, Attribute)
    assert isinstance(call_node.func.value.func.value, Call)
    assert call_node.func.value.func.attr.name == "add1"
    assert call_node.func.value.func.attr.id == 4
    assert isinstance(call_node.func.value.func.value.func, Attribute)
    assert call_node.func.value.func.value.func.value.name == "foo"
    assert call_node.func.value.func.value.func.attr.name == "add1"
    assert call_node.func.value.func.value.func.attr.id == 4
