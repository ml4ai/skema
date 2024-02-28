# import json               NOTE: json and Path aren't used right now,
# from pathlib import Path        but will be used in the future
from skema.program_analysis.CAST.python.ts2cast import TS2CAST
from skema.program_analysis.CAST2FN.model.cast import (
    Assignment,
    Attribute,
    FunctionDef,
    Call,
    Var,
    Loop,
    ModelReturn,
    Name,
    CASTLiteralValue,
    ModelIf,
    StructureType,
    ScalarType,
    Operator
)

def comp1():
    return """
L = [a*b for a in range(10) for b in range(10)]
    """

def comp2():
    return """
L = [a for a in range(10) if a % 2 == 0]
    """

def comp3():
    return """
L = [a*b for a in range(10) if a % 2 == 0 for b in range(10) if b % 2 == 1]
    """

def lambda1():
    return """
y = 2
F = lambda x : x * y
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_comp1():
    comp_cast = generate_cast(comp1())
    
    # Test list comprehension with double for loops
    func_node = comp_cast.nodes[0].body[0]
    assert isinstance(func_node, FunctionDef)
    assert "%comprehension_list_0" == func_node.name.name
    assert len(func_node.func_args) == 0
    
    func_node_body = func_node.body
    func_asg = func_node.body[0]
    func_ret = func_node.body[2]
    assert isinstance(func_asg, Assignment)
    assert isinstance(func_asg.left, Var)
    assert isinstance(func_asg.left.val, Name)
    assert "list__temp_" == func_asg.left.val.name

    assert isinstance(func_asg.right, CASTLiteralValue)
    assert func_asg.right.value_type == StructureType.LIST
    assert func_asg.right.value == []

    assert isinstance(func_ret, ModelReturn)
    assert isinstance(func_ret.value, Name)
    assert "list__temp_" == func_ret.value.name 

    func_loop = func_node_body[1]
    assert isinstance(func_loop, Loop)
    assert func_loop.post == [] 

    # Loop Pre
    #############
    func_loop_pre = func_loop.pre   
    loop_pre_iter = func_loop_pre[0]
    assert isinstance(loop_pre_iter, Assignment)
    assert isinstance(loop_pre_iter.left, Var)
    assert isinstance(loop_pre_iter.left.val, Name)
    assert "generated_iter_0" == loop_pre_iter.left.val.name

    assert isinstance(loop_pre_iter.right, Call)
    assert isinstance(loop_pre_iter.right.func, Name)
    assert loop_pre_iter.right.func.name == "iter"

    assert len(loop_pre_iter.right.arguments) == 1
    range_call = loop_pre_iter.right.arguments[0]
    assert isinstance(range_call, Call)
    assert isinstance(range_call.func, Name)
    assert range_call.func.name == "range"
    assert len(range_call.arguments) == 3
    assert isinstance(range_call.arguments[0], CASTLiteralValue)
    assert isinstance(range_call.arguments[1], CASTLiteralValue)
    assert isinstance(range_call.arguments[2], CASTLiteralValue)

    loop_pre_next = func_loop_pre[1]
    assert isinstance(loop_pre_next, Assignment)
    assert isinstance(loop_pre_next.left, CASTLiteralValue)
    assert loop_pre_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_pre_next.left.value[0], Var)
    assert isinstance(loop_pre_next.left.value[0].val, Name)
    assert loop_pre_next.left.value[0].val.name == "a"

    assert isinstance(loop_pre_next.left.value[1], Var)
    assert isinstance(loop_pre_next.left.value[1].val, Name)
    assert "generated_iter_0" == loop_pre_next.left.value[1].val.name

    assert isinstance(loop_pre_next.left.value[2], Var)
    assert isinstance(loop_pre_next.left.value[2].val, Name)
    assert "sc_" in loop_pre_next.left.value[2].val.name 

    assert isinstance(loop_pre_next.right, Call)
    assert isinstance(loop_pre_next.right.func, Name)
    assert loop_pre_next.right.func.name == "next"
    assert len(loop_pre_next.right.arguments) == 1
    assert isinstance(loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_0" == loop_pre_next.right.arguments[0].val.name 


    # Loop Test
    #############
    func_loop_test = func_loop.expr
    assert isinstance(func_loop_test, Operator)
    assert func_loop_test.op == "ast.Eq"

    assert isinstance(func_loop_test.operands[0], Name)
    assert "sc_" in func_loop_test.operands[0].name

    assert isinstance(func_loop_test.operands[1], CASTLiteralValue)
    assert func_loop_test.operands[1].value == False
    assert func_loop_test.operands[1].value_type == ScalarType.BOOLEAN


    # Loop Body
    #############
    inner_loop_body = func_loop.body[0]  
    assert isinstance(inner_loop_body, Loop) 

    def test_inner_loop(inner_loop: Loop):
        # Inner Loop Pre
        ##################
        inner_loop_pre = inner_loop.pre
        inner_loop_iter = inner_loop_pre[0]
        assert isinstance(inner_loop_iter, Assignment)
        assert isinstance(inner_loop_iter.left, Var)
        assert isinstance(inner_loop_iter.left.val, Name)
        assert "generated_iter_1" == inner_loop_iter.left.val.name

        assert isinstance(inner_loop_iter.right, Call)
        assert isinstance(inner_loop_iter.right.func, Name)
        assert inner_loop_iter.right.func.name == "iter"

        assert len(inner_loop_iter.right.arguments) == 1
        range_call = inner_loop_iter.right.arguments[0]
        assert isinstance(range_call, Call)
        assert isinstance(range_call.func, Name)
        assert range_call.func.name == "range"
        assert len(range_call.arguments) == 3
        assert isinstance(range_call.arguments[0], CASTLiteralValue)
        assert isinstance(range_call.arguments[1], CASTLiteralValue)
        assert isinstance(range_call.arguments[2], CASTLiteralValue)

        inner_loop_pre_next = inner_loop_pre[1]
        assert isinstance(inner_loop_pre_next, Assignment)
        assert isinstance(inner_loop_pre_next.left, CASTLiteralValue)
        assert inner_loop_pre_next.left.value_type == StructureType.TUPLE
        assert isinstance(inner_loop_pre_next.left.value[0], Var)
        assert isinstance(inner_loop_pre_next.left.value[0].val, Name)
        assert inner_loop_pre_next.left.value[0].val.name == "b" # NOTE

        assert isinstance(inner_loop_pre_next.left.value[1], Var)
        assert isinstance(inner_loop_pre_next.left.value[1].val, Name)
        assert "generated_iter_1" == inner_loop_pre_next.left.value[1].val.name

        assert isinstance(inner_loop_pre_next.left.value[2], Var)
        assert isinstance(inner_loop_pre_next.left.value[2].val, Name)
        assert "sc_" in inner_loop_pre_next.left.value[2].val.name 

        assert isinstance(inner_loop_pre_next.right, Call)
        assert isinstance(inner_loop_pre_next.right.func, Name)
        assert inner_loop_pre_next.right.func.name == "next"
        assert len(inner_loop_pre_next.right.arguments) == 1
        assert isinstance(inner_loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
        assert "generated_iter_1" == inner_loop_pre_next.right.arguments[0].val.name 

        # Inner Loop Test
        ##################
        inner_loop_test = inner_loop.expr
        assert isinstance(inner_loop_test, Operator)
        assert inner_loop_test.op == "ast.Eq"

        assert isinstance(inner_loop_test.operands[0], Name)
        assert "sc_" in inner_loop_test.operands[0].name

        assert isinstance(inner_loop_test.operands[1], CASTLiteralValue)
        assert inner_loop_test.operands[1].value == False
        assert inner_loop_test.operands[1].value_type == ScalarType.BOOLEAN

        # Inner Loop Body
        ##################
        loop_body_append = inner_loop.body[0]
        assert isinstance(loop_body_append, Call)
        assert isinstance(loop_body_append.func, Attribute)
        assert isinstance(loop_body_append.func.attr, Name)
        assert loop_body_append.func.attr.name == "append"

        assert isinstance(loop_body_append.func.value, Name)
        assert "list__temp_" == loop_body_append.func.value.name

        assert len(loop_body_append.arguments) == 1
        assert isinstance(loop_body_append.arguments[0], Operator)
        assert loop_body_append.arguments[0].op == "ast.Mult"

        assert isinstance(loop_body_append.arguments[0].operands[0], Name)
        assert loop_body_append.arguments[0].operands[0].name == "a"

        assert isinstance(loop_body_append.arguments[0].operands[1], Name)
        assert loop_body_append.arguments[0].operands[1].name == "b"

        loop_body_next = inner_loop.body[1]
        assert isinstance(loop_body_next, Assignment)
        assert isinstance(loop_body_next.left, CASTLiteralValue)
        assert loop_body_next.left.value_type == StructureType.TUPLE
        assert isinstance(loop_body_next.left.value[0], Var)
        assert isinstance(loop_body_next.left.value[0].val, Name)
        assert loop_body_next.left.value[0].val.name == "b"

        assert isinstance(loop_body_next.left.value[1], Var)
        assert isinstance(loop_body_next.left.value[1].val, Name)
        assert "generated_iter_1" == loop_body_next.left.value[1].val.name

        assert isinstance(loop_body_next.left.value[2], Var)
        assert isinstance(loop_body_next.left.value[2].val, Name)
        assert "sc_" in loop_body_next.left.value[2].val.name 

        assert isinstance(loop_body_next.right, Call)
        assert isinstance(loop_body_next.right.func, Name)
        assert loop_body_next.right.func.name == "next"
        assert len(loop_body_next.right.arguments) == 1
        assert isinstance(loop_body_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
        assert "generated_iter_1" == loop_body_next.right.arguments[0].val.name 

    test_inner_loop(inner_loop_body)
    # Loop Call
    #############

    call_node = comp_cast.nodes[0].body[1]
    assert isinstance(call_node, Assignment)
    assert isinstance(call_node.left, Var)
    assert isinstance(call_node.left.val, Name)
    assert call_node.left.val.name == "L"

    assert isinstance(call_node.right, Call)
    assert isinstance(call_node.right.func, Name)
    assert "%comprehension_list_0" == call_node.right.func.name 
    assert len(call_node.right.arguments) == 0

def test_comp2():
    comp_cast = generate_cast(comp2())
    
    # Test list comprehension with conditional
    func_node = comp_cast.nodes[0].body[0]
    assert isinstance(func_node, FunctionDef)
    assert "%comprehension_list_0" == func_node.name.name
    assert len(func_node.func_args) == 0
    
    func_node_body = func_node.body
    func_asg = func_node.body[0]
    func_ret = func_node.body[2]
    assert isinstance(func_asg, Assignment)
    assert isinstance(func_asg.left, Var)
    assert isinstance(func_asg.left.val, Name)
    assert "list__temp_" == func_asg.left.val.name

    assert isinstance(func_asg.right, CASTLiteralValue)
    assert func_asg.right.value_type == StructureType.LIST
    assert func_asg.right.value == []

    assert isinstance(func_ret, ModelReturn)
    assert isinstance(func_ret.value, Name)
    assert "list__temp_" == func_ret.value.name 

    func_loop = func_node_body[1]
    assert isinstance(func_loop, Loop)
    assert func_loop.post == [] 

    # Loop Pre
    #############
    func_loop_pre = func_loop.pre   
    loop_pre_iter = func_loop_pre[0]
    assert isinstance(loop_pre_iter, Assignment)
    assert isinstance(loop_pre_iter.left, Var)
    assert isinstance(loop_pre_iter.left.val, Name)
    assert "generated_iter_0" == loop_pre_iter.left.val.name

    assert isinstance(loop_pre_iter.right, Call)
    assert isinstance(loop_pre_iter.right.func, Name)
    assert loop_pre_iter.right.func.name == "iter"

    assert len(loop_pre_iter.right.arguments) == 1
    range_call = loop_pre_iter.right.arguments[0]
    assert isinstance(range_call, Call)
    assert isinstance(range_call.func, Name)
    assert range_call.func.name == "range"
    assert len(range_call.arguments) == 3
    assert isinstance(range_call.arguments[0], CASTLiteralValue)
    assert isinstance(range_call.arguments[1], CASTLiteralValue)
    assert isinstance(range_call.arguments[2], CASTLiteralValue)

    loop_pre_next = func_loop_pre[1]
    assert isinstance(loop_pre_next, Assignment)
    assert isinstance(loop_pre_next.left, CASTLiteralValue)
    assert loop_pre_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_pre_next.left.value[0], Var)
    assert isinstance(loop_pre_next.left.value[0].val, Name)
    assert loop_pre_next.left.value[0].val.name == "a"

    assert isinstance(loop_pre_next.left.value[1], Var)
    assert isinstance(loop_pre_next.left.value[1].val, Name)
    assert "generated_iter_0" == loop_pre_next.left.value[1].val.name

    assert isinstance(loop_pre_next.left.value[2], Var)
    assert isinstance(loop_pre_next.left.value[2].val, Name)
    assert "sc_" in loop_pre_next.left.value[2].val.name 

    assert isinstance(loop_pre_next.right, Call)
    assert isinstance(loop_pre_next.right.func, Name)
    assert loop_pre_next.right.func.name == "next"
    assert len(loop_pre_next.right.arguments) == 1
    assert isinstance(loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_0" == loop_pre_next.right.arguments[0].val.name 


    # Loop Test
    #############
    func_loop_test = func_loop.expr
    assert isinstance(func_loop_test, Operator)
    assert func_loop_test.op == "ast.Eq"

    assert isinstance(func_loop_test.operands[0], Name)
    assert "sc_" in func_loop_test.operands[0].name

    assert isinstance(func_loop_test.operands[1], CASTLiteralValue)
    assert func_loop_test.operands[1].value == False
    assert func_loop_test.operands[1].value_type == ScalarType.BOOLEAN


    # Loop Body
    #############
    func_loop_body = func_loop.body  
    loop_body_if = func_loop_body[0]
    assert isinstance(loop_body_if, ModelIf)
    body_if_test = loop_body_if.expr
    assert isinstance(body_if_test, Operator)
    assert body_if_test.op == "ast.Eq"
    assert isinstance(body_if_test.operands[0], Operator)
    assert body_if_test.operands[0].op == "ast.Mod" 
    assert isinstance(body_if_test.operands[0].operands[0], Name)
    assert body_if_test.operands[0].operands[0].name == "a"

    assert isinstance(body_if_test.operands[0].operands[1], CASTLiteralValue)
    assert isinstance(body_if_test.operands[1], CASTLiteralValue)

    body_if_body = loop_body_if.body[0]
     
    assert isinstance(body_if_body, Call)
    assert isinstance(body_if_body.func, Attribute)
    assert isinstance(body_if_body.func.attr, Name)
    assert body_if_body.func.attr.name == "append"

    assert isinstance(body_if_body.func.value, Name)
    assert "list__temp_" in body_if_body.func.value.name

    assert len(body_if_body.arguments) == 1
    assert isinstance(body_if_body.arguments[0], Var)
    assert body_if_body.arguments[0].val.name == "a"
    
    loop_body_next = func_loop_body[1]
    assert isinstance(loop_body_next, Assignment)
    assert isinstance(loop_body_next.left, CASTLiteralValue)
    assert loop_body_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_body_next.left.value[0], Var)
    assert isinstance(loop_body_next.left.value[0].val, Name)
    assert loop_body_next.left.value[0].val.name == "a"

    assert isinstance(loop_body_next.left.value[1], Var)
    assert isinstance(loop_body_next.left.value[1].val, Name)
    assert "generated_iter_0" == loop_body_next.left.value[1].val.name

    assert isinstance(loop_body_next.left.value[2], Var)
    assert isinstance(loop_body_next.left.value[2].val, Name)
    assert "sc_" in loop_body_next.left.value[2].val.name 

    assert isinstance(loop_body_next.right, Call)
    assert isinstance(loop_body_next.right.func, Name)
    assert loop_body_next.right.func.name == "next"
    assert len(loop_body_next.right.arguments) == 1
    assert isinstance(loop_body_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_0" == loop_body_next.right.arguments[0].val.name 

    call_node = comp_cast.nodes[0].body[1]
    assert isinstance(call_node, Assignment)
    assert isinstance(call_node.left, Var)
    assert isinstance(call_node.left.val, Name)
    assert call_node.left.val.name == "L"

    assert isinstance(call_node.right, Call)
    assert isinstance(call_node.right.func, Name)
    assert "%comprehension_list_0" == call_node.right.func.name 
    assert len(call_node.right.arguments) == 0

def test_comp3():
    comp_cast = generate_cast(comp3())
    
    # Test list comprehension with conditional
    func_node = comp_cast.nodes[0].body[0]
    assert isinstance(func_node, FunctionDef)
    assert "%comprehension_list_0" == func_node.name.name
    assert len(func_node.func_args) == 0
    
    func_node_body = func_node.body
    func_asg = func_node.body[0]
    func_ret = func_node.body[2]
    assert isinstance(func_asg, Assignment)
    assert isinstance(func_asg.left, Var)
    assert isinstance(func_asg.left.val, Name)
    assert "list__temp_" == func_asg.left.val.name

    assert isinstance(func_asg.right, CASTLiteralValue)
    assert func_asg.right.value_type == StructureType.LIST
    assert func_asg.right.value == []

    assert isinstance(func_ret, ModelReturn)
    assert isinstance(func_ret.value, Name)
    assert "list__temp_" == func_ret.value.name 

    func_loop = func_node_body[1]
    assert isinstance(func_loop, Loop)
    assert func_loop.post == [] 

    # Loop Pre
    #############
    func_loop_pre = func_loop.pre   
    loop_pre_iter = func_loop_pre[0]
    assert isinstance(loop_pre_iter, Assignment)
    assert isinstance(loop_pre_iter.left, Var)
    assert isinstance(loop_pre_iter.left.val, Name)
    assert "generated_iter_0" == loop_pre_iter.left.val.name

    assert isinstance(loop_pre_iter.right, Call)
    assert isinstance(loop_pre_iter.right.func, Name)
    assert loop_pre_iter.right.func.name == "iter"

    assert len(loop_pre_iter.right.arguments) == 1
    range_call = loop_pre_iter.right.arguments[0]
    assert isinstance(range_call, Call)
    assert isinstance(range_call.func, Name)
    assert range_call.func.name == "range"
    assert len(range_call.arguments) == 3
    assert isinstance(range_call.arguments[0], CASTLiteralValue)
    assert isinstance(range_call.arguments[1], CASTLiteralValue)
    assert isinstance(range_call.arguments[2], CASTLiteralValue)

    loop_pre_next = func_loop_pre[1]
    assert isinstance(loop_pre_next, Assignment)
    assert isinstance(loop_pre_next.left, CASTLiteralValue)
    assert loop_pre_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_pre_next.left.value[0], Var)
    assert isinstance(loop_pre_next.left.value[0].val, Name)
    assert loop_pre_next.left.value[0].val.name == "a"

    assert isinstance(loop_pre_next.left.value[1], Var)
    assert isinstance(loop_pre_next.left.value[1].val, Name)
    assert "generated_iter_0" == loop_pre_next.left.value[1].val.name

    assert isinstance(loop_pre_next.left.value[2], Var)
    assert isinstance(loop_pre_next.left.value[2].val, Name)
    assert "sc_" in loop_pre_next.left.value[2].val.name 

    assert isinstance(loop_pre_next.right, Call)
    assert isinstance(loop_pre_next.right.func, Name)
    assert loop_pre_next.right.func.name == "next"
    assert len(loop_pre_next.right.arguments) == 1
    assert isinstance(loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_0" == loop_pre_next.right.arguments[0].val.name 


    # Loop Test
    #############
    func_loop_test = func_loop.expr
    assert isinstance(func_loop_test, Operator)
    assert func_loop_test.op == "ast.Eq"

    assert isinstance(func_loop_test.operands[0], Name)
    assert "sc_" in func_loop_test.operands[0].name

    assert isinstance(func_loop_test.operands[1], CASTLiteralValue)
    assert func_loop_test.operands[1].value == False
    assert func_loop_test.operands[1].value_type == ScalarType.BOOLEAN


    # Loop Body
    #############
    func_loop_body = func_loop.body 
    loop_body_if = func_loop_body[0]
    assert isinstance(loop_body_if, ModelIf)

    body_if_test = loop_body_if.expr
    assert isinstance(body_if_test, Operator)
    assert body_if_test.op == "ast.Eq"
    assert isinstance(body_if_test.operands[0], Operator)
    assert body_if_test.operands[0].op == "ast.Mod" 
    assert isinstance(body_if_test.operands[0].operands[0], Name)
    assert body_if_test.operands[0].operands[0].name == "a"
    assert isinstance(body_if_test.operands[0].operands[1], CASTLiteralValue)
    assert isinstance(body_if_test.operands[1], CASTLiteralValue)

    body_if_body = loop_body_if.body[0]
    assert isinstance(body_if_body, Loop)
    
    def test_inner_body_inner_loop(node: Loop):
        pre = node.pre

        loop_pre_iter = pre[0]
        assert isinstance(loop_pre_iter, Assignment)
        assert isinstance(loop_pre_iter.left, Var)
        assert isinstance(loop_pre_iter.left.val, Name)
        assert "generated_iter_1" == loop_pre_iter.left.val.name

        assert isinstance(loop_pre_iter.right, Call)
        assert isinstance(loop_pre_iter.right.func, Name)
        assert loop_pre_iter.right.func.name == "iter"

        assert len(loop_pre_iter.right.arguments) == 1
        range_call = loop_pre_iter.right.arguments[0]
        assert isinstance(range_call, Call)
        assert isinstance(range_call.func, Name)
        assert range_call.func.name == "range"
        assert len(range_call.arguments) == 3
        assert isinstance(range_call.arguments[0], CASTLiteralValue)
        assert isinstance(range_call.arguments[1], CASTLiteralValue)
        assert isinstance(range_call.arguments[2], CASTLiteralValue)

        loop_pre_next = pre[1]
        assert isinstance(loop_pre_next, Assignment)
        assert isinstance(loop_pre_next.left, CASTLiteralValue)
        assert loop_pre_next.left.value_type == StructureType.TUPLE
        assert isinstance(loop_pre_next.left.value[0], Var)
        assert isinstance(loop_pre_next.left.value[0].val, Name)
        assert loop_pre_next.left.value[0].val.name == "b"

        assert isinstance(loop_pre_next.left.value[1], Var)
        assert isinstance(loop_pre_next.left.value[1].val, Name)
        assert "generated_iter_1" == loop_pre_next.left.value[1].val.name

        assert isinstance(loop_pre_next.left.value[2], Var)
        assert isinstance(loop_pre_next.left.value[2].val, Name)
        assert "sc_" in loop_pre_next.left.value[2].val.name 

        assert isinstance(loop_pre_next.right, Call)
        assert isinstance(loop_pre_next.right.func, Name)
        assert loop_pre_next.right.func.name == "next"
        assert len(loop_pre_next.right.arguments) == 1
        assert isinstance(loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
        assert "generated_iter_1" == loop_pre_next.right.arguments[0].val.name 

        test = node.expr
        assert isinstance(test, Operator)
        assert test.op =="ast.Eq"
        assert isinstance(test.operands[0], Name)
        assert "sc_" in test.operands[0].name 
        assert isinstance(test.operands[1], CASTLiteralValue)
        assert test.operands[1].value_type == ScalarType.BOOLEAN
        assert test.operands[1].value == False

        body = node.body[0]
        assert isinstance(body, ModelIf)
        assert isinstance(body.expr, Operator)
        assert body.expr.op == "ast.Eq"
        assert isinstance(body.expr.operands[0], Operator)
        assert body.expr.operands[0].op == "ast.Mod"
        assert isinstance(body.expr.operands[0].operands[0], Name)
        assert body.expr.operands[0].operands[0].name == "b"
        assert isinstance(body.expr.operands[0].operands[1], CASTLiteralValue)

        assert isinstance(body.expr.operands[1], CASTLiteralValue)

        assert isinstance(body.body[0], Call)
        assert isinstance(body.body[0].func, Attribute)
        assert isinstance(body.body[0].func.attr, Name)
        assert body.body[0].func.attr.name == "append"

        assert isinstance(body.body[0].func.value, Name)
        assert "list__temp_" in body.body[0].func.value.name

        assert len(body.body[0].arguments) == 1
        assert isinstance(body.body[0].arguments[0], Operator) 
        assert body.body[0].arguments[0].op == "ast.Mult"

        assert isinstance(body.body[0].arguments[0].operands[0], Name)
        assert body.body[0].arguments[0].operands[0].name == "a"

        assert isinstance(body.body[0].arguments[0].operands[1], Name)
        assert body.body[0].arguments[0].operands[1].name == "b"

        body_2 = node.body[1]
        assert isinstance(body_2, Assignment)
        assert isinstance(body_2.left, CASTLiteralValue)
        assert body_2.left.value_type == StructureType.TUPLE
        assert isinstance(body_2.left.value[0], Var)
        assert isinstance(body_2.left.value[0].val, Name)
        assert body_2.left.value[0].val.name == "b"

        assert isinstance(body_2.left.value[1], Var)
        assert isinstance(body_2.left.value[1].val, Name)
        assert "generated_iter_1" == body_2.left.value[1].val.name

        assert isinstance(body_2.left.value[2], Var)
        assert isinstance(body_2.left.value[2].val, Name)
        assert "sc_" in body_2.left.value[2].val.name 

        assert isinstance(body_2.right, Call)
        assert isinstance(body_2.right.func, Name)
        assert body_2.right.func.name == "next"
        assert len(body_2.right.arguments) == 1
        assert isinstance(body_2.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
        assert "generated_iter_1" == body_2.right.arguments[0].val.name 

    test_inner_body_inner_loop(body_if_body)

    loop_body_next = func_loop_body[1]
    assert isinstance(loop_body_next, Assignment)
    assert isinstance(loop_body_next.left, CASTLiteralValue)
    assert loop_body_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_body_next.left.value[0], Var)
    assert isinstance(loop_body_next.left.value[0].val, Name)
    assert loop_body_next.left.value[0].val.name == "a"

    assert isinstance(loop_body_next.left.value[1], Var)
    assert isinstance(loop_body_next.left.value[1].val, Name)
    assert "generated_iter_0" == loop_body_next.left.value[1].val.name

    assert isinstance(loop_body_next.left.value[2], Var)
    assert isinstance(loop_body_next.left.value[2].val, Name)
    assert "sc_" in loop_body_next.left.value[2].val.name 

    assert isinstance(loop_body_next.right, Call)
    assert isinstance(loop_body_next.right.func, Name)
    assert loop_body_next.right.func.name == "next"
    assert len(loop_body_next.right.arguments) == 1
    assert isinstance(loop_body_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_0" == loop_body_next.right.arguments[0].val.name 


    # Calling the comprehension node
    call_node = comp_cast.nodes[0].body[1]
    assert isinstance(call_node, Assignment)
    assert isinstance(call_node.left, Var)
    assert isinstance(call_node.left.val, Name)
    assert call_node.left.val.name == "L"

    assert isinstance(call_node.right, Call)
    assert isinstance(call_node.right.func, Name)
    assert "%comprehension_list_0" == call_node.right.func.name 
    assert len(call_node.right.arguments) == 0

