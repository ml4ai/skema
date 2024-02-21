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
L = [a for a in range(10)]
    """

def comp2():
    return """
D = {a:a*2 for a in range(10)}
    """

def lambda1():
    return """
F = lambda x : x * 10
    """

def generate_cast(test_file_string):
    # use Python to CAST
    out_cast = TS2CAST(test_file_string, from_file=False).out_cast

    return out_cast

def test_comp1():
    comp_cast = generate_cast(comp1())
    
    # Test basic list comprehension
    func_node = comp_cast.nodes[0].body[0]
    assert isinstance(func_node, FunctionDef)
    assert "%comprehension_list_" in func_node.name.name
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
    assert "generated_iter" in loop_pre_iter.left.val.name

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
    assert "generated_iter_" in loop_pre_next.left.value[1].val.name

    assert isinstance(loop_pre_next.left.value[2], Var)
    assert isinstance(loop_pre_next.left.value[2].val, Name)
    assert "sc_" in loop_pre_next.left.value[2].val.name 

    assert isinstance(loop_pre_next.right, Call)
    assert isinstance(loop_pre_next.right.func, Name)
    assert loop_pre_next.right.func.name == "next"
    assert len(loop_pre_next.right.arguments) == 1
    assert isinstance(loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_" in loop_pre_next.right.arguments[0].val.name 


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
    loop_body_append = func_loop_body[0]
    assert isinstance(loop_body_append, Call)
    assert isinstance(loop_body_append.func, Attribute)
    assert isinstance(loop_body_append.func.attr, Name)
    assert loop_body_append.func.attr.name == "append"

    assert isinstance(loop_body_append.func.value, Name)
    assert "list__temp_" in loop_body_append.func.value.name

    assert len(loop_body_append.arguments) == 1
    assert isinstance(loop_body_append.arguments[0], Var)
    assert loop_body_append.arguments[0].val.name == "a"

    loop_body_next = func_loop_body[1]
    assert isinstance(loop_body_next, Assignment)
    assert isinstance(loop_body_next.left, CASTLiteralValue)
    assert loop_body_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_body_next.left.value[0], Var)
    assert isinstance(loop_body_next.left.value[0].val, Name)
    assert loop_body_next.left.value[0].val.name == "a"

    assert isinstance(loop_body_next.left.value[1], Var)
    assert isinstance(loop_body_next.left.value[1].val, Name)
    assert "generated_iter_" in loop_body_next.left.value[1].val.name

    assert isinstance(loop_body_next.left.value[2], Var)
    assert isinstance(loop_body_next.left.value[2].val, Name)
    assert "sc_" in loop_body_next.left.value[2].val.name 

    assert isinstance(loop_body_next.right, Call)
    assert isinstance(loop_body_next.right.func, Name)
    assert loop_body_next.right.func.name == "next"
    assert len(loop_body_next.right.arguments) == 1
    assert isinstance(loop_body_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_" in loop_body_next.right.arguments[0].val.name 

    call_node = comp_cast.nodes[0].body[1]
    assert isinstance(call_node, Assignment)
    assert isinstance(call_node.left, Var)
    assert isinstance(call_node.left.val, Name)
    assert call_node.left.val.name == "L"

    assert isinstance(call_node.right, Call)
    assert isinstance(call_node.right.func, Name)
    assert "%comprehension_list_" in call_node.right.func.name 
    assert len(call_node.right.arguments) == 0

def test_comp2():
    comp_cast = generate_cast(comp2())
    
    # Test basic dict comprehension
    # Very similar to list comp, but using dictionaries instead
    func_node = comp_cast.nodes[0].body[0]
    assert isinstance(func_node, FunctionDef)
    assert "%comprehension_dict_" in func_node.name.name
    assert len(func_node.func_args) == 0
    
    func_node_body = func_node.body
    func_asg = func_node.body[0]
    func_ret = func_node.body[2]
    assert isinstance(func_asg, Assignment)
    assert isinstance(func_asg.left, Var)
    assert isinstance(func_asg.left.val, Name)
    assert "dict__temp_" == func_asg.left.val.name

    assert isinstance(func_asg.right, CASTLiteralValue)
    assert func_asg.right.value_type == StructureType.MAP
    assert func_asg.right.value == {}

    assert isinstance(func_ret, ModelReturn)
    assert isinstance(func_ret.value, Name)
    assert "dict__temp_" == func_ret.value.name 

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
    assert "generated_iter" in loop_pre_iter.left.val.name

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
    assert "generated_iter_" in loop_pre_next.left.value[1].val.name

    assert isinstance(loop_pre_next.left.value[2], Var)
    assert isinstance(loop_pre_next.left.value[2].val, Name)
    assert "sc_" in loop_pre_next.left.value[2].val.name 

    assert isinstance(loop_pre_next.right, Call)
    assert isinstance(loop_pre_next.right.func, Name)
    assert loop_pre_next.right.func.name == "next"
    assert len(loop_pre_next.right.arguments) == 1
    assert isinstance(loop_pre_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_" in loop_pre_next.right.arguments[0].val.name 


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
    loop_body_set = func_loop_body[0]
    assert isinstance(loop_body_set, Assignment)
    assert isinstance(loop_body_set.left, Var)
    assert isinstance(loop_body_set.left.val, Name)
    assert "dict__temp_" in loop_body_set.left.val.name

    assert isinstance(loop_body_set.right, Call)
    assert isinstance(loop_body_set.right.func, Name)
    assert loop_body_set.right.func.name == "_set" 
    assert len(loop_body_set.right.arguments) == 3

    assert isinstance(loop_body_set.right.arguments[0], Name)
    assert "dict__temp_" in loop_body_set.right.arguments[0].name 

    assert isinstance(loop_body_set.right.arguments[1], Name)
    assert loop_body_set.right.arguments[1].name == "a"
    
    assert isinstance(loop_body_set.right.arguments[2], Operator)
    assert loop_body_set.right.arguments[2].op == "ast.Mult"
    assert isinstance(loop_body_set.right.arguments[2].operands[0], Name)
    assert loop_body_set.right.arguments[2].operands[0].name == "a"
    assert isinstance(loop_body_set.right.arguments[2].operands[1], CASTLiteralValue)
    assert loop_body_set.right.arguments[2].operands[1].value == '2'
    
    loop_body_next = func_loop_body[1]
    assert isinstance(loop_body_next, Assignment)
    assert isinstance(loop_body_next.left, CASTLiteralValue)
    assert loop_body_next.left.value_type == StructureType.TUPLE
    assert isinstance(loop_body_next.left.value[0], Var)
    assert isinstance(loop_body_next.left.value[0].val, Name)
    assert loop_body_next.left.value[0].val.name == "a"

    assert isinstance(loop_body_next.left.value[1], Var)
    assert isinstance(loop_body_next.left.value[1].val, Name)
    assert "generated_iter_" in loop_body_next.left.value[1].val.name

    assert isinstance(loop_body_next.left.value[2], Var)
    assert isinstance(loop_body_next.left.value[2].val, Name)
    assert "sc_" in loop_body_next.left.value[2].val.name 

    assert isinstance(loop_body_next.right, Call)
    assert isinstance(loop_body_next.right.func, Name)
    assert loop_body_next.right.func.name == "next"
    assert len(loop_body_next.right.arguments) == 1
    assert isinstance(loop_body_next.right.arguments[0], Var) # NOTE: This should be Name, not Var, but it's ok for now
    assert "generated_iter_" in loop_body_next.right.arguments[0].val.name 

    call_node = comp_cast.nodes[0].body[1]
    assert isinstance(call_node, Assignment)
    assert isinstance(call_node.left, Var)
    assert isinstance(call_node.left.val, Name)
    assert call_node.left.val.name == "D"

    assert isinstance(call_node.right, Call)
    assert isinstance(call_node.right.func, Name)
    assert "%comprehension_dict_" in call_node.right.func.name 
    assert len(call_node.right.arguments) == 0


def test_lambda1():
    lambda_cast = generate_cast(lambda1())

    # Test basic lambda
    func_node = lambda_cast.nodes[0].body[0]
    assert isinstance(func_node, FunctionDef)
    assert "%lambda_" in func_node.name.name
    assert len(func_node.func_args) == 1
    assert isinstance(func_node.func_args[0], Var)
    assert func_node.func_args[0].val.name == "x"

    func_body = func_node.body[0]
    assert isinstance(func_body, ModelReturn)
    
    assert isinstance(func_body.value, Operator)
    assert isinstance(func_body.value.operands[0], Name)
    assert func_body.value.operands[0].name == "x"

    assert isinstance(func_body.value.operands[1], CASTLiteralValue)
    assert func_body.value.operands[1].value_type == ScalarType.INTEGER
    assert func_body.value.operands[1].value == '10'

    call_node = lambda_cast.nodes[0].body[1]
    assert isinstance(call_node, Assignment)
    assert isinstance(call_node.left, Var)
    assert isinstance(call_node.left.val, Name)
    assert call_node.left.val.name == "F"

    assert isinstance(call_node.right, Call)
    assert isinstance(call_node.right.func, Name)
    assert "%lambda_" in call_node.right.func.name 

    assert len(call_node.right.arguments) == 1
    assert isinstance(call_node.right.arguments[0], Name)
    assert call_node.right.arguments[0].name == "x"

