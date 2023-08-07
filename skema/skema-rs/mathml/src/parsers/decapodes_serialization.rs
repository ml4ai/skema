use crate::parsers::math_expression_tree::MathExpressionTree;
use crate::{
    ast::{
        operator::{Derivative, Operator},
        Math, MathExpression, Mi, Mrow,
    },
    parsers::interpreted_mathml::interpreted_math,
};

use derive_new::new;
use nom::error::Error;
use serde::{Deserialize, Serialize};
use serde_json::{json, Result};
use std::{fmt, fs::File, io::Write, str::FromStr};

#[cfg(test)]
use crate::parsers::first_order_ode::{first_order_ode, FirstOrderODE};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Type {
    Form0,
    Form1,
    infer,
    Constant,
    Literal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    r#type: Type,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnaryOperator {
    pub src: usize,
    pub tgt: usize,
    pub op1: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionOperator {
    pub proj1: usize,
    pub proj2: usize,
    pub res: usize,
    pub op2: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sum {
    pub sum: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summation {
    pub summand: usize,
    pub summation: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WiringDiagram {
    pub Var: Vec<Variable>,
    pub Op1: Vec<UnaryOperator>,
    pub Op2: Vec<ProjectionOperator>,
    pub Σ: Vec<Sum>,
    pub Summand: Vec<Summation>,
}

pub fn to_decapodes_serialization(
    input: &MathExpressionTree,
    variables: &mut Vec<Variable>,
    projection_operators: &mut Vec<ProjectionOperator>,
    sum_op: &mut Vec<Sum>,
    summand_op: &mut Vec<Summation>,
    mut variable_count: usize,
    mut operation_count: usize,
    mut multiply_count: usize,
) -> (
    Vec<Variable>,
    Vec<ProjectionOperator>,
    Vec<Sum>,
    Vec<Summation>,
    usize,
    usize,
    usize,
) {
    if let MathExpressionTree::Atom(i) = input {
        println!("i={:?}", i);
        println!("----INSIDE ELSE IF");
        match i {
            MathExpression::Ci(x) => {
                variable_count += 1;
                println!("---- Ci INSIDE ELSE IF");
                println!("x.content={}", x.content);
                println!("variable_count in Ci(x)= {}", variable_count);
                let variable = Variable {
                    r#type: Type::infer,
                    name: x.content.to_string(),
                };
                variables.push(variable.clone());
            }
            _ => println!("Unhandled"),
        }
    } else if let MathExpressionTree::Cons(head, rest) = input {
        println!("head={:?}, rest={:?}", head, rest);
        match head {
            Operator::Add => {
                println!("+");
                operation_count += 1;
                println!("op_count in + = {}", operation_count);
            }
            Operator::Subtract => println!("-"),
            Operator::Multiply => {
                println!("*");
                if operation_count == 0 {
                    operation_count += 1;
                    multiply_count += 1;
                    variable_count += 1;
                    let temp_variable = Variable {
                        r#type: Type::infer,
                        name: "mult_1".to_string(),
                    };
                    variables.push(temp_variable.clone());
                    let projection = ProjectionOperator {
                        proj1: variable_count,
                        proj2: variable_count + 3,
                        res: variable_count - 1,
                        op2: "*".to_string(),
                    };
                    println!("projection = {:?}", projection);
                    projection_operators.push(projection.clone());
                } else if operation_count != 0 && multiply_count == 0 {
                    let temp_str = format!("•{}", (operation_count - 1).to_string());
                    operation_count += 1;
                    multiply_count += 1;
                    println!("B VARIABLE COUNT ====== {}", variable_count);
                    let summands1 = Summation {
                        summand: variable_count,
                        summation: 1,
                    };
                    summand_op.push(summands1.clone());
                    variable_count += 1;
                    let temp_variable = Variable {
                        r#type: Type::infer,
                        name: temp_str,
                    };
                    variables.push(temp_variable.clone());
                    let summands2 = Summation {
                        summand: variable_count,
                        summation: 1,
                    };
                    summand_op.push(summands2.clone());
                    println!("VARIABLE COUNT ====== {}", variable_count);
                    let projection = ProjectionOperator {
                        proj1: variable_count + 1,
                        proj2: variable_count + 2,
                        res: variable_count,
                        op2: "*".to_string(),
                    };
                    println!("projection = {:?}", projection);
                    projection_operators.push(projection.clone());
                } else {
                    multiply_count += 1;
                    operation_count += 1;
                    println!(">>>>>>>>ANOTHER MULTIPLICATION");
                    println!("variable_count={}", variable_count);
                    let projection = ProjectionOperator {
                        proj1: variable_count,
                        proj2: variable_count + 3,
                        res: variable_count - 1,
                        op2: "*".to_string(),
                    };
                    println!(">>>>>>projection = {:?}", projection);
                    projection_operators.push(projection.clone());
                }

                println!("var_count={}", variable_count);
                println!("op_count in *={}", operation_count);
                /*let projection = ProjectionOperator {
                    proj1: variable_count,
                    proj2: variable_count + 3,
                    res: variable_count - 1,
                    op2: "*".to_string(),
                };
                println!("projection = {:?}", projection);*/
            }
            Operator::Equals => {}
            Operator::Divide => println!("/"),
            Operator::Derivative(Derivative { order, var_index, .. })
                if (order, var_index) == (&1_u8, &1_u8) =>
            {
                println!("diff");
            }
            _ => {}
        }
        for s in rest.iter() {
            println!("---------");
            println!("s={:?}", s);
            match s {
                MathExpressionTree::Atom(i) => match i {
                    MathExpression::Ci(x) => {
                        variable_count += 1;
                        println!("x.content={}", x.content);
                        println!("variable_count in Ci(x)= {}", variable_count);
                        let variable = Variable {
                            r#type: Type::infer,
                            name: x.content.to_string(),
                        };
                        variables.push(variable.clone());
                    }
                    MathExpression::Mi(Mi(id)) => {
                        println!("id={}", id);
                        println!("variable_count in Mi(Mi(id) = {}", variable_count);
                        variable_count += 1;
                        let variable = Variable {
                            r#type: Type::infer,
                            name: id.to_string(),
                        };
                        variables.push(variable.clone());
                    }
                    MathExpression::Mn(number) => {
                        println!("number={}", number);
                        variable_count += 1;
                        println!("variable_count in Mn(number)= {}", variable_count);
                        let variable = Variable {
                            r#type: Type::Literal,
                            name: number.to_string(),
                        };
                        variables.push(variable.clone());
                    }
                    MathExpression::Mrow(_) => {
                        panic!("All Mrows should have been removed by now!");
                    }
                    t => panic!("Unhandled MathExpression: {:?}", t),
                },
                MathExpressionTree::Cons(head, comp) => {
                    println!("Operator={:?} , comp = {:?}", head, comp);
                    println!("Operator={:?} , comp = {:?}", head, comp.to_vec());
                    println!("comp[0]={:?}", comp[0]);
                    println!("comp[1]={:?}", comp[1]);

                    match head {
                        Operator::Add => {
                            println!("+");
                            if operation_count == 0 {
                                operation_count += 1;
                                let summing = Sum {
                                    sum: variable_count,
                                };
                                println!("summing={:?}", summing);
                                sum_op.push(summing.clone());
                            }
                            println!("---op_count in + = {}", operation_count);
                        }
                        Operator::Subtract => println!("-"),
                        Operator::Multiply => {
                            println!("**");
                            println!("--op_count in *={}", operation_count);
                            if operation_count == 0 {
                                operation_count += 1;
                                variable_count += 1;
                                multiply_count += 1;
                                let temp_variable = Variable {
                                    r#type: Type::infer,
                                    name: "mult_1".to_string(),
                                };
                                variables.push(temp_variable.clone());
                                println!("variable_count in temp_variable= {}", variable_count);
                                println!("multiply_count = {}", multiply_count);
                            } else if operation_count != 0 && multiply_count == 0 {
                                println!("operation_count for • = {}", operation_count);
                                variable_count += 1;
                                let temp_str = format!("•{}", operation_count.to_string());
                                let temp_variable = Variable {
                                    r#type: Type::infer,
                                    name: temp_str,
                                };
                                println!("variable_count in temp varible '•' = {}", variable_count);
                                operation_count += 1;
                            } else {
                                println!("$$$$$$$$$$ANOTHER MULTIPLICATION");
                            }
                            println!("--var_count={}", variable_count);
                            println!("--op_count={}", operation_count);
                            let projection = ProjectionOperator {
                                proj1: variable_count + 1,
                                proj2: variable_count + 2,
                                res: variable_count,
                                op2: "*".to_string(),
                            };
                            println!("projection = {:?}", projection);
                            projection_operators.push(projection.clone());
                        }
                        Operator::Equals => {}
                        Operator::Divide => println!("/"),
                        Operator::Derivative(Derivative { order, var_index , ..})
                            if (order, var_index) == (&1_u8, &1_u8) =>
                        {
                            println!("diff");
                        }
                        _ => {}
                    }
                    println!("comp.len()={}", comp.len());
                    for c in comp.iter() {
                        println!("c ={:?}", c);
                        let (
                            v1,
                            p1,
                            summing_op,
                            summations,
                            mut var_count,
                            mut op_count,
                            mut mul_count,
                        ) = to_decapodes_serialization(
                            &c,
                            variables,
                            projection_operators,
                            sum_op,
                            summand_op,
                            variable_count,
                            operation_count,
                            multiply_count,
                        );
                        operation_count += op_count;
                        variable_count = var_count;
                        multiply_count = mul_count;
                        println!("v1={:?}", v1);
                        println!(
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 operation_count={}",
                            operation_count
                        );
                        println!("end variable_count={}", variable_count);
                    }
                }
            }
        }
    }

    println!("variables={:?}", variables);
    return (
        variables.to_vec(),
        projection_operators.to_vec(),
        sum_op.to_vec(),
        summand_op.to_vec(),
        variable_count,
        operation_count,
        multiply_count,
    );
}

pub fn to_wiring_diagram(input: &MathExpressionTree) -> WiringDiagram {
    let mut variables: Vec<Variable> = Vec::new();
    let mut projection_operators: Vec<ProjectionOperator> = Vec::new();
    let mut unary_operators: Vec<UnaryOperator> = Vec::new();
    let mut sum_op: Vec<Sum> = Vec::new();
    let mut summand_op: Vec<Summation> = Vec::new();
    let (
        variables,
        projection_operators,
        sum_op,
        summand_op,
        variables_count,
        operation_count,
        multiply_count,
    ) = to_decapodes_serialization(
        &input,
        &mut variables,
        &mut projection_operators,
        &mut sum_op,
        &mut summand_op,
        0,
        0,
        0,
    );

    return WiringDiagram {
        Var: variables,
        Op1: unary_operators,
        Op2: projection_operators,
        Σ: sum_op,
        Summand: summand_op,
    };
}

pub fn to_decapodes_json(input: WiringDiagram) -> Result<()> {
    let json_wiring_diagram = serde_json::to_string_pretty(&input).unwrap();
    println!("json_wiring_diagran={:?}", json_wiring_diagram);

    Ok(())
}

#[test]
fn test_serialize1() {
    let input = "
    <math>
        <msub><mi>C</mi><mi>o</mi></msub>
        <mo>=</mo>
        <msub><mi>ρ</mi><mi>w</mi></msub>
        <msub><mi>c</mi><mi>w</mi></msub>
        <mi>dz</mi>
    </math>
    ";
    let expression = input.parse::<MathExpressionTree>().unwrap();
    println!("expression={:?}", expression);
    println!("expression.to_string()={}", expression.to_string());
    let mut var: Vec<Variable> = Vec::new();
    let mut proj: Vec<ProjectionOperator> = Vec::new();
    let wiring_diagram = to_wiring_diagram(&expression);
    println!("wiring_diagram = {:?}", wiring_diagram);
    let json = to_decapodes_json(wiring_diagram);
    println!("json={:?}", json);
}

#[test]
fn test_serialize_APlusBT() {
    let input = "
    <math>
        <mrow><mi>OLR</mi></mrow>
        <mo>=</mo>
        <mi>A</mi>
        <mo>+</mo>
        <mi>B</mi>
        <mi>T</mi>
    </math>
    ";
    let expression = input.parse::<MathExpressionTree>().unwrap();
    println!("expression={:?}", expression);
    println!("expression.to_string()={}", expression.to_string());
    let mut var: Vec<Variable> = Vec::new();
    let mut proj: Vec<ProjectionOperator> = Vec::new();
    let wiring_diagram = to_wiring_diagram(&expression);
    println!("wiring_diagram = {:?}", wiring_diagram);
    let json = to_decapodes_json(wiring_diagram);
    println!("json={:?}", json);
}
