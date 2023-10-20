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
    unary_operators: &mut Vec<UnaryOperator>,
    projection_operators: &mut Vec<ProjectionOperator>,
    sum_op: &mut Vec<Sum>,
    summand_op: &mut Vec<Summation>,
    mut variable_count: usize,
    mut operation_count: usize,
    mut multiply_count: usize,
    mut addition_count: usize,
    mut subtract_count: usize,
    mut power_count: usize,
) -> (
    Vec<Variable>,
    Vec<UnaryOperator>,
    Vec<ProjectionOperator>,
    Vec<Sum>,
    Vec<Summation>,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    if let MathExpressionTree::Atom(i) = input {
        match i {
            MathExpression::Ci(x) => {
                println!("Ci(x)={:?}", x);
                variable_count += 1;
                println!("variable_count={}", variable_count);
                let variable = Variable {
                    r#type: Type::infer,
                    name: x.content.to_string(),
                };
                variables.push(variable.clone());
            }
            MathExpression::Mn(number) => {
                println!("number={:?}", number);
                variable_count += 1;
                println!("variable_count={}", variable_count);
                let variable = Variable {
                    r#type: Type::Literal,
                    name: number.to_string(),
                };
                variables.push(variable.clone());
            }
            _ => println!("Unhandled"),
        }
    } else if let MathExpressionTree::Cons(head, rest) = input {
        //println!("head={:?}, rest={:?}", head, rest);
        match head {
            Operator::Div => {
                println!("Div");
                operation_count += 1;
                variable_count += 1;
                println!("variable_count={}", variable_count);
                let temp_str = format!("•{}", (variable_count).to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                variables.push(temp_variable.clone());

                let unary = UnaryOperator {
                    src: variable_count,
                    tgt: variable_count + 1,
                    op1: "Div".to_string(),
                };
                unary_operators.push(unary.clone());
            }
            Operator::Add => {
                println!("+");
                //if operation_count == 0 {
                operation_count += 1;
                addition_count += 1;
                let temp_sum = format!("sum_{}", addition_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_sum,
                };
                variables.push(temp_variable.clone());
                let summing = Sum {
                    sum: variable_count,
                };
                sum_op.push(summing.clone());
                variable_count += 1;
                println!("variable_count={}", variable_count);
                //println!("+++++++++++++++++++++++++++++++++");
                //println!("sum_op = {:#?}", sum_op);
                //println!("+++++++++++++++++++++++++++++++++");
                let summands1 = Summation {
                    summand: variable_count,
                    summation: addition_count,
                };
                summand_op.push(summands1.clone());
                //}
            }
            Operator::Power => {
                println!("^");
                operation_count += 1;
                power_count += 1;
                variable_count += 1;
                println!("variable_count={}", variable_count);
                let projection = ProjectionOperator {
                    proj1: variable_count,
                    proj2: variable_count + 3,
                    res: variable_count - 1,
                    op2: "^".to_string(),
                };
                println!("projection_power = {:?}", projection);
                projection_operators.push(projection.clone());
            }
            Operator::Subtract => {
                println!("-");
                operation_count += 1;
                subtract_count += 1;
                variable_count += 1;
                println!("variable_count={}", variable_count);
                let projection = ProjectionOperator {
                    proj1: variable_count,
                    proj2: variable_count + 3,
                    res: variable_count - 1,
                    op2: "-".to_string(),
                };
                //println!("projection = {:?}", projection);
                projection_operators.push(projection.clone());
            }
            Operator::Multiply => {
                operation_count += 1;
                multiply_count += 1;
                variable_count += 1;
                println!("*");

                println!("variable_count={}", variable_count);
                let temp_mult = format!("mult_{}", multiply_count.to_string());
                //if operation_count == 0 {
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_mult,
                };
                variables.push(temp_variable.clone());
                let projection = ProjectionOperator {
                    proj1: variable_count,
                    proj2: variable_count + 3,
                    res: variable_count - 1,
                    op2: "*".to_string(),
                };
                //println!("projection = {:?}", projection);
                projection_operators.push(projection.clone());
                //} else if operation_count != 0 && multiply_count == 0 {
                //let temp_str = format!("•{}", (operation_count - 1).to_string());
                //operation_count += 1;
                //multiply_count += 1;
                /*let summands1 = Summation {
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
                let projection = ProjectionOperator {
                    proj1: variable_count + 1,
                    proj2: variable_count + 2,
                    res: variable_count,
                    op2: "*".to_string(),
                };
                projection_operators.push(projection.clone());
                */
                /*} else {
                    multiply_count += 1;
                    operation_count += 1;
                    let projection = ProjectionOperator {
                        proj1: variable_count,
                        proj2: variable_count + 3,
                        res: variable_count - 1,
                        op2: "*".to_string(),
                    };
                    projection_operators.push(projection.clone());
                }*/
            }
            Operator::Equals => {}
            Operator::Divide => println!("/"),
            Operator::Derivative(Derivative {
                order, var_index, ..
            }) if (order, var_index) == (&1_u8, &1_u8) => {
                println!("diff");
            }
            Operator::Grad => {
                println!("Grad!!!!!!!!!!!!!!!!!");
            }
            _ => {}
        }
        for s in rest.iter() {
            println!("s={:?}", s);
            match s {
                MathExpressionTree::Atom(i) => match i {
                    MathExpression::Ci(x) => {
                        variable_count += 1;
                        println!("variable_count={}", variable_count);
                        let variable = Variable {
                            r#type: Type::infer,
                            name: x.content.to_string(),
                        };
                        variables.push(variable.clone());
                    }
                    MathExpression::Mi(Mi(id)) => {
                        variable_count += 1;
                        println!("variable_count={}", variable_count);
                        let variable = Variable {
                            r#type: Type::infer,
                            name: id.to_string(),
                        };
                        variables.push(variable.clone());
                    }
                    MathExpression::Mn(number) => {
                        variable_count += 1;
                        println!("variable_count={}", variable_count);
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
                    //println!("comp[1]={:?}", comp[1]);

                    match head {
                        Operator::Power => {
                            println!("^^");
                            operation_count += 1;
                            power_count += 1;
                            variable_count += 1;
                            println!("variable_count={}", variable_count);
                            let projection = ProjectionOperator {
                                proj1: variable_count,
                                proj2: variable_count + 3,
                                res: variable_count - 1,
                                op2: "^".to_string(),
                            };
                            projection_operators.push(projection.clone());
                        }
                        Operator::Add => {
                            println!("++");

                            operation_count += 1;
                            addition_count += 1;
                            let temp_sum = format!("sum_{}", addition_count.to_string());
                            let temp_variable = Variable {
                                r#type: Type::infer,
                                name: temp_sum,
                            };
                            variables.push(temp_variable.clone());
                            let summing = Sum {
                                sum: variable_count,
                            };
                            sum_op.push(summing.clone());
                            //println!("+++++++++++++++++++++++++++++++++");
                            //println!("sum_op = {:#?}", sum_op);
                            //println!("+++++++++++++++++++++++++++++++++");
                            variable_count += 1;
                            println!("variable_count={}", variable_count);
                            let summands1 = Summation {
                                summand: variable_count,
                                summation: addition_count,
                            };
                            summand_op.push(summands1.clone());
                            let summands2 = Summation {
                                summand: variable_count + 1,
                                summation: addition_count,
                            };
                            summand_op.push(summands2.clone());
                        }
                        Operator::Subtract => {
                            println!("--");
                            operation_count += 1;
                            subtract_count += 1;
                            variable_count += 1;
                            println!("variable_count={}", variable_count);
                            let projection = ProjectionOperator {
                                proj1: variable_count,
                                proj2: variable_count + 3,
                                res: variable_count - 1,
                                op2: "-".to_string(),
                            };
                            projection_operators.push(projection.clone());
                        }
                        Operator::Multiply => {
                            //if operation_count == 0 {
                            println!("**");
                            operation_count += 1;
                            variable_count += 1;
                            println!("variable_count={}", variable_count);
                            multiply_count += 1;
                            //} else if operation_count != 0 && multiply_count == 0 {
                            //variable_count += 1;
                            let temp_mult = format!("mult_{}", multiply_count.to_string());
                            let temp_variable = Variable {
                                r#type: Type::infer,
                                name: temp_mult,
                            };
                            variables.push(temp_variable.clone());
                            //} else {
                            //    println!("$$$$$$$$$$ANOTHER MULTIPLICATION");
                            //}
                            let projection = ProjectionOperator {
                                proj1: variable_count + 1,
                                proj2: variable_count + 2,
                                res: variable_count,
                                op2: "*".to_string(),
                            };
                            projection_operators.push(projection.clone());
                        }
                        Operator::Equals => {}
                        Operator::Divide => println!("/"),
                        Operator::Derivative(Derivative {
                            order, var_index, ..
                        }) if (order, var_index) == (&1_u8, &1_u8) => {
                            println!("diff");
                        }
                        _ => {}
                    }
                    println!("comp.len()={}", comp.len());
                    for c in comp.iter() {
                        println!("c ={:?}", c);
                        let (
                            v1,
                            un_op,
                            p1,
                            summing_op,
                            summations,
                            mut var_count,
                            mut op_count,
                            mut mul_count,
                            mut add_count,
                            mut sub_count,
                            mut pow_count,
                        ) = to_decapodes_serialization(
                            &c,
                            variables,
                            unary_operators,
                            projection_operators,
                            sum_op,
                            summand_op,
                            variable_count,
                            operation_count,
                            multiply_count,
                            addition_count,
                            subtract_count,
                            power_count,
                        );
                        operation_count += op_count;
                        variable_count += var_count;
                        multiply_count += mul_count;
                        addition_count += add_count;
                        subtract_count += sub_count;
                        power_count += pow_count;
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
        unary_operators.to_vec(),
        projection_operators.to_vec(),
        sum_op.to_vec(),
        summand_op.to_vec(),
        variable_count,
        operation_count,
        multiply_count,
        addition_count,
        subtract_count,
        power_count,
    );
}

pub fn to_wiring_diagram(input: &MathExpressionTree) -> WiringDiagram {
    let mut variables: Vec<Variable> = Vec::new();
    let mut unary_operators: Vec<UnaryOperator> = Vec::new();
    let mut projection_operators: Vec<ProjectionOperator> = Vec::new();
    let mut unary_operators: Vec<UnaryOperator> = Vec::new();
    let mut sum_op: Vec<Sum> = Vec::new();
    let mut summand_op: Vec<Summation> = Vec::new();
    let (
        variables,
        unary_operators,
        projection_operators,
        sum_op,
        summand_op,
        variables_count,
        operation_count,
        multiply_count,
        addition_count,
        subtract_count,
        power_count,
    ) = to_decapodes_serialization(
        &input,
        &mut variables,
        &mut unary_operators,
        &mut projection_operators,
        &mut sum_op,
        &mut summand_op,
        0,
        0,
        0,
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
    println!("wiring_diagram = {:#?}", wiring_diagram);
    let json = to_decapodes_json(wiring_diagram);
    println!("json={:#?}", json);
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
    println!("wiring_diagram = {:#?}", wiring_diagram);
    let json = to_decapodes_json(wiring_diagram);
    println!("json={:#?}", json);
}

#[test]
fn test_serialize_multiply_sup() {
    let input = "
    <math>
        <mi>Γ</mi>
        <msup><mi>H</mi><mrow><mi>n</mi><mo>+</mo><mn>2</mn></mrow></msup>
    </math>
    ";
    let expression = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", expression);
    let s_exp = expression.to_string();
    println!("|||||||||||||||||||||||||||||||||||||");
    println!("S-exp={:?}", s_exp);
    println!("|||||||||||||||||||||||||||||||||||||");
    let mut var: Vec<Variable> = Vec::new();
    let mut proj: Vec<ProjectionOperator> = Vec::new();
    let wiring_diagram = to_wiring_diagram(&expression);
    println!("wiring_diagram = {:#?}", wiring_diagram);
}

#[test]
fn test_serialize_halfar_dome3() {
    let input = "
    <math>
        <mo>&#x2207;</mo>
        <mo>&#x22c5;</mo>
        <mo>(</mo>
        <mi>Γ</mi>
        <msup><mi>H</mi><mrow><mi>n</mi><mo>+</mo><mn>2</mn></mrow></msup>
        <mo>|</mo><mrow><mo>&#x2207;</mo><mi>H</mi></mrow>
        <msup><mo>|</mo>
        <mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msup>
        <mo>&#x2207;</mo><mi>H</mi>
        <mo>)</mo>
    </math>
    ";
    let expression = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", expression);
    let s_exp = expression.to_string();
    println!("|||||||||||||||||||||||||||||||||||||");
    println!("S-exp={:?}", s_exp);
    println!("|||||||||||||||||||||||||||||||||||||");
    let mut var: Vec<Variable> = Vec::new();
    let mut proj: Vec<ProjectionOperator> = Vec::new();
    let wiring_diagram = to_wiring_diagram(&expression);
    println!("wiring_diagram = {:#?}", wiring_diagram);
}
