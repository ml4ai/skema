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
use std::{fmt, io, str::FromStr};

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
    mut variable_count: usize,
    mut operation_count: usize,
) -> (Vec<Variable>, Vec<ProjectionOperator>, usize, usize) {
    //let mut variables: Vec<Variable> = Vec::new();
    //let mut projection_operators: Vec<ProjectionOperator> = Vec::new();
    if let MathExpressionTree::Cons(head, rest) = input {
        //let mut variables: Vec<Variable> = Vec::new();
        //let mut projection_operators: Vec<ProjectionOperator> = Vec::new();
        //let mut variable_count = 0;
        //let mut operation_count = 0;
        match head {
            Operator::Add => println!("+"),
            Operator::Subtract => println!("-"),
            Operator::Multiply => {
                println!("*");
                operation_count += 1;
                variable_count += 1;
                let temp_str = format!("•{}", operation_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                variables.push(temp_variable.clone());
                println!("temp_variable={:?}", temp_variable);
                let projection = ProjectionOperator {
                    proj1: variable_count,
                    proj2: variable_count,
                    res: variable_count,
                    op2: "*".to_string(),
                };
                println!("projection = {:?}", projection);
                projection_operators.push(projection.clone());
            }
            Operator::Equals => {}
            Operator::Divide => println!("/"),
            Operator::Derivative(Derivative { order, var_index })
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
                MathExpressionTree::Cons(Operator, comp) => {
                    return to_decapodes_serialization(
                        &MathExpressionTree::Cons(Operator::Multiply, comp.to_vec()),
                        //s,
                        variables,
                        projection_operators,
                        variable_count,
                        operation_count,
                    );
                }
            }
        }
        println!("variables={:?}", variables);
    }

    return (
        variables.to_vec(),
        projection_operators.to_vec(),
        variable_count,
        operation_count,
    );
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
    //let serialize = expression.to_decapodes_serialization();
    //let serialize = to_decapodes_serialization(expression);
    let serialize = to_decapodes_serialization(&expression, &mut var, &mut proj, 0, 0);
    println!("|||||||||||||||||||||||||||||||||||||||");
    println!("serialize = {:?}", serialize);
}
