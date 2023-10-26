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
pub struct TableCounts {
    pub variable_count: usize,
    pub operation_count: usize,
    pub sum_count: usize,
    pub multi_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tables {
    pub variables: Vec<Variable>,
    pub projection_operators: Vec<ProjectionOperator>,
    pub unary_operators: Vec<UnaryOperator>,
    pub sum_op: Vec<Sum>,
    pub summand_op: Vec<Summation>,
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
    tables: &mut Tables,
    table_counts: &mut TableCounts,
) -> (usize) {
    if let MathExpressionTree::Atom(i) = input {
        match i {
            MathExpression::Ci(x) => {
                let index = tables
                    .variables
                    .iter()
                    .position(|variable| variable.name == x.content.to_string());
                match index {
                    Some(idx) => idx + 1,
                    None => {
                        let variable = Variable {
                            r#type: Type::infer,
                            name: x.content.to_string(),
                        };
                        tables.variables.push(variable.clone());
                        table_counts.variable_count += 1;
                        table_counts.variable_count
                    }
                }
            }
            MathExpression::Mn(number) => {
                let index = tables
                    .variables
                    .iter()
                    .position(|variable| variable.name == number.to_string());
                match index {
                    Some(idx) => idx + 1,
                    None => {
                        let variable = Variable {
                            r#type: Type::infer,
                            name: number.to_string(),
                        };
                        tables.variables.push(variable.clone());
                        table_counts.variable_count += 1;
                        table_counts.variable_count
                    }
                }
            }
            _ => 0,
        }
    } else if let MathExpressionTree::Cons(head, rest) = input {
        match head {
            Operator::Equals => {
                to_decapodes_serialization(&rest[0], tables, table_counts);
                to_decapodes_serialization(&rest[1], tables, table_counts);
                0
            }
            Operator::Div => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", (table_counts.operation_count).to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();

                let unary = UnaryOperator {
                    src: to_decapodes_serialization(&rest[0], tables, table_counts),
                    tgt: tgt_idx,
                    op1: "Div".to_string(),
                };
                tables.unary_operators.push(unary.clone());
                tgt_idx
            }
            Operator::Abs => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", (table_counts.operation_count).to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();

                let unary = UnaryOperator {
                    src: to_decapodes_serialization(&rest[0], tables, table_counts),
                    tgt: tgt_idx,
                    op1: "Abs".to_string(),
                };
                tables.unary_operators.push(unary.clone());
                tgt_idx
            }
            Operator::Add => {
                table_counts.sum_count += 1;
                let temp_sum = format!("sum_{}", table_counts.sum_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_sum,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let summing = Sum {
                    sum: table_counts.variable_count,
                };
                tables.sum_op.push(summing.clone());
                let tgt_idx = table_counts.sum_count.clone();

                for r in rest.iter() {
                    let summands = Summation {
                        summand: to_decapodes_serialization(r, tables, table_counts),
                        summation: tgt_idx,
                    };
                    tables.summand_op.push(summands.clone());
                }
                tgt_idx
            }
            Operator::Power => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", table_counts.operation_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();

                let projection = ProjectionOperator {
                    proj1: to_decapodes_serialization(&rest[0], tables, table_counts),
                    proj2: to_decapodes_serialization(&rest[1], tables, table_counts),
                    res: tgt_idx,
                    op2: "^".to_string(),
                };
                tables.projection_operators.push(projection.clone());
                tgt_idx
            }
            Operator::Subtract => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", table_counts.operation_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();
                let projection = ProjectionOperator {
                    proj1: to_decapodes_serialization(&rest[0], tables, table_counts),
                    proj2: to_decapodes_serialization(&rest[1], tables, table_counts),
                    res: tgt_idx,
                    op2: "-".to_string(),
                };
                tables.projection_operators.push(projection.clone());
                tgt_idx
            }
            Operator::Multiply => {
                table_counts.multi_count += 1;
                let temp_multi = format!("mult_{}", table_counts.multi_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_multi,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();
                let projection = ProjectionOperator {
                    proj1: to_decapodes_serialization(&rest[0], tables, table_counts),
                    proj2: to_decapodes_serialization(&rest[1], tables, table_counts),
                    res: tgt_idx,
                    op2: "*".to_string(),
                };
                tables.projection_operators.push(projection.clone());
                tgt_idx
            }
            Operator::Divide => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", table_counts.operation_count.to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();
                let projection = ProjectionOperator {
                    proj1: to_decapodes_serialization(&rest[0], tables, table_counts),
                    proj2: to_decapodes_serialization(&rest[1], tables, table_counts),
                    res: tgt_idx,
                    op2: "/".to_string(),
                };
                tables.projection_operators.push(projection.clone());
                tgt_idx
            }
            Operator::Derivative(Derivative {
                order,
                var_index,
                bound_var,
            }) => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", (table_counts.operation_count).to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();
                let derivative_str = format!("D({},{})", order.to_string(), bound_var.to_string());
                let unary = UnaryOperator {
                    src: to_decapodes_serialization(&rest[0], tables, table_counts),
                    tgt: tgt_idx,
                    op1: derivative_str,
                };
                tables.unary_operators.push(unary.clone());
                tgt_idx
            }
            Operator::Grad => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", (table_counts.operation_count).to_string());
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count.clone();
                let unary = UnaryOperator {
                    src: to_decapodes_serialization(&rest[0], tables, table_counts),
                    tgt: tgt_idx,
                    op1: "Grad".to_string(),
                };
                tables.unary_operators.push(unary.clone());
                tgt_idx
            }
            _ => {
                return 0;
            }
        }
    } else {
        return 0;
    }
}

pub fn to_wiring_diagram(input: &MathExpressionTree) -> WiringDiagram {
    let mut table_counts = TableCounts {
        variable_count: 0,
        operation_count: 0,
        sum_count: 0,
        multi_count: 0,
    };

    let mut tables = Tables {
        variables: Vec::new(),
        projection_operators: Vec::new(),
        unary_operators: Vec::new(),
        sum_op: Vec::new(),
        summand_op: Vec::new(),
    };

    to_decapodes_serialization(&input, &mut tables, &mut table_counts);

    return WiringDiagram {
        Var: tables.variables,
        Op1: tables.unary_operators,
        Op2: tables.projection_operators,
        Σ: tables.sum_op,
        Summand: tables.summand_op,
    };
}

pub fn to_decapodes_json(input: WiringDiagram) -> Result<()> {
    let json_wiring_diagram = serde_json::to_string(&input).unwrap();
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
fn test_serialize_hackathon2_scenario1_eq5() {
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>D</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mi>α</mi>
        <mi>ρ</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
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
    let json = to_decapodes_json(wiring_diagram);
    println!("json={:?}", json);
}

#[test]
fn test_serialize_halfar_dome() {
    let input = "
    <math>
        <mfrac><mrow><mi>∂</mi><mi>H</mi></mrow><mrow><mi>∂</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo>
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
    let s_exp = expression.to_string();
    println!("S-exp={:?}", s_exp);
    let wiring_diagram = to_wiring_diagram(&expression);
    println!("wiring_diagram = {:#?}", wiring_diagram);
    let json = to_decapodes_json(wiring_diagram);
    println!("json={:#?}", json);
}
