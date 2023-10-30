use crate::ast::{
    operator::{Derivative, Operator},
    MathExpression,
};
use crate::parsers::math_expression_tree::MathExpressionTree;

use serde::{Deserialize, Serialize};

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
) -> usize {
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
                    .position(|variable| variable.name == *number);
                match index {
                    Some(idx) => idx + 1,
                    None => {
                        let variable = Variable {
                            r#type: Type::Literal,
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
                to_decapodes_serialization(&rest[1], tables, table_counts);
                to_decapodes_serialization(&rest[0], tables, table_counts);
                0
            }
            Operator::Div => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", (table_counts.operation_count));
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;

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
                let temp_str = format!("•{}", (table_counts.operation_count));
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;

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
                let temp_sum = format!("sum_{}", table_counts.sum_count);
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
                let tgt_idx = table_counts.variable_count;
                let sum_tgt_idx = table_counts.sum_count;

                for r in rest.iter() {
                    let summands = Summation {
                        summand: to_decapodes_serialization(r, tables, table_counts),
                        summation: sum_tgt_idx,
                    };
                    tables.summand_op.push(summands.clone());
                }
                tgt_idx
            }
            Operator::Power => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", table_counts.operation_count);
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;

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
                let temp_str = format!("•{}", table_counts.operation_count);
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;
                if rest.len() == 1 {
                    let unary = UnaryOperator {
                        src: to_decapodes_serialization(&rest[0], tables, table_counts),
                        tgt: tgt_idx,
                        op1: "-".to_string(),
                    };
                    tables.unary_operators.push(unary.clone());
                } else {
                    let projection = ProjectionOperator {
                        proj1: to_decapodes_serialization(&rest[0], tables, table_counts),
                        proj2: to_decapodes_serialization(&rest[1], tables, table_counts),
                        res: tgt_idx,
                        op2: "-".to_string(),
                    };
                    tables.projection_operators.push(projection.clone());
                }
                tgt_idx
            }
            Operator::Multiply => {
                table_counts.multi_count += 1;
                let temp_multi = format!("mult_{}", table_counts.multi_count);
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_multi,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;
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
                let temp_str = format!("•{}", table_counts.operation_count);
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;
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
                var_index: _,
                bound_var,
            }) => {
                table_counts.operation_count += 1;
                let temp_str = format!("•{}", (table_counts.operation_count));
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;
                let derivative_str = format!("D({},{})", order, bound_var);
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
                let temp_str = format!("•{}", (table_counts.operation_count));
                let temp_variable = Variable {
                    r#type: Type::infer,
                    name: temp_str,
                };
                tables.variables.push(temp_variable.clone());
                table_counts.variable_count += 1;
                let tgt_idx = table_counts.variable_count;
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

    to_decapodes_serialization(input, &mut tables, &mut table_counts);

    WiringDiagram {
        Var: tables.variables,
        Op1: tables.unary_operators,
        Op2: tables.projection_operators,
        Σ: tables.sum_op,
        Summand: tables.summand_op,
    }
}

pub fn to_decapodes_json(input: WiringDiagram) -> String {
    serde_json::to_string(&input).unwrap()
    //Ok(json_wiring_diagram)
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
    let wiring_diagram = to_wiring_diagram(&expression);
    let json = to_decapodes_json(wiring_diagram);
    assert_eq!(json, "{\"Var\":[{\"type\":\"infer\",\"name\":\"mult_1\"},{\"type\":\"infer\",\"name\":\"mult_2\"},{\"type\":\"infer\",\"name\":\"ρ_{w}\"},{\"type\":\"infer\",\"name\":\"c_{w}\"},{\"type\":\"infer\",\"name\":\"dz\"},{\"type\":\"infer\",\"name\":\"C_{o}\"}],\"Op1\":[],\"Op2\":[{\"proj1\":3,\"proj2\":4,\"res\":2,\"op2\":\"*\"},{\"proj1\":2,\"proj2\":5,\"res\":1,\"op2\":\"*\"}],\"Σ\":[],\"Summand\":[]}");
}

#[test]
fn test_serialize_aplus_bt() {
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
    let wiring_diagram = to_wiring_diagram(&expression);
    let json = to_decapodes_json(wiring_diagram);
    assert_eq!(json, "{\"Var\":[{\"type\":\"infer\",\"name\":\"sum_1\"},{\"type\":\"infer\",\"name\":\"A\"},{\"type\":\"infer\",\"name\":\"mult_1\"},{\"type\":\"infer\",\"name\":\"B\"},{\"type\":\"infer\",\"name\":\"T\"},{\"type\":\"infer\",\"name\":\"OLR\"}],\"Op1\":[],\"Op2\":[{\"proj1\":4,\"proj2\":5,\"res\":3,\"op2\":\"*\"}],\"Σ\":[{\"sum\":1}],\"Summand\":[{\"summand\":2,\"summation\":1},{\"summand\":3,\"summation\":1}]}");
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
    let wiring_diagram = to_wiring_diagram(&expression);
    let json = to_decapodes_json(wiring_diagram);
    assert_eq!(json, "{\"Var\":[{\"type\":\"infer\",\"name\":\"mult_1\"},{\"type\":\"infer\",\"name\":\"Γ\"},{\"type\":\"infer\",\"name\":\"•1\"},{\"type\":\"infer\",\"name\":\"H\"},{\"type\":\"infer\",\"name\":\"sum_1\"},{\"type\":\"infer\",\"name\":\"n\"},{\"type\":\"Literal\",\"name\":\"2\"}],\"Op1\":[],\"Op2\":[{\"proj1\":4,\"proj2\":5,\"res\":3,\"op2\":\"^\"},{\"proj1\":2,\"proj2\":3,\"res\":1,\"op2\":\"*\"}],\"Σ\":[{\"sum\":5}],\"Summand\":[{\"summand\":6,\"summation\":1},{\"summand\":7,\"summation\":1}]}");
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
    let wiring_diagram = to_wiring_diagram(&expression);
    let json = to_decapodes_json(wiring_diagram);
    assert_eq!(json, "{\"Var\":[{\"type\":\"infer\",\"name\":\"mult_1\"},{\"type\":\"infer\",\"name\":\"mult_2\"},{\"type\":\"infer\",\"name\":\"α\"},{\"type\":\"infer\",\"name\":\"ρ\"},{\"type\":\"infer\",\"name\":\"I\"},{\"type\":\"infer\",\"name\":\"•1\"},{\"type\":\"infer\",\"name\":\"D\"}],\"Op1\":[{\"src\":7,\"tgt\":6,\"op1\":\"D(1,t)\"}],\"Op2\":[{\"proj1\":3,\"proj2\":4,\"res\":2,\"op2\":\"*\"},{\"proj1\":2,\"proj2\":5,\"res\":1,\"op2\":\"*\"}],\"Σ\":[],\"Summand\":[]}");
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
    let wiring_diagram = to_wiring_diagram(&expression);
    let json = to_decapodes_json(wiring_diagram);
    assert_eq!(json,"{\"Var\":[{\"type\":\"infer\",\"name\":\"•1\"},{\"type\":\"infer\",\"name\":\"mult_1\"},{\"type\":\"infer\",\"name\":\"mult_2\"},{\"type\":\"infer\",\"name\":\"mult_3\"},{\"type\":\"infer\",\"name\":\"Γ\"},{\"type\":\"infer\",\"name\":\"•2\"},{\"type\":\"infer\",\"name\":\"H\"},{\"type\":\"infer\",\"name\":\"sum_1\"},{\"type\":\"infer\",\"name\":\"n\"},{\"type\":\"Literal\",\"name\":\"2\"},{\"type\":\"infer\",\"name\":\"•3\"},{\"type\":\"infer\",\"name\":\"•4\"},{\"type\":\"infer\",\"name\":\"•5\"},{\"type\":\"infer\",\"name\":\"•6\"},{\"type\":\"Literal\",\"name\":\"1\"},{\"type\":\"infer\",\"name\":\"•7\"},{\"type\":\"infer\",\"name\":\"•8\"}],\"Op1\":[{\"src\":7,\"tgt\":13,\"op1\":\"Grad\"},{\"src\":13,\"tgt\":12,\"op1\":\"Abs\"},{\"src\":7,\"tgt\":16,\"op1\":\"Grad\"},{\"src\":2,\"tgt\":1,\"op1\":\"Div\"},{\"src\":7,\"tgt\":17,\"op1\":\"D(1,t)\"}],\"Op2\":[{\"proj1\":7,\"proj2\":8,\"res\":6,\"op2\":\"^\"},{\"proj1\":5,\"proj2\":6,\"res\":4,\"op2\":\"*\"},{\"proj1\":9,\"proj2\":15,\"res\":14,\"op2\":\"-\"},{\"proj1\":12,\"proj2\":14,\"res\":11,\"op2\":\"^\"},{\"proj1\":4,\"proj2\":11,\"res\":3,\"op2\":\"*\"},{\"proj1\":3,\"proj2\":16,\"res\":2,\"op2\":\"*\"}],\"Σ\":[{\"sum\":8}],\"Summand\":[{\"summand\":9,\"summation\":1},{\"summand\":10,\"summation\":1}]}");
}

#[test]
fn test_serialize_halfar_dome_3_2() {
    let input = "
    <math>
        <mi>Γ</mi>
        <mo>=</mo>
        <mfrac><mn>2</mn><mrow><mi>n</mi><mo>+</mo><mn>2</mn></mrow></mfrac>
        <mi>A</mi>
        <mo>(</mo>
        <mi>ρ</mi>
        <mi>g</mi>
        <msup><mo>)</mo>
        <mi>n</mi></msup>
    </math>
    ";
    let expression = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = expression.to_string();
    let wiring_diagram = to_wiring_diagram(&expression);
    let json = to_decapodes_json(wiring_diagram);
    assert_eq!(json, "{\"Var\":[{\"type\":\"infer\",\"name\":\"mult_1\"},{\"type\":\"infer\",\"name\":\"mult_2\"},{\"type\":\"infer\",\"name\":\"•1\"},{\"type\":\"Literal\",\"name\":\"2\"},{\"type\":\"infer\",\"name\":\"sum_1\"},{\"type\":\"infer\",\"name\":\"n\"},{\"type\":\"infer\",\"name\":\"A\"},{\"type\":\"infer\",\"name\":\"•2\"},{\"type\":\"infer\",\"name\":\"mult_3\"},{\"type\":\"infer\",\"name\":\"ρ\"},{\"type\":\"infer\",\"name\":\"g\"},{\"type\":\"infer\",\"name\":\"Γ\"}],\"Op1\":[],\"Op2\":[{\"proj1\":4,\"proj2\":5,\"res\":3,\"op2\":\"/\"},{\"proj1\":3,\"proj2\":7,\"res\":2,\"op2\":\"*\"},{\"proj1\":10,\"proj2\":11,\"res\":9,\"op2\":\"*\"},{\"proj1\":9,\"proj2\":6,\"res\":8,\"op2\":\"^\"},{\"proj1\":2,\"proj2\":8,\"res\":1,\"op2\":\"*\"}],\"Σ\":[{\"sum\":5}],\"Summand\":[{\"summand\":6,\"summation\":1},{\"summand\":4,\"summation\":1}]}");
}
