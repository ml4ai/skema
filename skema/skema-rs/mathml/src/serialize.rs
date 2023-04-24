use mathml::ast::Operator::Equals;
//use mathml::ast::Operator::Minus;
use mathml::ast::Operator::Other;
use mathml::petri_net::{
    recognizers::{get_polarity, get_specie_var, is_add_or_subtract_operator, is_var_candidate},
    Polarity, Rate, Specie, Var,
};
pub use mathml::{
    ast::{
        Math,
        MathExpression::{Mfrac, Mi, Mn, Mo, Mover, Mrow, Msub, Msubsup, Msup},
        Operator,
    },
    parsing::parse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::read_to_string;
//use Clap::Parser;
use mathml::mml2pn::get_mathml_asts_from_file;
use std::{
    fs::File,
    io::{self, BufRead, Write},
};

use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Mo_other {
    Other(String),
}

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

//fn diagram(Vec<Math>) -> Vec<WiringDiagram> {
//  let mathml_exp = get_mathml_asts_from_file("../tests/easyexample.xml");

//for mathml in mathml_exp.iter(){
//  println!("\n{:?}", mathml.content[0].clone());
//}

//}
//

fn main() {
    //let mathml_exp = get_mathml_asts_from_file("../tests/hydrostatic_3_6.xml");
    let mathml_exp = get_mathml_asts_from_file("../tests/model_descrip_3_1.xml");
    //let mathml_exp = get_mathml_asts_from_file("../tests/molecular_viscosity.xml");
    let mut diagram: WiringDiagram;
    //let mut var: Vec<Variable> = Vec::new();
    //let mut un_op: Vec<UnaryOperator> = Vec::new();
    for mathml in mathml_exp.iter() {
        let mut var: Vec<Variable> = Vec::new();
        let mut un_op: Vec<UnaryOperator> = Vec::new();
        let mut proj_op: Vec<ProjectionOperator> = Vec::new();
        let mut sum_op: Vec<Sum> = Vec::new();
        let mut summand_op: Vec<Summation> = Vec::new();
        //for content in mathml.content.iter() {
        let mut count = 0;
        let mut var_count = 0;
        for (index, content) in mathml.content.iter().enumerate() {
            //let mml = mathml.content[0].clone();
            //println!(":?", mathml.clone());
            println!("index={:?}, content={:?}", index, content);
            //let mut count = 0;
            //let mut var_count = 0;
            if let Mrow(components) = content {
                //let mut count = 0;
                //let mut var_count = 0;
                for component in components.iter() {
                    //let mut count = 0;
                    //let mut var_count = 0;
                    match component {
                        Mfrac(num, denom) => {
                            if let (Mrow(num_exp), Mrow(denom_exp)) = (&**num, &**denom) {
                                if let (Mi(var1), Mi(var3)) = (&num_exp[0], &denom_exp[0]) {
                                    if (var1 == "d" && var3 == "d") || (var1 == "∂" && var3 == "∂")
                                    {
                                        if let (Mi(var2), Mi(var4)) = (&num_exp[1], &denom_exp[1]) {
                                            var_count += 1;
                                            let variable = Variable {
                                                r#type: Type::infer,
                                                name: var2.to_owned(),
                                            };
                                            var.push(variable.clone());

                                            println!("var2={:?}", var);
                                            let target =
                                                format!("{}{}_{}{}", var1, var2, var3, var4);
                                            var_count += 1;
                                            let variable2 = Variable {
                                                r#type: Type::infer,
                                                name: target.clone().to_owned(),
                                            };
                                            var.push(variable2.clone());

                                            let diff_op = format!("{}_{}{}", var1, var3, var4);
                                            let operation = UnaryOperator {
                                                src: var_count - 1,
                                                tgt: var_count,
                                                op1: diff_op,
                                            };
                                            un_op.push(operation.clone());
                                        }
                                    }
                                }
                                //println!("num_exp={:?}",num_exp);
                                //if let Msup(var1, var2) = &num_exp.iter()
                                if let (Msup(var1, var2), Msub(var3, var4)) =
                                    (&num_exp[0], &num_exp[1])
                                {
                                    if let (Mover(v1, v2), Mi(v3)) = (&denom_exp[0], &denom_exp[1])
                                    {
                                        var_count += 1;
                                        let sup_term = format!("{}^{}", var1, var2);
                                        let variable4 = Variable {
                                            r#type: Type::Constant,
                                            name: sup_term.clone().to_owned(),
                                        };
                                        var.push(variable4.clone());

                                        var_count += 1;
                                        let sub_term = format!("{}_{}", var3, var4);
                                        let variable5 = Variable {
                                            r#type: Type::Constant,
                                            name: sub_term.clone().to_owned(),
                                        };
                                        var.push(variable5.clone());

                                        count += 1;
                                        var_count += 1;
                                        let count_str = format!("•{}", count.to_string());
                                        let temp_var1 = Variable {
                                            r#type: Type::infer,
                                            name: count_str,
                                        };
                                        var.push(temp_var1.clone());
                                        let projection1 = ProjectionOperator {
                                            proj1: var_count - 2,
                                            proj2: var_count - 1,
                                            res: var_count,
                                            op2: "*".to_string(),
                                        };
                                        proj_op.push(projection1.clone());

                                        var_count += 1;
                                        let over_term = format!("{}_{}", v1, v2);
                                        let variable6 = Variable {
                                            r#type: Type::Constant,
                                            name: over_term.clone().to_owned(),
                                        };
                                        var.push(variable6.clone());

                                        var_count += 1;
                                        let variable7 = Variable {
                                            r#type: Type::Constant,
                                            name: v3.to_owned(),
                                        };
                                        var.push(variable7.clone());

                                        count += 1;
                                        var_count += 1;
                                        let count_str2 = format!("•{}", count.to_string());
                                        let temp_var2 = Variable {
                                            r#type: Type::infer,
                                            name: count_str2,
                                        };
                                        var.push(temp_var2.clone());

                                        let projection2 = ProjectionOperator {
                                            proj1: var_count - 2,
                                            proj2: var_count - 1,
                                            res: var_count,
                                            op2: "*".to_string(),
                                        };
                                        proj_op.push(projection2.clone());

                                        let projection3 = ProjectionOperator {
                                            proj1: var_count - 3,
                                            proj2: var_count,
                                            res: var_count + 1,
                                            op2: "/".to_string(),
                                        };
                                        proj_op.push(projection3.clone());
                                    }
                                }
                            }
                        }

                        Mn(num) => {
                            var_count += 1;
                            let variable3 = Variable {
                                r#type: Type::Literal,
                                name: num.to_owned(),
                            };
                            var.push(variable3.clone());
                        }

                        Mi(num) => {
                            var_count += 1;
                            let variable8 = Variable {
                                r#type: Type::infer,
                                name: num.to_owned(),
                            };
                            var.push(variable8.clone());
                        }

                        Mo(num) => {}

                        Mover(exp1, exp2) => match (&**exp1, &**exp2) {
                            (Mi(exp1_str), Mo(Other(exp2_str))) => {
                                var_count += 1;
                                println!("varcount ={}", var_count);
                                let target1 = format!("{}{}", exp1_str, exp2_str);
                                let variable9 = Variable {
                                    r#type: Type::infer,
                                    name: target1.clone().to_owned(),
                                };
                                println!("variable9={:?}", variable9.clone());
                                var.push(variable9.clone());
                            }
                            _ => {
                                panic!("Unhandled inside Mover");
                            }
                        },

                        Msup(exp1, exp2) => match (&**exp1, &**exp2) {
                            (Mrow(row_comp1), Mrow(row_comp2)) => {
                                //println!("row_comp1.len()= {}", row_comp1.len());
                                //let mut count=0;
                                for row_comps1 in row_comp1.iter() {
                                    //println!("index={:?}, content={:?}", index, content);

                                    match row_comps1 {
                                        Mo(v1) => {
                                            if let v1 = Equals {
                                            } else {
                                                println!(" Mo operator is {}", v1)
                                            }
                                        }
                                        Mfrac(num, denom) => {
                                            if let (Msub(sub1, sub2), Msub(sub3, sub4)) =
                                                (&**num, &**denom)
                                            {
                                                if let Mrow(row2) = &**sub1 {
                                                    var_count += 1;
                                                    println!("varcount 2 ={}", var_count);
                                                    let sub_term3 =
                                                        format!("{}_{}", row2[0], &**sub2);
                                                    let variable14 = Variable {
                                                        r#type: Type::infer,
                                                        name: sub_term3.clone().to_owned(),
                                                    };
                                                    var.push(variable14.clone());
                                                    var_count += 1;
                                                    let sub_term4 =
                                                        format!("{}_{}", &**sub3, &**sub4);
                                                    let variable15 = Variable {
                                                        r#type: Type::infer,
                                                        name: sub_term4.clone().to_owned(),
                                                    };
                                                    var.push(variable15.clone());

                                                    count += 1;
                                                    var_count += 1;
                                                    let count_str =
                                                        format!("•{}", count.to_string());
                                                    let temp_var1 = Variable {
                                                        r#type: Type::infer,
                                                        name: count_str,
                                                    };
                                                    var.push(temp_var1.clone());

                                                    let projection1 = ProjectionOperator {
                                                        proj1: var_count - 2,
                                                        proj2: var_count - 1,
                                                        res: var_count,
                                                        op2: "/".to_string(),
                                                    };
                                                    proj_op.push(projection1.clone());

                                                    let summands =  Summation{
                                                        summand: var_count,
                                                        summation:1,
                                                    };
                                                    summand_op.push(summands.clone());

                                                }
                                            }
                                        }

                                        _ => {
                                            panic!("Unhandled inside Msup inside Mrow");
                                        }
                                    }
                                }
                                //for row_comps2 in row_comp1.iter(){
                                if let (Mo(v1), Mn(v2)) = (&row_comp2[0], &row_comp2[1]) {
                                    var_count += 1;
                                    if let v1 = "Minus".to_string() {
                                        let v11 = '-'.to_string();
                                        let targ = format!("{}{}", v11, v2);
                                        let variable11 = Variable {
                                            r#type: Type::Literal,
                                            name: targ.clone().to_owned(),
                                        };
                                        var.push(variable11.clone());
                                    }
                                }
                                var_count += 1;
                                let variable12 = Variable {
                                    r#type: Type::infer,
                                    name: "sum_1".to_string(),
                                };
                                var.push(variable12.clone());

                                let summing = Sum {
                                    sum: var_count,
                                };
                                sum_op.push(summing.clone());

                                let projection2 = ProjectionOperator {
                                    proj1: var_count,
                                    proj2: var_count - 1,
                                    res: 1,
                                    op2: "^".to_string(),
                                };
                                proj_op.push(projection2.clone());
                            }

                            _ => {
                                panic!("Unhandled inside Msup");
                            }
                        },

                        _ => {
                            panic!("Unhandled type!");
                        }
                    }
                }
            }
        }
        //json_wiringdiagram = serde_json::to_string(&diagram).unwrap()

        let diagram = WiringDiagram {
            Var: var,
            Op1: un_op,
            Op2: proj_op,
            Σ: sum_op,
            Summand: summand_op,
        };
        //println!("diagram = {:#?}", diagram);
        let json_wiringdiagram = serde_json::to_string_pretty(&diagram).unwrap();
        let filepath = "../tests/model_descrip_3_1.json";

        let mut file = File::create(filepath).expect("Cannot create file");
                file.write_all(json_wiringdiagram.as_bytes())
                    .expect("Cannotwrite to file");
        
        println!("json_wiringdiagram = {}", json_wiringdiagram);
    }
}
