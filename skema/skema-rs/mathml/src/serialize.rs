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
    pub tgt: usize,
    pub proj: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WiringDiagram {
    pub Var: Vec<Variable>,
    pub Op1: Vec<UnaryOperator>,
}

//fn diagram(Vec<Math>) -> Vec<WiringDiagram> {
//  let mathml_exp = get_mathml_asts_from_file("../tests/easyexample.xml");

//for mathml in mathml_exp.iter(){
//  println!("\n{:?}", mathml.content[0].clone());
//}

//}
//

fn main() {
    let mathml_exp = get_mathml_asts_from_file("../tests/hydrostatic_3_6.xml");
    let mut diagram: WiringDiagram;
    //let mut var: Vec<Variable> = Vec::new();
    //let mut un_op: Vec<UnaryOperator> = Vec::new();
    for mathml in mathml_exp.iter() {
        let mut var: Vec<Variable> = Vec::new();
        let mut un_op: Vec<UnaryOperator> = Vec::new();
        for content in mathml.content.iter() {
            //let mml = mathml.content[0].clone();
            //println!(":?", mathml.clone());
            println!("{:?}", content);
            if let Mrow(components) = content {
                for component in components.iter() {
                    match component {
                        Mfrac(num, denom) => {
                            if let (Mrow(num_exp), Mrow(denom_exp)) = (&**num, &**denom) {
                                if let (Mi(var1), Mi(var3)) = (&num_exp[0], &denom_exp[0]) {
                                    if (var1 == "d" && var3 == "d") || (var1 == "∂" && var3 == "∂")
                                    {
                                        if let (Mi(var2), Mi(var4)) = (&num_exp[1], &denom_exp[1]) {
                                            let variable = Variable {
                                                r#type: Type::infer,
                                                name: var2.to_owned(),
                                            };
                                            var.push(variable.clone());

                                            println!("var2={:?}", var);
                                            let target =
                                                format!("{}{}_{}{}", var1, var2, var3, var4);
                                            let variable2 = Variable {
                                                r#type: Type::infer,
                                                name: target.clone().to_owned(),
                                            };
                                            var.push(variable2.clone());

                                            let diff_op = format!("{}_{}{}", var1, var3, var4);
                                            let operation = UnaryOperator {
                                                src: 1,
                                                tgt: 2,
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
                                        let sup_term = format!("{}^{}", var1, var2);
                                        let variable4 = Variable {
                                            r#type: Type::Constant,
                                            name: sup_term.clone().to_owned(),
                                        };
                                        var.push(variable4.clone());

                                        let sub_term = format!("{}_{}", var3, var4);
                                        let variable5 = Variable {
                                            r#type: Type::Constant,
                                            name: sub_term.clone().to_owned(),
                                        };
                                        var.push(variable5.clone());

                                        let over_term = format!("{}_{}", v1, v2);
                                        let variable6 = Variable {
                                            r#type: Type::Constant,
                                            name: over_term.clone().to_owned(),
                                        };
                                        var.push(variable6.clone());

                                        let variable7 = Variable {
                                            r#type: Type::Constant,
                                            name: v3.to_owned(),
                                        };
                                        var.push(variable7.clone());
                                    }
                                }
                            }
                        }

                        Mn(num) => {
                            let variable3 = Variable {
                                r#type: Type::Literal,
                                name: num.to_owned(),
                            };
                            var.push(variable3.clone());
                            println!("var={:?}", var);
                        }

                        Mi(num) => {
                            let variable8 = Variable {
                                r#type: Type::infer,
                                name: num.to_owned(),
                            };
                            var.push(variable8.clone());
                        }

                        Mo(num) => {}
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
        };
        //println!("diagram = {:#?}", diagram);
        let json_wiringdiagram = serde_json::to_string_pretty(&diagram).unwrap();
        let filepath = "../tests/hydrostatic_3_6.json";

        let mut file = File::create(filepath).expect("Cannot create file");
        file.write_all(json_wiringdiagram.as_bytes())
            .expect("Cannotwrite to file");

        println!("json_wiringdiagram = {}", json_wiringdiagram);
    }
}
