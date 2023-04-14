use mathml::petri_net::{
    recognizers::{get_polarity, get_specie_var, is_add_or_subtract_operator, is_var_candidate},
    Polarity, Rate, Specie, Var,
};
pub use mathml::{
    ast::{
        Math,
        MathExpression::{Mfrac, Mi, Mn, Mo, Mrow},
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
    pub src: String,
    pub tgt: String,
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
    let mathml_exp = get_mathml_asts_from_file("../tests/easyexample.txt");
    let mut diagram: WiringDiagram;

    for mathml in mathml_exp.iter() {
        let mml = mathml.content[0].clone();

        println!("{:?}", mml);
        let mut var: Vec<Variable> = Vec::new();
        let mut un_op: Vec<UnaryOperator> = Vec::new();
        //println!(":?", mathml.clone());
        if let Mrow(components) = mml {
            for component in components.iter() {
                println!("components = {}", component);
                match component {
                    Mfrac(num, denom) => {
                        if let (Mrow(num_exp), Mrow(denom_exp)) = (&**num, &**denom) {
                            if let (Mi(var1), Mi(var3)) = (&num_exp[0], &denom_exp[0]) {
                                if var1 == "d" && var3 == "d" {
                                    if let (Mi(var2), Mi(var4)) = (&num_exp[1], &denom_exp[1]) {
                                        let variable = Variable {
                                            r#type: Type::infer,
                                            name: var2.to_owned(),
                                        };
                                        var.push(variable.clone());

                                        println!("var2={:?}", var);
                                        let target = format!("{}{}_{}{}", var1, var2, var3, var4);
                                        let variable2 = Variable {
                                            r#type: Type::infer,
                                            name: target.clone().to_owned(),
                                        };
                                        var.push(variable2.clone());

                                        let diff_op = format!("{}_{}{}", var1, var3, var4);
                                        let operation = UnaryOperator {
                                            src: variable.name,
                                            tgt: target,
                                            op1: diff_op,
                                        };
                                        un_op.push(operation.clone());
                                    }
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
                    Mo(num) => {}
                    _ => {
                        panic!("Unhandled type!");
                    }
                }
            }
        }

        let diagram = WiringDiagram {
            Var: var,
            Op1: un_op,
        };
        println!("diagram = {:#?}", diagram);
        let json_wiringdiagram = serde_json::to_string_pretty(&diagram).unwrap();
        let filepath = "simple_example.json";

        let mut file = File::create(filepath).expect("Cannot create file");
        file.write_all(json_wiringdiagram.as_bytes())
            .expect("Cannotwrite to file");
        println!("json_wiringdiagram = {}", json_wiringdiagram);
    }
    //json_wiringdiagram = serde_json::to_string(&diagram).unwrap()
}
