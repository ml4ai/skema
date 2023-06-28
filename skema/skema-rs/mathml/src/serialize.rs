use mathml::ast::Operator::Equals;
//use mathml::ast::Operator::Minus;
use mathml::ast::Operator::Other;
use mathml::petri_net::{
    recognizers::{get_polarity, get_specie_var, is_add_or_subtract_operator, is_var_candidate},
    Polarity, Rate, Specie, Var,
};

//use mathml::parse_decapodes::parse;
pub use mathml::{
    ast::{
        Math, MathExpression,
        MathExpression::{GroupTuple, Mfrac, Mi, Mn, Mo, Mover, Mrow, Msub, Msubsup, Msup},
        Operator,
    },
    //parse_decapodes::parse,
    parsing::parse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::read_to_string;
//use Clap::Parser;
use mathml::mml2pn::get_mathml_asts_from_file;
use std::{
    collections::{HashMap, HashSet},
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

impl Default for Variable {
    fn default() -> Self {
        Variable {
            r#type: Type::infer,
            name: String::new(),
        }
    }
}

impl Default for UnaryOperator {
    fn default() -> Self {
        UnaryOperator {
            src: 0,
            tgt: 0,
            op1: String::new(),
        }
    }
}

//fn diagram(Vec<Math>) -> Vec<WiringDiagram> {
//  let mathml_exp = get_mathml_asts_from_file("../tests/easyexample.xml");

//for mathml in mathml_exp.iter(){
//  println!("\n{:?}", mathml.content[0].clone());
//}

//}
//

pub fn mfrac_derivative(
    numerator: &Box<MathExpression>,
    denominator: &Box<MathExpression>,
    mut var_count: usize,
) -> Option<(Variable, Variable, UnaryOperator, usize)> {
    //let mut numerator_has_derivative = false;
    //let mut denominator_has_derivative = false;
    let mut variable = Variable::default();
    let mut variable2 = Variable::default();
    let mut operation = UnaryOperator::default();
    if let (Mrow(num_exp), Mrow(denom_exp)) = (&**numerator, &**denominator) {
        if let (Mi(num_id), Mi(denom_id)) = (&num_exp[0], &denom_exp[0]) {
            if (num_id == "d" && denom_id == "d") || (num_id == "∂" && denom_id == "∂") {
                //numerator_has_derivative = true;
                //numerator_has_derivative = true;
                if let (Mi(var2), Mi(var4)) = (&num_exp[1], &denom_exp[1]) {
                    var_count += 1;
                    let variable = Variable {
                        r#type: Type::infer,
                        name: var2.to_owned(),
                    };
                    //var.push(variable.clone());

                    //println!("var2={:?}", var);
                    let target = format!("{}{}_{}{}", num_id, var2, denom_id, var4);
                    var_count += 1;
                    let variable2 = Variable {
                        r#type: Type::infer,
                        name: target.clone().to_owned(),
                    };
                    //var.push(variable2.clone());

                    let diff_op = format!("{}_{}{}", num_id, denom_id, var4);
                    let operation = UnaryOperator {
                        src: var_count - 1,
                        tgt: var_count,
                        op1: diff_op,
                    };
                    //un_op.push(operation.clone());
                    return Some((variable, variable2, operation, var_count));
                }
            }
        }
    }
    //(variable, variable2, operation, var_count)
    //else {
    None
    //}
}

pub fn mfrac_case1(
    numerator: &Box<MathExpression>,
    denominator: &Box<MathExpression>,
    mut var_count: usize,
    mut count: usize,
) -> Option<(
    Variable,
    Variable,
    Variable,
    ProjectionOperator,
    Variable,
    Variable,
    Variable,
    ProjectionOperator,
    ProjectionOperator,
    usize,
    usize,
)> {
    if let (Mrow(num_exp), Mrow(denom_exp)) = (&**numerator, &**denominator) {
        if let (Msup(var1, var2), Msub(var3, var4)) = (&num_exp[0], &num_exp[1]) {
            if let (Mover(v1, v2), Mi(v3)) = (&denom_exp[0], &denom_exp[1]) {
                var_count += 1;
                let sup_term = format!("{}^{}", var1, var2);
                let variable1 = Variable {
                    r#type: Type::Constant,
                    name: sup_term.clone().to_owned(),
                };
                //var.push(variable1.clone());

                var_count += 1;
                let sub_term = format!("{}_{}", var3, var4);
                let variable2 = Variable {
                    r#type: Type::Constant,
                    name: sub_term.clone().to_owned(),
                };
                //var.push(variable2.clone());

                count += 1;
                var_count += 1;
                let count_str = format!("•{}", count.to_string());
                let temp_var1 = Variable {
                    r#type: Type::infer,
                    name: count_str,
                };
                //var.push(temp_var1.clone());
                let projection1 = ProjectionOperator {
                    proj1: var_count - 2,
                    proj2: var_count - 1,
                    res: var_count,
                    op2: "*".to_string(),
                };
                //proj_op.push(projection1.clone());

                var_count += 1;
                let over_term = format!("{}_{}", v1, v2);
                let variable3 = Variable {
                    r#type: Type::Constant,
                    name: over_term.clone().to_owned(),
                };
                //var.push(variable3.clone());

                var_count += 1;
                let variable4 = Variable {
                    r#type: Type::Constant,
                    name: v3.to_owned(),
                };
                //var.push(variable4.clone());

                count += 1;
                var_count += 1;
                let count_str2 = format!("•{}", count.to_string());
                let temp_var2 = Variable {
                    r#type: Type::infer,
                    name: count_str2,
                };
                //var.push(temp_var2.clone());

                let projection2 = ProjectionOperator {
                    proj1: var_count - 2,
                    proj2: var_count - 1,
                    res: var_count,
                    op2: "*".to_string(),
                };
                //proj_op.push(projection2.clone());

                let projection3 = ProjectionOperator {
                    proj1: var_count - 3,
                    proj2: var_count,
                    res: var_count + 1,
                    op2: "/".to_string(),
                };
                //proj_op.push(projection3.clone());
                return Some((
                    variable1,
                    variable2,
                    temp_var1,
                    projection1,
                    variable3,
                    variable4,
                    temp_var2,
                    projection2,
                    projection3,
                    var_count,
                    count,
                ));
            }
        }
    }
    None
}

pub fn mover_case1(
    comp1: &Box<MathExpression>,
    comp2: &Box<MathExpression>,
    mut var_count: usize,
) -> Option<(String, Variable, usize)> {
    if let (Mi(exp1_str), Mo(Other(exp2_str))) = (&**comp1, &**comp2) {
        var_count += 1;
        println!(
            "var_count for mover_case1 that takes Mi() and Mo()  ={}",
            var_count
        );
        let target1 = format!("{}{}", exp1_str, exp2_str);
        let variable = Variable {
            r#type: Type::infer,
            name: target1.clone().to_owned(),
        };
        return Some((target1, variable, var_count));
    }
    //println!("variable={:?}", variable.clone());
    //var.push(variable9.clone());
    None
}

pub fn group_parencomp(
    group_comp: Vec<MathExpression>,
    mut var_count: usize,
    mut count: usize,
) -> (String, String, Option<Variable>, usize, usize) {
    //for comp in group_comp.iter() {
    let mut delta_comp = String::new();
    let mut combo_frac = String::new();
    //let mut iden = String::new();
    let mut result_wo_paren = String::new();
    let mut result = String::new();
    let mut duplicate_id: HashSet<String> = HashSet::new();
    let mut duplicate_count: HashMap<String, usize> = HashMap::new();
    let mut variable1 = Variable::default();
    //let mut variable1 = Variable::new();
    //let mut variable1: Vec<Variable> = Vec::new();
    let mut variable2: Vec<Variable> = Vec::new();
    //let mut variable2 = Variable::default();
    //println!("group_comp.len() = {}", group_comp.len());
    //count += 1;
    //for comp in group_comp.iter() {
    for (j, comp) in group_comp.iter().enumerate() {
        if j > 0 {
            if let (Mrow(row_del), Mi(iden)) = (&group_comp[j - 1], &group_comp[j]) {
                if let Mi(delta) = &row_del[0] {
                    if delta == "Δ" {
                        var_count += 1;
                        println!("var_count for Δz  = {}", var_count);
                        let delta_comp = format!("{}{}", delta, iden);
                        let v2 = Variable {
                            r#type: Type::infer,
                            name: delta_comp.clone().to_owned(),
                        };
                        variable2.push(v2.clone());
                        println!("inside if ----> variable2 = {:?}", variable2);
                        //let variable1 = Some(&variable2[0]);
                        //println!("inside if ----> variable1 = {:?}", variable1);

                        count += 1;
                        let count_str = format!("•{}", count.to_string());
                        var_count += 1;
                        println!("var_count for temp_var = {}", var_count);
                        let temp_var1 = Variable {
                            r#type: Type::infer,
                            name: count_str,
                        };
                        println!("temp_var1 = {:?}", temp_var1);

                        let projection2 = ProjectionOperator {
                            proj1: var_count - 2,
                            proj2: var_count - 1,
                            res: var_count,
                            op2: "*".to_string(),
                        };
                        println!("projection2 = {:?}", projection2);

                        count += 1;
                        let count_str2 = format!("•{}", count.to_string());
                        var_count += 1;
                        println!("var_count for temp_var2 = {}", var_count);
                        let temp_var2 = Variable {
                            r#type: Type::infer,
                            name: count_str2,
                        };
                        println!("temp_var2 = {:?}", temp_var2);
                        let summands2 = Summation {
                            summand: var_count,
                            summation: 1,
                        };
                        println!("summands2 ={:?}", summands2);

                        let projection3 = ProjectionOperator {
                            proj1: var_count - 3,
                            proj2: var_count - 2,
                            res: var_count,
                            op2: "*".to_string(),
                        };
                        println!("projection3 = {:?}", projection3);
                    }
                }
            }
        }
        //println!("----> variable1 = {:?}", variable1);
        /*if variable2.is_empty() {
        } else {
            //let variable1 = &variable2[0];
            //println!("----variable1 = {:?}", variable1);
            //println!("----variable2 = {:?}", variable2);
            println!("----variable2[0] = {:?}", variable2[0]);
        }
        */
        //println!("!!!!!!!!!!!!!!!!v1 = {:?}", variable1);
        println!("var_count before match = {}", var_count);
        match comp {
            //match group_comp[j-1]{
            Mo(oper) => {
                let op = format!("{oper}");
                if op == "(".to_string() || op == ")".to_string() {
                    result.push_str(&op.to_string())
                } else {
                    result.push_str(&op.to_string());
                    result_wo_paren.push_str(&op.to_string())
                }
            }
            Mi(op) => {
                var_count += 1;
                println!("var_count inside Mi = {}", var_count);
                result.push_str(&op.to_string());
                result_wo_paren.push_str(&op.to_string())
            }
            // Mfrac(Mn("1"), Mn("2")), Mrow([Mi("Δ")]),
            Mfrac(c1, c2) => {
                var_count += 2;
                if let (Mn(n1), Mn(n2)) = (&**c1, &**c2) {
                    let combo_frac = format!("{}/{}", n1, n2);
                    var_count += 1;
                    println!("var_count for  Mn(1) = {}", var_count);
                    let variable3 = Variable {
                        r#type: Type::Literal,
                        name: n1.clone().to_owned(),
                    };
                    println!("variable3 = {:?}", variable3);
                    var_count += 1;
                    println!("var_count for Mn(2) = {}", var_count);
                    let variable4 = Variable {
                        r#type: Type::Literal,
                        name: n2.clone().to_owned(),
                    };
                    println!("variable4 = {:?}", variable4);
                    var_count += 1;
                    println!("var_count for Mn(1)Mn(2) combo = {}", var_count);
                    result.push_str(&combo_frac.to_string());
                    let variable5 = Variable {
                        r#type: Type::Literal,
                        name: combo_frac.clone().to_owned(),
                    };
                    println!("variable5 = {:?}", variable5);
                    result_wo_paren.push_str(&combo_frac.to_string());

                    let projection1 = ProjectionOperator {
                        proj1: var_count - 2,
                        proj2: var_count - 1,
                        res: var_count,
                        op2: "/".to_string(),
                    };
                    println!("projection1 = {:?}", projection1);
                }
            }
            Mrow(row1) => {
                if let Mi(id) = &row1[0] {
                    result.push_str(&id.to_string());
                    result_wo_paren.push_str(&id.to_string())
                }
            }
            //return Some((result, variable, var_count));
            _ => {}
        }
        //println!("result_wo_paren = {}", result_wo_paren);

        //None
    }
    /*println!("|||||||||||||||||||||||||||||||||||||||||");
    println!("var_count = {}", var_count);
    println!("result_wo_paren = {}", result_wo_paren);
    if result_wo_paren.contains("+") {

    }*/
    /*
        var_count += 1;
        println!("var_count before for temp_var1 = {}", var_count);
        let count_str = format!("•{}", count.to_string());
        let temp_var1 = Variable {
            r#type: Type::infer,
            name: count_str,
        };
        println!("temp_var1 = {:?}", temp_var1);

        //let v4 = var_count-4;
        println!("var_count before projection= {}", var_count);
        let projection1 = ProjectionOperator {
            //proj1: var_count-4,
            proj1: 3,
            proj2: var_count - 1,
            res: var_count,
            op2: "*".to_string(),
        };
        println!("projection1 = {:?}", projection1);

        var_count += 1;
        println!("var_count before temp_var2 = {}", var_count);
        let count_str2 = format!("•{}", (count + 1).to_string());
        let temp_var2 = Variable {
            r#type: Type::infer,
            name: count_str2,
        };
        println!("temp_var1 = {:?}", temp_var2);
        let summands2 = Summation {
            summand: var_count,
            summation: 1,
        };
        println!("summands2 = {:?}", summands2);

        //var_count += 1;
        let projection2 = ProjectionOperator {
            proj1: 3,
            proj2: var_count - 2,
            res: var_count,
            op2: "*".to_string(),
        };
        println!("projection2 = {:?}", projection2);
        println!("var_count before result = {}", var_count);
        let variable = Variable {
            r#type: Type::infer,
            name: result_wo_paren.clone().to_owned(),
        };
        println!("result ={}", result);
        println!("variable ={:?}", variable);
    */
    println!("----------------------------------------------");
    //println!("variable2[0] = {:?}", variable2[0]);
    //return Some((result_wo_paren, result, variable2[0].clone(), var_count, count));

    if variable2.is_empty() {
        (result_wo_paren, result, None, var_count, count)
    } else {
        //let variable1 = &variable2[0];
        //println!("----variable1 = {:?}", variable1);
        //println!("----variable2 = {:?}", variable2);
        //println!("----variable2[0] = {:?}", variable2[0]);
        let vb = variable2[0].clone();
        (
            result_wo_paren,
            result,
            Some(vb),
            var_count,
            count,
        )
    }
    //println!("!!!!!!!!!!!!!!!!v1 = {:?}", variable1);
}
fn main() {
    //let mathml_exp = get_mathml_asts_from_file("../tests/easyexample.txt");
    //let mathml_exp = get_mathml_asts_from_file("../tests/hydrostatic_3_6.xml");
    let mathml_exp = get_mathml_asts_from_file("../tests/model_descrip_3_2.xml");
    //let mathml_exp = get_mathml_asts_from_file("../tests/continuity_eq.xml");
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
        let mut left_brac = String::new();
        let mut right_brac = String::new();
        let mut identifier = String::new();
        let mut other_identifier = Variable::default();
        let mut count = 0;
        let mut var_count = 0;
        let mut exp_count = 0;
        for (index, content) in mathml.content.iter().enumerate() {
            //let mml = mathml.content[0].clone();
            //println!("{:?}", mathml.clone());
            //println!("index={:?}, content={:?}", index, content);
            //let mut count = 0;
            //let mut var_count = 0;
            if let Mrow(components) = content {
                //let mut count = 0;
                //let mut var_count = 0;
                //println!("components.len() = {}", components.len());
                /*for i in 1..components.len() {
                    if let (Mover(over1, over2), GroupTuple(g_comp)) =
                        (&components[i - 1], &components[i])
                    {
                        //if let Some((mover_str, mover_var, var_count)) =mover_case1(over1, over2, var_count)

                        //println!("------ mover_str = {:?}", mover_str);
                        //if let GroupTuple(g_comp) = &components[i] {
                        if let Some((mover_str, mover_var, mut var_count)) =
                            mover_case1(over1, over2, var_count)
                        {
                            //println!("------ mover_str = {:?}", mover_str);
                            if let Some((g_wo_paren, g_paren, mut var_count, count)) =
                                group_parencomp(g_comp.to_vec(), var_count, count)
                            {
                                //println!("------ mi_combo = {:?}", mi_combo);
                                // println!("------ mover_str = {:?}", mover_str);
                                //println!("------ g_wo_paren = {:?}", g_wo_paren);
                                //println!("------ g_paren = {:?}", g_paren);
                                let combine = format!("{}{}", mover_str, g_paren);
                                println!("----combine ={} ", combine);
                                //println!("mover comp1={}, mover comp2 ={}", over1, over2);
                                //println!("g_comp={:?}", g_comp);
                                println!("var_count for z ( g_wo_paren )= {}", var_count);
                                let variable = Variable {
                                    r#type: Type::infer,
                                    name: g_wo_paren.clone().to_owned(),
                                };
                                println!("variable = {:?}", variable);
                                var_count += 1;
                                println!(
                                    "!!!!!!!!!!!!var_count for combine({}) ={}",
                                    combine, var_count
                                );
                                let variable2 = Variable {
                                    r#type: Type::infer,
                                    name: combine.clone().to_owned(),
                                };
                                println!("variable2 = {:?}", variable2);
                                let operation = UnaryOperator {
                                    src: var_count - 1,
                                    tgt: var_count,
                                    op1: mover_str.clone().to_owned(),
                                };
                                println!("!!!!!! operation= {:?}", operation);
                            }
                        }
                        //}
                    }
                }*/
                for (comp_indx, component) in components.iter().enumerate() {
                    //let mut count = 0;
                    //let mut var_count = 0;
                    //println!("comp_indx={:?}, component={:?}", comp_indx, component);
                    //if let Mover(mover_comp1, mover_comp2)
                    if comp_indx > 0 {
                        if let (Mover(over1, over2), GroupTuple(g_comp)) =
                            (&components[comp_indx - 1], &components[comp_indx])
                        {
                            if let Some((mover_str, mover_var, mut var_count)) =
                                mover_case1(over1, over2, var_count)
                            {
                                if let (g_wo_paren, g_paren, Some(vb2), mut var_count, count) =
                                    group_parencomp(g_comp.to_vec(), var_count, count)
                                {
                                    let combine = format!("{}{}", mover_str, g_paren);
                                    println!("----combine ={} ", combine);
                                    println!("var_count for z ( g_wo_paren )= {}", var_count);
                                    let variable = Variable {
                                        r#type: Type::infer,
                                        name: g_wo_paren.clone().to_owned(),
                                    };
                                    println!("variable = {:?}", variable);
                                    let summands1 = Summation {
                                        summand: var_count,
                                        summation: 1,
                                    };
                                    println!("summands1 ={:?}", summands1);
                                    var_count += 1;
                                    println!("!!!!!!!!!!!!var_count before combine ={}", var_count);

                                    let variable2 = Variable {
                                        r#type: Type::infer,
                                        name: combine.clone().to_owned(),
                                    };
                                    println!("variable2 = {:?}", variable2);
                                    let operation = UnaryOperator {
                                        src: var_count - 1,
                                        tgt: var_count,
                                        op1: mover_str.clone().to_owned(),
                                    };
                                    println!("!!!!!! operation= {:?}", operation);
                                    var_count += 1;
                                    let variable10 = Variable {
                                        r#type: Type::infer,
                                        name: "sum_1".to_string(),
                                    };
                                    println!("var_count for variable 10 = {}", var_count);
                                    println!("variable10 = {:?}", variable10);
                                    let projection4 = ProjectionOperator {
                                        proj1: var_count + 3,
                                        proj2: var_count,
                                        res: var_count - 1,
                                        op2: "*".to_string(),
                                    };
                                    println!("projection4 = {:?}", projection4);
                                    let sum2 = Sum { sum: var_count };
                                    println!("sum2 = {:?}", sum2);
                                }
                            }
                        }
                    }
                    println!(">>>>>>>var_count = {}", var_count);
                    match component {
                        Mfrac(num, denom) => {
                            if let Some((v1, v2, operation, var_count)) =
                                mfrac_derivative(num, denom, var_count)
                            {
                                var.push(v1.clone());
                                var.push(v2.clone());
                                un_op.push(operation.clone());
                            }
                            if let Some((
                                var1,
                                var2,
                                var3,
                                p1,
                                var4,
                                var5,
                                var6,
                                p2,
                                p3,
                                var_count,
                                count,
                            )) = mfrac_case1(num, denom, var_count, count)
                            {
                                var.push(var1.clone());
                                var.push(var2.clone());
                                var.push(var3.clone());
                                proj_op.push(p1.clone());
                                var.push(var4.clone());
                                var.push(var5.clone());
                                var.push(var6.clone());
                                proj_op.push(p2.clone());
                                proj_op.push(p3.clone());
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
                            exp_count += 1;
                            //println!("exp_count inside mi={}", exp_count);
                            //println!("comp_indx inside mi={}", comp_indx);
                            let identifier = num;
                            let other_identifier = Variable {
                                r#type: Type::infer,
                                name: num.to_owned(),
                            };
                            //var.push(identifier.clone());
                        }

                        Mo(num) => {
                            match num {
                                Equals => {}
                                Other(oper) => {
                                    let op = format!("{oper}");
                                    if let op = "(" {
                                        exp_count += 1;
                                        //println!("exp_count inside (={}", exp_count);
                                        //println!("comp_indx inside (={}", comp_indx);
                                        let left_brac = op.to_owned();
                                        println!("...inside left brac= {}", left_brac);
                                    }
                                    if let op = ")" {
                                        exp_count += 1;
                                        //println!("exp_count inside )={}", exp_count);
                                        //println!("comp_indx inside )={}", comp_indx);
                                        let right_brac = op.to_owned();
                                        println!("inside right brac= {}", right_brac);
                                    }
                                    //println!("inside left brac= {}", left_brac);
                                    //println!("exp_count={}", exp_count);

                                    if exp_count == 3 && left_brac == "(" && right_brac == ")" {
                                        let exp_3 =
                                            format!("{}{}{}", left_brac, identifier, right_brac);
                                        //println!("exp_3={}", exp_3);
                                        let variable15 = Variable {
                                            r#type: Type::infer,
                                            name: exp_3.clone().to_owned(),
                                        };
                                        // println!("variable15={:?}", variable15);
                                        var.push(variable15.clone());
                                    } else {
                                        var.push(other_identifier.clone());
                                    }
                                }
                                _ => {
                                    panic!("Unhandled inside Mo");
                                }
                            }
                        }

                        Mover(exp1, exp2) => {
                            if let Some((mover_str, mover_var, mut var_count)) =
                                mover_case1(exp1, exp2, var_count)
                            {
                                var.push(mover_var.clone());
                                println!("var_count mover_var = {}", var_count);
                                println!("mover_var = {:?}", mover_var);
                            }
                        } /*match (&**exp1, &**exp2) {
                        (Mi(exp1_str), Mo(Other(exp2_str))) => {
                        var_count += 1;
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
                        },*/
                        GroupTuple(group_comp) => {
                            //for comp in group_comp.iter() {
                            if group_comp.len() == 3 {
                                //println!("g_comp{}", comp);
                                println!(
                                    " inside GroupTuple match:group_comp[0]:{}",
                                    group_comp[0]
                                );
                                if let (Mo(lp), Mi(op), Mo(rp)) =
                                    (&group_comp[0], &group_comp[1], &group_comp[2])
                                {
                                    let group = format!("{}{}{}", lp, op, rp);
                                    println!("group:{}", group);
                                }
                            }
                        }
                        Mrow(row_comp) => {
                            let mut counting = 0;
                            for row_comp in row_comp.iter() {
                                match row_comp {
                                    GroupTuple(group_comp) => {
                                        println!("group tuple inside Mrow inside Mrow");
                                        println!("group_comp.len()={}", group_comp.len());
                                        //for i in 1..group_comp.len() {
                                        for (i, grouping) in group_comp.iter().enumerate() {
                                            if i > 0 {
                                                if let (Mover(over1, over2), GroupTuple(g_comp)) =
                                                    (&group_comp[i - 1], &group_comp[i])
                                                {
                                                    if let Some((
                                                        mover_str,
                                                        mover_var,
                                                        mut var_count,
                                                    )) = mover_case1(over1, over2, var_count)
                                                    {
                                                        if let (
                                                            g_wo_paren,
                                                            g_paren,
                                                            Some(vb2),
                                                            mut var_count,
                                                            count,
                                                        ) = group_parencomp(
                                                            g_comp.to_vec(),
                                                            var_count,
                                                            count,
                                                        ) {
                                                            println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                                                            println!(
                                                                "var_count before counting= {}",
                                                                var_count
                                                            );
                                                            println!("vb2 = {:?}", vb2);
                                                            counting += 1;
                                                            var_count -= 1;
                                                            var_count += counting;
                                                            println!("counting = {}", counting);
                                                            let combine =
                                                                format!("{}{}", mover_str, g_paren);
                                                            println!("combine= {}", combine);

                                                            if g_paren.contains("+") {
                                                                let sum1 = Sum { sum: var_count };
                                                                println!("sum1 = {:?}", sum1);

                                                                let variable6 = Variable {
                                                                    r#type: Type::infer,
                                                                    name: g_wo_paren
                                                                        .clone()
                                                                        .to_owned(),
                                                                };
                                                                println!(
                                                                    "variable6 = {:?}",
                                                                    variable6
                                                                );
                                                                var_count += 1;
                                                                println!(
                                                                    "var_count for variable7 = {}",
                                                                    var_count
                                                                );
                                                                let variable7 = Variable {
                                                                    r#type: Type::infer,
                                                                    name: combine
                                                                        .clone()
                                                                        .to_owned(),
                                                                };
                                                                println!(
                                                                    "variable7 = {:?}",
                                                                    variable7
                                                                );
                                                                let operation = UnaryOperator {
                                                                    src: var_count - 1,
                                                                    tgt: var_count,
                                                                    op1: mover_str
                                                                        .clone()
                                                                        .to_owned(),
                                                                };
                                                                println!(
                                                                    "....... operation = {:?}",
                                                                    operation
                                                                );
                                                                let summands3 = Summation {
                                                                    summand: var_count,
                                                                    summation: 2,
                                                                };
                                                                println!(
                                                                    "summands3 = {:?}",
                                                                    summands3
                                                                );
                                                            } else if g_paren.contains("-") {
                                                                var_count += 1;
                                                                println!(
                                                                    "var_count for variable8 = {}",
                                                                    var_count
                                                                );
                                                                let variable8 = Variable {
                                                                    r#type: Type::infer,
                                                                    name: g_wo_paren
                                                                        .clone()
                                                                        .to_owned(),
                                                                };
                                                                println!(
                                                                    "variable8 = {:?}",
                                                                    variable8
                                                                );
                                                                let projection3 =
                                                                    ProjectionOperator {
                                                                        proj1: 2,
                                                                        proj2: var_count - 3,
                                                                        res: var_count,
                                                                        op2: "-".to_string(),
                                                                    };
                                                                println!(
                                                                    "projection3 = {:?}",
                                                                    projection3
                                                                );
                                                                var_count += 1;
                                                                println!(
                                                                    "var_count for variable9 ={}",
                                                                    var_count
                                                                );
                                                                let variable9 = Variable {
                                                                    r#type: Type::infer,
                                                                    name: combine
                                                                        .clone()
                                                                        .to_owned(),
                                                                };
                                                                println!(
                                                                    "variable9 = {:?}",
                                                                    variable9
                                                                );
                                                                let summands4 = Summation {
                                                                    summand: var_count,
                                                                    summation: 2,
                                                                };
                                                                println!(
                                                                    "summands4 = {:?}",
                                                                    summands4
                                                                );
                                                            }

                                                            /*let summands1 = Summation {
                                                                    summand: 2,
                                                                    summation: 1,
                                                                };
                                                                println!("summands1 ={:?}", summands1);
                                                                var_count += counting;
                                                                println!(".............................");
                                                                println!("....... counting = {}", counting);
                                                                println!(
                                                                    "....... var_count = {}",
                                                                    var_count
                                                                );
                                                                println!("....... g_paren = {}", g_paren);
                                                                if g_paren.contains("+") {
                                                                    println!(
                                                                        "g_paren.contains('+') = {}",
                                                                        g_paren.contains("+")
                                                                    );
                                                                    let sum1 = Sum { sum: var_count };
                                                                    let variable3 = Variable {
                                                                        r#type: Type::infer,
                                                                        name: g_wo_paren.clone().to_owned(),
                                                                    };
                                                                    println!("variable3 = {:?}", variable3);
                                                                    println!(
                                                                        "+++++++++++++++ sum1 = {:?}",
                                                                        sum1
                                                                    );

                                                                    var_count += 1;
                                                                    let operation = UnaryOperator {
                                                                        src: var_count - 1,
                                                                        tgt: var_count,
                                                                        op1: mover_str.clone().to_owned(),
                                                                    };
                                                                    println!(
                                                                        "var_count afteroperation={}",
                                                                        var_count
                                                                    );
                                                                    println!(
                                                                        "....... operation = {:?}",
                                                                        operation
                                                                    );
                                                                    let variable4 = Variable {
                                                                        r#type: Type::infer,
                                                                        name: combine.clone().to_owned(),
                                                                    };
                                                                    println!("variable4 = {:?}", variable4);

                                                                    let summands4 = Summation {
                                                                        summand: var_count,
                                                                        summation: 2,
                                                                    };
                                                                    println!("summands4 = {:?}", summands4);
                                                                } else if g_paren.contains("-") {
                                                                    println!(
                                                                        "g_paren.contains('-') = {}",
                                                                        g_paren.contains("-")
                                                                    );
                                                                    var_count += 1;
                                                                    let variable5 = Variable {
                                                                        r#type: Type::infer,
                                                                        name: g_wo_paren.clone().to_owned(),
                                                                    };
                                                                    println!("variable5 = {:?}", variable5);
                                                                    let projection3 = ProjectionOperator {
                                                                        proj1: 2,
                                                                        proj2: var_count - 3,
                                                                        res: var_count,
                                                                        op2: "-".to_string(),
                                                                    };
                                                                    println!(
                                                                        "projection3 = {:?}",
                                                                        projection3
                                                                    );

                                                                    let summands3 = Summation {
                                                                        summand: var_count,
                                                                        summation: 2,
                                                                    };
                                                                    println!("summands3 = {:?}", summands3);

                                                                    var_count += 1;
                                                                    let operation2 = UnaryOperator {
                                                                        src: var_count,
                                                                        tgt: var_count + 1,
                                                                        op1: mover_str.clone().to_owned(),
                                                                    };
                                                                    println!(
                                                                        "var_count after operation2={}",
                                                                        var_count
                                                                    );
                                                                    println!(
                                                                        "operation2 ={:?}",
                                                                        operation2
                                                                    );

                                                                    let variable5 = Variable {
                                                                        r#type: Type::infer,
                                                                        name: combine.clone().to_owned(),
                                                                    };
                                                                    println!("variable5 = {:?}", variable5);
                                                                }

                                                                var_count += 1;
                                                                let variable6 = Variable {
                                                                    r#type: Type::infer,
                                                                    name: "sum_1".to_string(),
                                                                };
                                                                println!(
                                                                    "var_count for variable 6 = {}",
                                                                    var_count
                                                                );
                                                                println!("variable6 = {:?}", variable6);

                                                                let projection4 = ProjectionOperator {
                                                                    proj1: 6,
                                                                    proj2: var_count,
                                                                    res: 4,
                                                                    op2: "*".to_string(),
                                                                };
                                                                println!("projection2 = {:?}", projection4);
                                                            */
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        panic!("Unhandled comp in Mrow[Mrow [comp]");
                                    }
                                }
                            }
                        }

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

                                                    let summands = Summation {
                                                        summand: var_count,
                                                        summation: 1,
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

                                let summing = Sum { sum: var_count };
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
        //let json_wiringdiagram = serde_json::to_string_pretty(&diagram).unwrap();
        /*let filepath = "../tests/model_descrip_3_1.json";

            let mut file = File::create(filepath).expect("Cannot create file");
                    file.write_all(json_wiringdiagram.as_bytes())
                        .expect("Cannotwrite to file");

            println!("json_wiringdiagram = {}", json_wiringdiagram);
        */
    }
}
