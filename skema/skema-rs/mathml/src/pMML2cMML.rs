use mathml::ast::Operator::Other;
pub use mathml::{
    ast::{
        Math, MathExpression,
        MathExpression::{GroupTuple, Mfrac, Mi, Mn, Mo, Mover, Mrow, Msub, Msubsup, Msup},
        Operator,
    },
    parsing::{parse, ParseError},
};
use mathml::{
    elem0, etag,
    parsing::{attribute, ws},
    stag, tag_parser,
};

//use nom::IResult;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alphanumeric1, multispace0, not_line_ending, one_of},
    combinator::{complete, map, map_parser, opt, recognize, value},
    multi::many0,
    sequence::{delimited, pair, preceded, separated_pair, tuple},
};

use nom_locate::LocatedSpan;

type Span<'a> = LocatedSpan<&'a str>;
type IResult<'a, O> = nom::IResult<Span<'a>, O, ParseError<'a>>;

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::read_to_string;
//use Clap::Parser;
use mathml::mml2pn::get_mathml_asts_from_file;
use std::collections::HashSet;
use std::fmt;
use std::{
    fs::File,
    io::{self, BufRead, Write},
};

fn parsed_msup(comp1: &Box<MathExpression>, comp2: &Box<MathExpression>) -> String {
    let mut msup_str = String::new();
    msup_str.push_str(&"<apply><power/>".to_string());
    match (&**comp1, &**comp2) {
        (Mi(id), Mn(num)) => msup_str.push_str(&format!("<ci>{}</ci><cn>{}</cn>", id, num)),
        (Mi(id1), Mi(id2)) => msup_str.push_str(&format!("<ci>{}</ci><ci>{}</ci>", id1, id2)),
        (Mn(num1), Mn(num2)) => msup_str.push_str(&format!("<cn>{}</cn><cn>{}</cn>", num1, num2)),
        (Mn(num), Mi(id)) => msup_str.push_str(&format!("<cn>{}</cn><ci>{}</ci>", num, id)),
        _ => {
            panic!("Unhandled Msup")
        }
    }
    msup_str.push_str(&"</apply>".to_string());
    msup_str
}

fn parsed_mfrac(numerator: &Box<MathExpression>, denominator: &Box<MathExpression>) -> String {
    let mut mfrac_str = String::new();

    match (&**numerator, &**denominator) {
        (Mn(num1), Mn(num2)) => mfrac_str.push_str(&format!(
            "<apply><divide/><cn>{}</cn><cn>{}</cn></apply>",
            num1, num2
        )),
        (Mi(id), Mn(num)) => mfrac_str.push_str(&format!(
            "<apply><divide/><ci>{}</ci><cn>{}</cn></apply>",
            id, num
        )),
        (Mrow(num_exp), Mrow(denom_exp)) => {
            if let (Mi(num_id), Mi(denom_id)) = (&num_exp[0], &denom_exp[0]) {
                if num_id == "d" && denom_id == "d" {
                    if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                        mfrac_str.push_str(&format!(
                            "<apply><diff/><bvar><ci>{}</ci></bvar><ci>{}</ci></apply>",
                            id1, id0
                        ))
                    }
                } else if num_id == "∂" && denom_id == "∂" {
                    if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                        mfrac_str.push_str(&format!(
                            "<apply><partialdiff/><bvar><ci>{}</ci></bvar><ci>{}</ci></apply>",
                            id1, id0
                        ))
                    }
                }
            }
            //else if let
        }
        _ => {
            panic!("Unhandled Mfrac")
        }
    }
    mfrac_str
}

fn parsed_msub(comp1: &Box<MathExpression>, comp2: &Box<MathExpression>) -> String {
    let mut msub_str = String::new();
    msub_str.push_str(&"<apply><selector/>".to_string());
    //match (&**comp1, &**comp2)
    match (&**comp1, &**comp2) {
        (Mi(id1), Mi(id2)) => msub_str.push_str(&format!("<ci>{}</ci><ci>{}</ci>", id1, id2)),
        (Mi(id), Mn(num)) => msub_str.push_str(&format!("<ci>{}</ci><cn>{}</cn>", id, num)),
        (Mn(num1), Mn(num2)) => msub_str.push_str(&format!("<cn>{}</cn><cn>{}</cn>", num1, num2)),
        (Mrow(row1), Mrow(row2)) => {
            let mut count_row1_op = 0;
            let has_op_in_row1 = row1.iter().for_each(|j| {
                //let has_op_in_row1 = row1.iter().any(|j| {
                if let Mo(operation) = j {
                    println!("oper={:?}", operation);
                    let oper = format!("{operation}");
                    println!("oper is{}", oper);
                    if oper == "+" {
                        count_row1_op += 1;
                        msub_str.push_str(&"<apply><plus/>".to_string());
                        //true
                    } else if oper == "-" {
                        count_row1_op += 1;
                        msub_str.push_str(&"<apply><minus/>".to_string());
                        //true
                    } else {
                        println!("Unhandled operation.");
                        //false
                    }
                } else {
                    //false
                }
            });
            for r1 in row1.iter() {
                match r1 {
                    Mi(id) => msub_str.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => msub_str.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {}
                    _ => {
                        panic!("Unhandled comp inside Mover")
                    }
                }
            }
            if count_row1_op > 0 {
                for _ in 0..count_row1_op {
                    msub_str.push_str(&"</apply>".to_string());
                }
            } else {
            }
            let mut count_row2_op = 0;
            //for r2 in row2.iter() {
            let has_op_in_row2 = row2.iter().for_each(|j| {
                //let has_op_in_row2 = row2.iter().any(|j| {
                if let Mo(operation) = j {
                    println!("oper={:?}", operation);
                    let oper = format!("{operation}");
                    println!("oper is{}", oper);
                    if oper == "+" {
                        count_row2_op += 1;
                        msub_str.push_str(&"<apply><plus/>".to_string());
                    } else if oper == "-" {
                        count_row2_op += 1;
                        msub_str.push_str(&"<apply><minus/>".to_string());
                    } else {
                        println!("Unhandled operation.");
                    }
                } else {
                }
            });
            for r2 in row2.iter() {
                match r2 {
                    Mi(id) => msub_str.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => msub_str.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {}
                    _ => {
                        panic!("Unhandled comp inside Mover")
                    }
                }
            }
            if count_row2_op > 0 {
                for _ in 0..count_row2_op {
                    msub_str.push_str(&"</apply>".to_string());
                }
            } else {
            }
        }
        _ => {
            panic!("Unhandled Msub")
        }
    }
    msub_str.push_str(&"</apply>".to_string());
    msub_str
}

fn parsed_mrow(row: &Vec<MathExpression>) -> String {
    let mut mrow_str = String::new();
    let mut count_op = 0;
    let mut before_equals: Vec<MathExpression> = Vec::new();
    let mut after_equals: Vec<MathExpression> = Vec::new();
    let mut equals_exists = false;
    for items in row.iter() {
        if let Mo(oper) = items {
            let op = format!("{oper}");
            if op == "=" {
                equals_exists = true;
                continue;
            }
        }
        if equals_exists {
            after_equals.push(items.clone());
        } else {
            before_equals.push(items.clone());
        }
        //println!("before_equals = {:?}", before_equals);
        //println!("after_equals={:?}", after_equals);
    }

    if !before_equals.is_empty() && !after_equals.is_empty() {
        println!("before_equals = {:?}", before_equals);
        println!("after_equals={:?}", after_equals);
        mrow_str.push_str(&"<apply><eq/>".to_string());
        //for before_comp in before_equals.iter() {
        //println!("beofre_comp = {:?}", before_comp);

        let mut count_before_op = 0;
        let has_op_in_before = before_equals.iter().for_each(|j| {
            //let has_op_in_row1 = row1.iter().any(|j| {
            if let Mo(operation) = j {
                println!("oper={:?}", operation);
                let oper = format!("{operation}");
                println!("oper is{}", oper);
                if oper == "+" {
                    count_before_op += 1;
                    mrow_str.push_str(&"<apply><plus/>".to_string());
                } else if oper == "-" {
                    count_before_op += 1;
                    mrow_str.push_str(&"<apply><minus/>".to_string());
                } else {
                    println!("Unhandled operation.");
                }
            } else {
            }
        });
        for before_component in before_equals.iter() {
            match before_component {
                Mi(id) => mrow_str.push_str(&format!("<ci>{}</ci>", id)),
                Mn(num) => mrow_str.push_str(&format!("<cn>{}</cn>", num)),
                Mo(op) => {}
                Msup(sup1, sup2) => {
                    let msup_comp = parsed_msup(sup1, sup2);
                    mrow_str.push_str(&msup_comp.to_string());
                }
                Msub(sub1, sub2) => {
                    let msub_comp = parsed_msub(sub1, sub2);
                    println!("------------------------------");
                    println!("msub_comp={:?}", msub_comp);
                    mrow_str.push_str(&msub_comp.to_string());
                }

                Mfrac(num, denom) => {
                    let mfrac_comp = parsed_mfrac(num, denom);
                    mrow_str.push_str(&mfrac_comp.to_string());
                }
                Mover(over1, over2) => {
                    let mover_comp = parsed_mover(over1, over2);
                    mrow_str.push_str(&mover_comp.to_string());
                }
                Mrow(row) => {
                    let mrow_comp = parsed_mrow(row);
                    mrow_str.push_str(&mrow_comp.to_string());
                }
                _ => {
                    panic!("Unhandled before equals components")
                }
            }
        }
        if count_before_op > 0 {
            for _ in 0..count_before_op {
                mrow_str.push_str(&"</apply>".to_string());
            }
        } else {
        }

        let mut count_after_op = 0;
        let has_op_in_after = after_equals.iter().for_each(|j| {
            //let has_op_in_row1 = row1.iter().any(|j| {
            if let Mo(operation) = j {
                println!("oper={:?}", operation);
                let oper = format!("{operation}");
                println!("oper is{}", oper);
                if oper == "+" {
                    count_after_op += 1;
                    mrow_str.push_str(&"<apply><plus/>".to_string());
                    //true
                } else if oper == "-" {
                    count_after_op += 1;
                    mrow_str.push_str(&"<apply><minus/>".to_string());
                    //true
                } else {
                    println!("Unhandled operation.");
                    //false
                }
            } else {
                //false
            }
        });

        for after_component in after_equals.iter() {
            match after_component {
                Mi(id) => mrow_str.push_str(&format!("<ci>{}</ci>", id)),
                Mn(num) => mrow_str.push_str(&format!("<cn>{}</cn>", num)),
                Mo(op) => {}
                Msup(sup1, sup2) => {
                    let msup_comp = parsed_msup(sup1, sup2);
                    mrow_str.push_str(&msup_comp.to_string());
                }
                Msub(sub1, sub2) => {
                    let msub_comp = parsed_msub(sub1, sub2);
                    println!("------------------------------");
                    println!("msub_comp={:?}", msub_comp);
                    mrow_str.push_str(&msub_comp.to_string());
                }
                Mfrac(num, denom) => {
                    let mfrac_comp = parsed_mfrac(num, denom);
                    mrow_str.push_str(&mfrac_comp.to_string());
                }

                Mover(over1, over2) => {
                    let mover_comp = parsed_mover(over1, over2);
                    mrow_str.push_str(&mover_comp.to_string());
                }

                Mrow(row) => {
                    let mrow_comp = parsed_mrow(row);
                    mrow_str.push_str(&mrow_comp.to_string());
                }
                _ => {
                    panic!("Unhandled after equals components")
                }
            }
            //cmathml.push_str(&"<apply/>".to_string());
        }
        if count_after_op > 0 {
            for _ in 0..count_after_op {
                mrow_str.push_str(&"</apply>".to_string());
            }
        } else {
        }
        mrow_str.push_str(&"<apply/>".to_string());
    }
    else {
        let has_op = row.iter().for_each(|j| {
            //let has_op_in_row1 = row1.iter().any(|j| {
            if let Mo(operation) = j {
                println!("oper={:?}", operation);
                let oper = format!("{operation}");
                println!("oper is{}", oper);
                if oper == "+" {
                    count_op += 1;
                    mrow_str.push_str(&"<apply><plus/>".to_string());
                } else if oper == "-" {
                    count_op += 1;
                    mrow_str.push_str(&"<apply><minus/>".to_string());
                } else {
                    println!("Unhandled operation inside mrow.");
                }
            } else {
            }
        });
        for c in row.iter() {
            match c {
                Mi(id) => mrow_str.push_str(&format!("<ci>{}</ci>", id)),
                Mn(num) => mrow_str.push_str(&format!("<cn>{}</cn>", num)),
                Mo(op) => {}
                Msup(sup1, sup2) => {
                    let msup_comp = parsed_msup(sup1, sup2);
                    mrow_str.push_str(&msup_comp.to_string());
                }
                Msub(sub1, sub2) => {
                    let msub_comp = parsed_msub(sub1, sub2);
                    mrow_str.push_str(&msub_comp.to_string());
                }

                Mfrac(num, denom) => {
                    let mfrac_comp = parsed_mfrac(num, denom);
                    mrow_str.push_str(&mfrac_comp.to_string());
                }
                Mover(over1, over2) => {
                    let mover_comp = parsed_mover(over1, over2);
                    mrow_str.push_str(&mover_comp.to_string());
                }
                _ => {
                    panic!("Unhandled component inside Mrow")
                }
            }
        }
        if count_op > 0 {
            for _ in 0..count_op {
                mrow_str.push_str(&"</apply>".to_string());
            }
        } else {
        }
    }
    mrow_str
}

fn parsed_mover(over1: &Box<MathExpression>, over2: &Box<MathExpression>) -> String {
    let mut mover_str = String::new();

    println!("over1={:?}, over2={:?}", over1, over2);
    if let Mo(over_op) = &**over2 {
        let over_term = format!("{over_op}");
        println!("over_term = {}", over_term);
        if over_term == "‾" {
            mover_str.push_str(&"<apply><conjugate/>".to_string());
            //for comps in &**over1{
            match &**over1 {
                Mi(id) => {
                    mover_str.push_str(&format!("<apply><conjugate/><ci>{}</ci></apply>", id))
                }
                Mrow(comp) => {
                    mover_str.push_str(&"<apply>".to_string());
                    let has_op = comp.iter().any(|j| {
                        if let Mo(operation) = j {
                            println!("oper={:?}", operation);
                            let oper = format!("{operation}");
                            println!("oper is{}", oper);
                            if oper == "+" {
                                mover_str.push_str(&"<plus/>".to_string());
                                true
                            } else if oper == "-" {
                                mover_str.push_str(&"<minus/>".to_string());
                                true
                            } else if oper == "=" {
                                mover_str.push_str(&"<eq/>".to_string());
                                true
                            } else {
                                println!("Unhandled operation.");
                                false
                            }
                        } else {
                            false
                        }
                    });
                    for c in comp.iter() {
                        match c {
                            Mi(id) => mover_str.push_str(&format!("<ci>{}</ci>", id)),
                            Mn(num) => mover_str.push_str(&format!("<cn>{}</cn>", num)),
                            Mo(op) => {}
                            _ => {
                                panic!("Unhandled comp inside Mover")
                            }
                        }
                    }
                    mover_str.push_str(&"</apply>".to_string());
                }

                _ => {
                    panic!("Unhandled comps inside Mover")
                } //}
                  //}
            }
        }
    }
    mover_str.push_str(&"</apply>".to_string());

    mover_str
}

pub fn parsed_pmathml_2_cmathml(pmathml: Vec<Math>) -> String
//fn parsed_pmathml_2_cmathml(pmathml: Vec<Math>)
{
    let mut cmathml = String::new();
    for pmml in pmathml.iter() {
        cmathml.push_str(&"<math>".to_string());
        let mut before_equals: Vec<MathExpression> = Vec::new();
        let mut after_equals: Vec<MathExpression> = Vec::new();
        let mut equals_exists = false;
        for (index, components) in pmml.content.iter().enumerate() {
            //println!("content={:?}", content);
            println!("components={:?}", components);
            let operations: HashSet<&str> = ["+", "-", "="].iter().copied().collect();
            //if let Mrow(components) = content {
            //let has_operation = components.iter().any(|i| {
            //let mut before_equals: Vec<MathExpression> = Vec::new();
            //let mut after_equals: Vec<MathExpression> = Vec::new();
            //let mut equals_exists = false;
            //for items in components.iter() {
            if let Mo(oper) = components {
                let op = format!("{oper}");
                if op == "=" {
                    equals_exists = true;
                    continue;
                }
            }
            if equals_exists {
                after_equals.push(components.clone());
            } else {
                before_equals.push(components.clone());
            }
            //println!("before_equals = {:?}", before_equals);
            //println!("after_equals={:?}", after_equals);
        }
        println!("before_equals = {:?}", before_equals);
        println!("after_equals={:?}", after_equals);

        /////////////////////// INSERT CODE TO iter through before equals and after
        // equals cases
        if !before_equals.is_empty() && !after_equals.is_empty() {
            println!("before_equals = {:?}", before_equals);
            println!("after_equals={:?}", after_equals);
            cmathml.push_str(&"<apply><eq/>".to_string());
            //for before_comp in before_equals.iter() {
            //println!("beofre_comp = {:?}", before_comp);

            let mut count_before_op = 0;
            let has_op_in_before = before_equals.iter().for_each(|j| {
                //let has_op_in_row1 = row1.iter().any(|j| {
                if let Mo(operation) = j {
                    println!("oper={:?}", operation);
                    let oper = format!("{operation}");
                    println!("oper is{}", oper);
                    if oper == "+" {
                        count_before_op += 1;
                        cmathml.push_str(&"<apply><plus/>".to_string());
                    } else if oper == "-" {
                        count_before_op += 1;
                        cmathml.push_str(&"<apply><minus/>".to_string());
                    } else {
                        println!("Unhandled operation.");
                    }
                } else {
                }
            });
            for before_component in before_equals.iter() {
                match before_component {
                    Mi(id) => cmathml.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => cmathml.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {}
                    Msup(sup1, sup2) => {
                        let msup_comp = parsed_msup(sup1, sup2);
                        cmathml.push_str(&msup_comp.to_string());
                    }
                    Msub(sub1, sub2) => {
                        let msub_comp = parsed_msub(sub1, sub2);
                        println!("------------------------------");
                        println!("msub_comp={:?}", msub_comp);
                        cmathml.push_str(&msub_comp.to_string());
                    }

                    Mfrac(num, denom) => {
                        let mfrac_comp = parsed_mfrac(num, denom);
                        cmathml.push_str(&mfrac_comp.to_string());
                    }
                    Mover(over1, over2) => {
                        let mover_comp = parsed_mover(over1, over2);
                        cmathml.push_str(&mover_comp.to_string());
                    }
                    Mrow(row) => {
                        let mrow_comp = parsed_mrow(row);
                        cmathml.push_str(&mrow_comp.to_string());
                    }
                    _ => {
                        panic!("Unhandled before equals components")
                    }
                }
            }
            if count_before_op > 0 {
                for _ in 0..count_before_op {
                    cmathml.push_str(&"</apply>".to_string());
                }
            } else {
            }

            let mut count_after_op = 0;
            let has_op_in_after = after_equals.iter().for_each(|j| {
                //let has_op_in_row1 = row1.iter().any(|j| {
                if let Mo(operation) = j {
                    println!("oper={:?}", operation);
                    let oper = format!("{operation}");
                    println!("oper is{}", oper);
                    if oper == "+" {
                        count_after_op += 1;
                        cmathml.push_str(&"<apply><plus/>".to_string());
                        //true
                    } else if oper == "-" {
                        count_after_op += 1;
                        cmathml.push_str(&"<apply><minus/>".to_string());
                        //true
                    } else {
                        println!("Unhandled operation.");
                        //false
                    }
                } else {
                    //false
                }
            });

            for after_component in after_equals.iter() {
                match after_component {
                    Mi(id) => cmathml.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => cmathml.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {}
                    Msup(sup1, sup2) => {
                        let msup_comp = parsed_msup(sup1, sup2);
                        cmathml.push_str(&msup_comp.to_string());
                    }
                    Msub(sub1, sub2) => {
                        let msub_comp = parsed_msub(sub1, sub2);
                        println!("------------------------------");
                        println!("msub_comp={:?}", msub_comp);
                        cmathml.push_str(&msub_comp.to_string());
                    }
                    Mfrac(num, denom) => {
                        let mfrac_comp = parsed_mfrac(num, denom);
                        cmathml.push_str(&mfrac_comp.to_string());
                    }

                    Mover(over1, over2) => {
                        let mover_comp = parsed_mover(over1, over2);
                        cmathml.push_str(&mover_comp.to_string());
                    }

                    Mrow(row) => {
                        let mrow_comp = parsed_mrow(row);
                        cmathml.push_str(&mrow_comp.to_string());
                    }
                    _ => {
                        panic!("Unhandled after equals components")
                    }
                }
                //cmathml.push_str(&"<apply/>".to_string());
            }
            if count_after_op > 0 {
                for _ in 0..count_after_op {
                    cmathml.push_str(&"</apply>".to_string());
                }
            } else {
            }
            cmathml.push_str(&"<apply/>".to_string());
        }
        //for operator in components.iter() {
        //if before_equals.is_empty() && after_equals.is_empty() {
        else {
            //cmathml.push_str(&"<apply>".to_string());

            let mut count_op_components = 0;
            let has_op_components = pmml.content.iter().for_each(|j| {
                //let has_op_in_row1 = row1.iter().any(|j| {
                if let Mo(operation) = j {
                    println!("oper={:?}", operation);
                    let oper = format!("{operation}");
                    println!("oper is{}", oper);
                    if oper == "+" {
                        count_op_components += 1;
                        cmathml.push_str(&"<apply><plus/>".to_string());
                    } else if oper == "-" {
                        count_op_components += 1;
                        cmathml.push_str(&"<apply><minus/>".to_string());
                    } else {
                        println!("Unhandled operation.");
                    }
                } else {
                }
            });

            for (comp_idx, component) in pmml.content.iter().enumerate() {
                match component {
                    Mi(id) => cmathml.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => cmathml.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {}
                    Msup(sup1, sup2) => {
                        let msup_comp = parsed_msup(sup1, sup2);
                        cmathml.push_str(&msup_comp.to_string());
                    }
                    Msub(sub1, sub2) => {
                        let msub_comp = parsed_msub(sub1, sub2);
                        println!("------------------------------");
                        println!("msub_comp={:?}", msub_comp);
                        cmathml.push_str(&msub_comp.to_string());
                    }
                    Mfrac(num, denom) => {
                        let mfrac_comp = parsed_mfrac(num, denom);
                        cmathml.push_str(&mfrac_comp.to_string());
                    }
                    Mover(over1, over2) => {
                        let mover_comp = parsed_mover(over1, over2);
                        cmathml.push_str(&mover_comp.to_string());
                    }
                    Mrow(row) => {
                        let mrow_comp = parsed_mrow(row);
                        cmathml.push_str(&mrow_comp.to_string());
                    }
                    _ => {
                        panic!("Unhandled after equals components")
                    }
                }
            }

            if count_op_components > 0 {
                for _ in 0..count_op_components {
                    cmathml.push_str(&"</apply>".to_string());
                }
            } else {
            }
            //}
            //}
            /*
            for (comp_idx, component) in components.iter().enumerate() {
                match component {
                    Mi(id) => cmathml.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => cmathml.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {}
                    Msup(sup1, sup2) => {
                        cmathml.push_str(&"<apply><power/>".to_string());
                        match (&**sup1, &**sup2) {
                            (Mi(id), Mn(num)) => {
                                cmathml.push_str(&format!("<ci>{}</ci><cn>{}</cn>", id, num))
                            }
                            (Mi(id1), Mi(id2)) => {
                                cmathml.push_str(&format!("<ci>{}</ci><ci>{}</ci>", id1, id2))
                            }
                            (Mn(num1), Mn(num2)) => {
                                cmathml.push_str(&format!("<cn>{}</cn><cn>{}</cn>", num1, num2))
                            }
                            (Mn(num), Mi(id)) => {
                                cmathml.push_str(&format!("<cn>{}</cn><ci>{}</ci>", num, id))
                            }

                            _ => {
                                panic!("Unhandled Msup")
                            }
                        }
                        cmathml.push_str(&"</apply>".to_string());
                    }
                    Msub(sub1, sub2) => {
                        let msub_comp = parsed_msub(sub1, sub2);
                        println!("------------------------------");
                        println!("msub_comp={:?}", msub_comp);
                        cmathml.push_str(&msub_comp.to_string());
                    }
                    Mover(over1, over2) => {
                        println!("over1={:?}, over2={:?}", over1, over2);
                        if let Mo(over_op) = &**over2 {
                            let over_term = format!("{over_op}");
                            println!("over_term = {}", over_term);
                            if over_term == "‾" {
                                cmathml.push_str(&"<apply><conjugate/>".to_string());
                                //for comps in &**over1{
                                match &**over1 {
                                    Mi(id) => cmathml.push_str(&format!(
                                        "<apply><conjugate/><ci>{}</ci></apply>",
                                        id
                                    )),
                                    Mrow(comp) => {
                                        cmathml.push_str(&"<apply>".to_string());
                                        let has_op = comp.iter().any(|j| {
                                            if let Mo(operation) = j {
                                                println!("oper={:?}", operation);
                                                let oper = format!("{operation}");
                                                println!("oper is{}", oper);
                                                if oper == "+" {
                                                    cmathml.push_str(&"<plus/>".to_string());
                                                    true
                                                } else if oper == "-" {
                                                    cmathml.push_str(&"<minus/>".to_string());
                                                    true
                                                } else if oper == "=" {
                                                    cmathml.push_str(&"<eq/>".to_string());
                                                    true
                                                } else {
                                                    println!("Unhandled operation.");
                                                    false
                                                }
                                            } else {
                                                false
                                            }
                                        });
                                        for c in comp.iter() {
                                            match c {
                                                Mi(id) => cmathml
                                                    .push_str(&format!("<ci>{}</ci>", id)),
                                                Mn(num) => cmathml
                                                    .push_str(&format!("<cn>{}</cn>", num)),
                                                Mo(op) => {}
                                                _ => {
                                                    panic!("Unhandled comp inside Mover")
                                                }
                                            }
                                        }
                                        cmathml.push_str(&"</apply>".to_string());
                                    }

                                    _ => {
                                        panic!("Unhandled comps inside Mover")
                                    } //}
                                      //}
                                }
                            }
                        }
                        cmathml.push_str(&"</apply>".to_string());
                    }
                    Mfrac(numerator, denominator) => match (&**numerator, &**denominator) {
                        (Mn(num1), Mn(num2)) => cmathml.push_str(&format!(
                            "<apply><divide/><cn>{}</cn><cn>{}</cn></apply>",
                            num1, num2
                        )),
                        (Mi(id), Mn(num)) => cmathml.push_str(&format!(
                            "<apply><divide/><ci>{}</ci><cn>{}</cn></apply>",
                            id, num
                        )),
                        (Mrow(num_exp), Mrow(denom_exp)) => {
                            if let (Mi(num_id), Mi(denom_id)) = (&num_exp[0], &denom_exp[0]) {
                                if num_id == "d" && denom_id == "d" {
                                    if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                                        cmathml.push_str(&format!("<apply><diff/><bvar><ci>{}</ci></bvar><ci>{}</ci></apply>", id1, id0))
                                    }
                                } else if num_id == "∂" && denom_id == "∂" {
                                    if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                                        cmathml.push_str(&format!("<apply><partialdiff/><bvar><ci>{}</ci></bvar><ci>{}</ci></apply>", id1, id0))
                                    }
                                }
                            }
                            //else if let
                        }
                        _ => {
                            panic!("Unhandled Mfrac")
                        }
                    },
                    _ => {
                        panic!("Unhandled mathml component")
                    }
                }
            }*/
            //cmathml.push_str(&"</apply>".to_string());
            //}
        }
    }
    cmathml.push_str(&"</math>".to_string());
    cmathml
}

#[test]
fn test_content_mml() {
    let input = "tests/test_c2p_mml/test6.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    vector_mml.push(math);
    let mml = parsed_pmathml_2_cmathml(vector_mml);
    assert_eq!(mml, "<math><apply><minus/><ci>y</ci><apply><conjugate/><apply><plus/><ci>x</ci><cn>1</cn></apply></apply></apply></math>");
}

fn main() {
    let input = "../tests/test_c2p_mml/test4.xml";
    //let input = "../tests/sir.xml";
    //let input = "../tests/seir_eq1.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));

    //let parsed_mml = get_mathml_asts_from_file("../tests/test_c2p_mml/test6.xml");

    println!("math={:?}", math);
    //math.normalize();
    vector_mml.push(math);

    println!("vector_mml={:?}", vector_mml);
    let mml = parsed_pmathml_2_cmathml(vector_mml);
    println!("mml={:?}", mml);
    //assert_eq!(mml, "<math><apply><minus/><ci>y</ci><apply><conjugate/><apply><plus/><ci>x</ci><cn>1</cn></apply></apply></apply></math>");
}

/*
fn main() {
    let mathml_exp = get_mathml_asts_from_file("../tests/test_c2p_mml/test6.xml");
    println!("mathml_exp= {:?}", mathml_exp);
    let cmml = parsed_pmathml_2_cmathml(mathml_exp);
    println!("cmml={:}", cmml);
}
*/
