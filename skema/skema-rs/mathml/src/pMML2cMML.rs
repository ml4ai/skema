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


// Counts how many Mo operators (+, -) there are in a vector
fn counting_operators(x: &Vec<MathExpression>) -> (String, usize) {
    let mut count = 0;
    let mut op_str = String::new();
    x.iter().for_each(|j| {
        if let Mo(operation) = j {
            println!("operation={:?}", operation);
            let op = format!("{operation}");
            println!("operation in symbol is{}", op);
            if op == "+" {
                count += 1;
                op_str.push_str(&"<apply><plus/>".to_string());
            } else if op == "-" {
                count += 1;
                op_str.push_str(&"<apply><minus/>".to_string());
            } else {
                println!("Unhandled operation.");
            }
        } else {
        }
    });
    (op_str, count)
}

// Takes parsed presentation MathML's Msup components and turns it into content MathML representation
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

// Takes parsed  presentation MathML's Mfrac components and turns it into content MathML representation
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
        (Mi(id1), Mi(id2)) => mfrac_str.push_str(&format!(
            "<apply><divide/><ci>{}</ci><ci>{}</ci></apply>",
            id1, id2
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

// Takes parsed  presentation MathML's Msub components and turns it into content MathML representation
fn parsed_msub(comp1: &Box<MathExpression>, comp2: &Box<MathExpression>) -> String {
    let mut msub_str = String::new();
    msub_str.push_str(&"<apply><selector/>".to_string());
    match (&**comp1, &**comp2) {
        (Mi(id1), Mi(id2)) => msub_str.push_str(&format!("<ci>{}</ci><ci>{}</ci>", id1, id2)),
        (Mi(id), Mn(num)) => msub_str.push_str(&format!("<ci>{}</ci><cn>{}</cn>", id, num)),
        (Mn(num1), Mn(num2)) => msub_str.push_str(&format!("<cn>{}</cn><cn>{}</cn>", num1, num2)),
        (Mrow(row1), Mrow(row2)) => {
            /*let mut count_row1_op = 0;
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
            });*/

            let (r1_str, count_row1_op) = counting_operators(&row1);
            msub_str.push_str(&r1_str.to_string());
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
            /*let has_op_in_row2 = row2.iter().for_each(|j| {
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
            });*/

            let (r2_op_str, count_row2_op) = counting_operators(&row2);
            msub_str.push_str(&r2_op_str.to_string());
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

fn parenthesis_group(group: &Vec<MathExpression>) -> String {
    let mut group_str = String::new();
    println!("group={:?}", group);
    let mut count_groups = 0;
    let grouping = group.iter().for_each(|i| {
        if let GroupTuple(expression) = i {
            count_groups += 1;
        }
    });
    println!("count_groups={}", count_groups);

    if count_groups == 0 {
        /*let mut count_op = 0;
        let has_op = group.iter().for_each(|j| {
            //let has_op_in_row1 = row1.iter().any(|j| {
            if let Mo(operation) = j {
                println!("oper={:?}", operation);
                let oper = format!("{operation}");
                println!("oper is{}", oper);
                if oper == "+" {
                    count_op += 1;
                    group_str.push_str(&"<apply><plus/>".to_string());
                } else if oper == "-" {
                    count_op += 1;
                    group_str.push_str(&"<apply><minus/>".to_string());
                } else if oper == "(" || oper == ")" {
                } else {
                    println!("Unhandled operation inside grouping parenthesis.");
                }
            } else {
            }
        });*/

        let (op_str, count_op) = counting_operators(&group);
        group_str.push_str(&op_str.to_string());
        match count_op {
            0 => {
                let comp = vec_mathexp(&group);
                group_str.push_str(&comp.to_string());
            }
            1 => {
                let comp = vec_mathexp(&group);
                group_str.push_str(&comp.to_string());
                group_str.push_str(&"</apply>".to_string());
            }
            2 => {
                let comp = if_two_ops_after_equals((&group).to_vec());
                group_str.push_str(&comp.to_string());
            }
            _ => {
                panic!("Unhandled  operations count in grouping parenthesis")
            }
        }
    } else {
        println!("Unhandles nested grouping with parenthesis");
    }
    println!("group_str = {}", group_str);
    group_str
}

fn vec_mathexp(exp: &Vec<MathExpression>) -> String {
    let mut exp_str = String::new();
    for comp in exp.iter() {
        match comp {
            Mi(id) => exp_str.push_str(&format!("<ci>{}</ci>", id)),
            Mn(num) => exp_str.push_str(&format!("<cn>{}</cn>", num)),
            Mo(op) => {}
            Msup(sup1, sup2) => {
                let msup_comp = parsed_msup(sup1, sup2);
                exp_str.push_str(&msup_comp.to_string());
            }
            Msub(sub1, sub2) => {
                let msub_comp = parsed_msub(sub1, sub2);
                exp_str.push_str(&msub_comp.to_string());
            }

            Mfrac(num, denom) => {
                let mfrac_comp = parsed_mfrac(num, denom);
                exp_str.push_str(&mfrac_comp.to_string());
            }
            Mover(over1, over2) => {
                let mover_comp = parsed_mover(over1, over2);
                exp_str.push_str(&mover_comp.to_string());
            }
            Mrow(row) => {
                let mrow_comp = parsed_mrow(row);
                exp_str.push_str(&mrow_comp.to_string());
            }
            GroupTuple(group) => {
                let group_comp = parenthesis_group(group);
                exp_str.push_str(&group_comp.to_string());
            }
            _ => {
                panic!("Unhandled components in vec_mathexp function")
            }
        }
    }
    println!("exp_str = {}", exp_str);
    exp_str
}

fn if_one_op_after_equals(after_equals: Vec<MathExpression>) -> String {
    let mut mrow_str = String::new();
    let mut before_op: Vec<MathExpression> = Vec::new();
    let mut after_op: Vec<MathExpression> = Vec::new();
    let mut after_op_exists = false;
    let operations: HashSet<&str> = ["+", "-"].iter().copied().collect();

    for comps in after_equals.iter() {
        if let Mo(operation) = &comps {
            let op = format!("{operation}");
            if operations.contains(&op.as_str()) {
                after_op_exists = true;
                continue;
            }
        }
        if after_op_exists {
            after_op.push(comps.clone());
        } else {
            before_op.push(comps.clone());
        }
    }
    println!("before_op = {:?}", before_op);
    println!("before_op.len() = {:?}", before_op.len());
    println!("afterr_op = {:?}", after_op);
    println!("afterr_op.len() = {:?}", after_op.len());
    if !before_op.is_empty() && !after_op.is_empty() {
        match before_op.len() {
            1 => {
                let b_op = vec_mathexp(&before_op);
                mrow_str.push_str(&b_op.to_string());
            }
            2 => {
                for (i, comp) in before_op.iter().enumerate() {
                    if i > 0 {
                        if let (Mi(id1), Mi(id2)) = (&before_op[i - 1], &before_op[i]) {
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci></apply>",
                                id1, id2
                            ));
                        }
                    }
                }
            }
            3 => {
                for (i, comp) in before_op.iter().enumerate() {
                    if i > 1 {
                        if let (Mi(id1), Mi(id2), Mfrac(num, denom)) =
                            (&before_op[i - 2], &before_op[i - 1], &before_op[i])
                        {
                            let mfrac_comp = parsed_mfrac(&num, &denom);
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci>{}</apply>",
                                id1, id2, mfrac_comp
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
        match after_op.len() {
            1 => {
                let a_op = vec_mathexp(&after_op);
                mrow_str.push_str(&a_op.to_string());
            }
            2 => {
                for (i, comp) in after_op.iter().enumerate() {
                    if i > 0 {
                        if let (Mi(id1), Mi(id2)) = (&after_op[i - 1], &after_op[i]) {
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci></apply>",
                                id1, id2
                            ));
                        } else if let (GroupTuple(group), Mi(id)) = (&after_op[i - 1], &after_op[i])
                        {
                            let group_comp = parenthesis_group(&group);
                            mrow_str.push_str(&format!(
                                "<apply><times/>{}<ci>{}</ci></apply>",
                                group_comp, id
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
        mrow_str.push_str(&"</apply>".to_string());
    }
    mrow_str
}

fn if_two_ops_after_equals(after_equals: Vec<MathExpression>) -> String {
    let mut mrow_str = String::new();
    let mut before_op: Vec<MathExpression> = Vec::new();
    let mut after_op: Vec<MathExpression> = Vec::new();
    let mut after_op_exists = false;
    let operations: HashSet<&str> = ["+", "-"].iter().copied().collect();
    for comps in after_equals.iter() {
        if let Mo(operation) = &comps {
            let op = format!("{operation}");
            //if op == "-" {
            if operations.contains(&op.as_str()) {
                if after_op_exists {
                    after_op.push(comps.clone());
                } else {
                    after_op_exists = true;
                }
                //op_count += 1;
                //after_op.push(Vec::new());
                continue;
            }
        }
        if after_op_exists {
            after_op.push(comps.clone());
        } else {
            before_op.push(comps.clone());
        }
    }
    println!("before_op = {:?}", before_op);
    println!("after_op= {:?}", after_op);
    if !before_op.is_empty() && !after_op.is_empty() {
        let b_op = vec_mathexp(&before_op);
        println!("bop={:?}", b_op);
        mrow_str.push_str(&b_op.to_string());

        let mut before_new_op: Vec<MathExpression> = Vec::new();
        let mut after_new_op: Vec<MathExpression> = Vec::new();
        let mut another_op_exists = false;
        for a_comps in after_op.iter() {
            if let Mo(oper) = &a_comps {
                let op = format!("{oper}");
                if operations.contains(&op.as_str()) {
                    if another_op_exists {
                        after_new_op.push(a_comps.clone());
                    } else {
                        another_op_exists = true;
                    }
                    continue;
                }
            }

            if another_op_exists {
                after_new_op.push(a_comps.clone());
            } else {
                before_new_op.push(a_comps.clone());
            }
        }
        println!("before_new_op = {:?}", before_new_op);
        println!("length of before_new_op ={}", before_new_op.len());
        println!("after_new_op= {:?}", after_new_op);
        println!("length of after_new_op ={}", after_new_op.len());

        match before_new_op.len() {
            1 => {
                let new_b_op = vec_mathexp(&before_new_op);
                mrow_str.push_str(&new_b_op.to_string());
            }
            2 => {
                for (i, comp) in before_new_op.iter().enumerate() {
                    if i > 0 {
                        if let (Mi(id1), Mi(id2)) = (&before_new_op[i - 1], &before_new_op[i]) {
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci></apply>",
                                id1, id2
                            ));
                        }
                    }
                }
            }
            _ => {
                panic!("Unhandled length of before_new_op when there are two operations");
            }
        }
        mrow_str.push_str(&"</apply>".to_string());

        match after_new_op.len() {
            1 => {
                let new_a_op = vec_mathexp(&after_new_op);
                mrow_str.push_str(&new_a_op.to_string());
            }
            2 => {
                for (i, comp) in after_new_op.iter().enumerate() {
                    if i > 0 {
                        if let (Mi(id1), Mi(id2)) = (&after_new_op[i - 1], &after_new_op[i]) {
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci></apply>",
                                id1, id2
                            ));
                        } else if let (Mi(id), Mo(Rparenthesis)) =
                            (&after_new_op[i - 1], &after_new_op[i])
                        {
                            mrow_str.push_str(&format!("<ci>{}</ci>", id));
                        }
                    }
                }
            }
            3 => {
                for (i, comp) in after_new_op.iter().enumerate() {
                    if i > 1 {
                        if let (Mi(id1), Mi(id2), Mi(id3)) =
                            (&after_new_op[i - 2], &after_new_op[i - 1], &after_new_op[i])
                        {
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci><ci>{}</ci></apply>",
                                id1, id2, id3
                            ));
                        } else if let (Mi(id1), Mi(id2), Mfrac(num, denom)) =
                            (&after_new_op[i - 2], &after_new_op[i - 1], &after_new_op[i])
                        {
                            let mfrac_comp = parsed_mfrac(&num, &denom);
                            mrow_str.push_str(&format!(
                                "<apply><times/><ci>{}</ci><ci>{}</ci>{}</apply>",
                                id1, id2, mfrac_comp
                            ));
                        }
                    }
                }
            }
            _ => {
                panic!("Unhandled length of after_new_op when there are two operations");
            }
        }

        mrow_str.push_str(&"</apply>".to_string());
    }
    mrow_str
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

        //let mut count_before_op = 0;
        /*let has_op_in_before = before_equals.iter().for_each(|j| {
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
        });*/
        let (b_op_str, count_before_op) = counting_operators(&before_equals);
        mrow_str.push_str(&b_op_str.to_string());
        let before_component = vec_mathexp(&before_equals);
        println!("before_component = {}", before_component);
        mrow_str.push_str(&before_component.to_string());
        /*
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
        }*/
        /*if count_before_op > 0 {
            for _ in 0..count_before_op {
                mrow_str.push_str(&"</apply>".to_string());
            }
        } else {
        }*/

        //let mut after_minus_exists = false;

        /*let mut count_after_op = 0;
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
        });*/

        let (a_op_str, count_after_op) = counting_operators(&after_equals);
        mrow_str.push_str(&a_op_str.to_string());
        println!("count_after_op={}", count_after_op);
        let mut before_minus: Vec<MathExpression> = Vec::new();
        let mut after_minus: Vec<MathExpression> = Vec::new();
        let mut minus_count = 0;
        let operations: HashSet<&str> = ["+", "-"].iter().copied().collect();
        if count_after_op == 1 {
            let one_ops = if_one_op_after_equals(after_equals);
            mrow_str.push_str(&one_ops.to_string());
        } else if count_after_op == 2 {
            let two_ops_after_equals = if_two_ops_after_equals(after_equals);
            mrow_str.push_str(&two_ops_after_equals.to_string());
            /* for comps in after_equals.iter() {
                if let Mo(operation) = &comps {
                    let op = format!("{operation}");
                    //if op == "-" {
                    if operations.contains(&op.as_str()) {
                        if after_minus_exists {
                            after_minus.push(comps.clone());
                        } else {
                            after_minus_exists = true;
                        }
                        //op_count += 1;
                        //after_op.push(Vec::new());
                        continue;
                    }
                }
                if after_minus_exists {
                    after_minus.push(comps.clone());
                } else {
                    before_minus.push(comps.clone());
                }
            }
            println!("before_minus = {:?}", before_minus);
            println!("after_minus= {:?}", after_minus);
            if !before_minus.is_empty() && !after_minus.is_empty() {
                let b_op = vec_mathexp(&before_minus);
                println!("bop={:?}", b_op);
                mrow_str.push_str(&b_op.to_string());

                let mut before_new_op: Vec<MathExpression> = Vec::new();
                let mut after_new_op: Vec<MathExpression> = Vec::new();
                let mut another_op_exists = false;
                for a_comps in after_minus.iter() {
                    if let Mo(oper) = &a_comps {
                        let op = format!("{oper}");
                        if operations.contains(&op.as_str()) {
                            if another_op_exists {
                                after_new_op.push(a_comps.clone());
                            } else {
                                another_op_exists = true;
                            }
                            continue;
                        }
                    }

                    if another_op_exists {
                        after_new_op.push(a_comps.clone());
                    } else {
                        before_new_op.push(a_comps.clone());
                    }
                }
                println!("before_new_op = {:?}", before_new_op);
                println!("length of before_new_op ={}", before_new_op.len());
                println!("after_new_op= {:?}", after_new_op);
                println!("length of after_new_op ={}", after_new_op.len());

                if before_new_op.len() == 1 {
                    let new_b_op = vec_mathexp(&before_new_op);
                    mrow_str.push_str(&new_b_op.to_string());
                } else if before_new_op.len() == 2 {
                    for (i, comp) in before_new_op.iter().enumerate() {
                        if i > 0 {
                            if let (Mi(id1), Mi(id2)) = (&before_new_op[i - 1], &before_new_op[i]) {
                                mrow_str.push_str(&format!(
                                    "<apply><times/><ci>{}</ci><ci>{}</ci></apply>",
                                    id1, id2
                                ));
                            }
                        }
                    }
                }
                mrow_str.push_str(&"</apply>".to_string());

                if after_new_op.len() == 1 {
                    let new_a_op = vec_mathexp(&after_new_op);
                    mrow_str.push_str(&new_a_op.to_string());
                } else if after_new_op.len() == 2 {
                    for (i, comp) in after_new_op.iter().enumerate() {
                        if i > 0 {
                            if let (Mi(id1), Mi(id2)) = (&after_new_op[i - 1], &after_new_op[i]) {
                                mrow_str.push_str(&format!(
                                    "<apply><times/><ci>{}</ci><ci>{}</ci></apply>",
                                    id1, id2
                                ));
                            }
                        }
                    }
                } else if after_new_op.len() == 3 {
                    for (i, comp) in after_new_op.iter().enumerate() {
                        if i > 1 {
                            if let (Mi(id1), Mi(id2), Mi(id3)) =
                                (&after_new_op[i - 2], &after_new_op[i - 1], &after_new_op[i])
                            {
                                mrow_str.push_str(&format!(
                                    "<apply><times/><ci>{}</ci><ci>{}</ci><ci>{}</ci></apply>",
                                    id1, id2, id3
                                ));
                            } else if let (Mi(id1), Mi(id2), Mfrac(num, denom)) =
                                (&after_new_op[i - 2], &after_new_op[i - 1], &after_new_op[i])
                            {
                                let mfrac_comp = parsed_mfrac(&num, &denom);
                                mrow_str.push_str(&format!(
                                    "<apply><times/><ci>{}</ci><ci>{}</ci>{}</apply>",
                                    id1, id2, mfrac_comp
                                ));
                            }
                        }
                    }
                    mrow_str.push_str(&"</apply>".to_string());
                }
            }*/
        }
        //else {
        // println!("before_minus = {:?}", before_minus);
        //println!("after_minus= {:?}", after_minus);
        else {
            let after_component = vec_mathexp(&after_equals);
            mrow_str.push_str(&after_component.to_string());
            /*for after_component in after_equals.iter() {
                match after_component {
                    Mi(id) => mrow_str.push_str(&format!("<ci>{}</ci>", id)),
                    Mn(num) => mrow_str.push_str(&format!("<cn>{}</cn>", num)),
                    Mo(op) => {
                    }
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
            }*/
            /* if count_after_op > 1 {
                //for _ in 0..count_after_op {
                mrow_str.push_str(&"</apply>".to_string());
                //}
            } else {
            }*/
            //mrow_str.push_str(&"</apply>".to_string());
        }
        mrow_str.push_str(&"</apply>".to_string());
    } else {
        /*let has_op = row.iter().for_each(|j| {
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
        });*/

        let (row_str, count_op) = counting_operators(&row);
        mrow_str.push_str(&row_str.to_string());
        let row_comp = vec_mathexp(&row);
        mrow_str.push_str(&row_comp.to_string());
        /*
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
        }*/
        /*if count_op > 0 {
            for _ in 0..count_op {
                mrow_str.push_str(&"</apply>".to_string());
            }
        } else {
        }*/
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
                    /*mover_str.push_str(&"<apply>".to_string());
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
                    });*/

                    let (op_str, count_op) = counting_operators(&comp);
                    mover_str.push_str(&op_str.to_string());
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
                    panic!("Unhandled comps inside Mover when over term is ‾")
                } //}
                  //}
            }
        } else if over_term == "˙" {
            println!("Found over term ˙");
            mover_str.push_str(&"<apply><diff/>");
            match &**over1 {
                Mi(id) => {
                    mover_str.push_str(&format!("<ci>{}</ci>", id));
                }
                _ => {
                    panic!("Unhandled comps inside Mover when over term is ˙")
                }
            }
            mover_str.push_str(&"</apply>");
        } else {
            println!("Unhandled over term in Mover");
        }
    }
    //mover_str.push_str(&"</apply>".to_string());

    mover_str
}

pub fn to_content_mathml(pmathml: Vec<Math>) -> String
//fn parsed_pmathml_2_cmathml(pmathml: Vec<Math>)
{
    let mut cmathml = String::new();
    for pmml in pmathml.iter() {
        cmathml.push_str(&"<math>".to_string());
        let comp = &pmml.content;
        let components = parsed_mrow(&comp);
        println!("components={:?}", components);
        cmathml.push_str(&components.to_string());
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
    let mml = to_content_mathml(vector_mml);
    assert_eq!(mml, "<math><apply><minus/><ci>y</ci><apply><conjugate/><apply><plus/><ci>x</ci><cn>1</cn></apply></apply></apply></math>");
}

fn main() {
    //let input = "../tests/test_c2p_mml/test4.xml";
    //let input = "../tests/sir.xml";
    //let input = "../tests/seir_eq1.xml";
    //let input = "../tests/seir_eq2.xml";
    let input = "../tests/seir_eq3.xml";
    //let input = "../tests/seir_eq4.xml";
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
    let mml = to_content_mathml(vector_mml);
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
