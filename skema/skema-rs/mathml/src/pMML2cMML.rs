use crate::ast::Operator::Other;
use crate::{
    ast::{
        Math, MathExpression,
        MathExpression::{GroupTuple, Mfrac, Mi, Mn, Mo, Mover, Mrow, Msub, Msubsup, Msup},
        Operator,
    },
    parsing::{parse, ParseError},
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alphanumeric1, multispace0, not_line_ending, one_of},
    combinator::{complete, map, map_parser, opt, recognize, value},
    multi::many0,
    sequence::{delimited, pair, preceded, separated_pair, tuple},
};

use nom_locate::LocatedSpan;

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fmt;
use std::fs::read_to_string;
use std::{
    fs::File,
    io::{self, BufRead, Write},
};

/// Counts how many Mo operators (+, -) there are in a vector
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

/// Translate presentation mathml Mi to content mathml Ci
fn mi2ci(x: String) -> String {
    return format!("<ci>{}</ci>", x);
}

/// Translate presentation mathml Mn to content mathml Cn
fn mn2cn(x: String) -> String {
    return format!("<cn>{}</cn>", x);
}

/// Handles matching of presentation MathML components to content MathML by calling other functions
fn vec_mathexp(exp: &Vec<MathExpression>) -> String {
    let mut exp_str = String::new();
    for comp in exp.iter() {
        match comp {
            Mi(id) => exp_str.push_str(&mi2ci(id.to_string())),
            Mn(num) => exp_str.push_str(&mn2cn(num.to_string())),
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

/// Takes parsed presentation MathML's Msup components and turns it into content MathML representation
fn parsed_msup(comp1: &Box<MathExpression>, comp2: &Box<MathExpression>) -> String {
    let mut msup_str = String::new();
    msup_str.push_str(&"<apply><power/>".to_string());
    match (&**comp1, &**comp2) {
        (Mi(id), Mn(num)) => {
            let ci_str = mi2ci(id.to_string());
            let cn_str = mn2cn(num.to_string());
            msup_str.push_str(&format!("{}{}", ci_str, cn_str));
        }
        (Mi(id1), Mi(id2)) => {
            let ci_str = mi2ci(id1.to_string());
            let ci2_str = mi2ci(id2.to_string());
            msup_str.push_str(&format!("{}{}", ci_str, ci2_str));
        }
        (Mn(num1), Mn(num2)) => {
            let cn_str = mn2cn(num1.to_string());
            let cn2_str = mn2cn(num2.to_string());
            msup_str.push_str(&format!("{}{}", cn_str, cn2_str));
        }
        (Mn(num), Mi(id)) => {
            let ci_str = mi2ci(id.to_string());
            let cn_str = mn2cn(num.to_string());
            msup_str.push_str(&format!("{}{}", cn_str, ci_str));
        }
        _ => {
            panic!("Unhandled Msup")
        }
    }
    msup_str.push_str(&"</apply>".to_string());
    msup_str
}

/// Takes parsed  presentation MathML's Mfrac components and turns it into content MathML representation
fn parsed_mfrac(numerator: &Box<MathExpression>, denominator: &Box<MathExpression>) -> String {
    let mut mfrac_str = String::new();
    match (&**numerator, &**denominator) {
        (Mn(num1), Mn(num2)) => {
            let cn_str = mn2cn(num1.to_string());
            let cn2_str = mn2cn(num2.to_string());
            mfrac_str.push_str(&format!("<apply><divide/>{}{}</apply>", cn_str, cn2_str));
        }
        (Mi(id), Mn(num)) => {
            let ci_str = mi2ci(id.to_string());
            let cn_str = mn2cn(num.to_string());
            mfrac_str.push_str(&format!("<apply><divide/>{}{}</apply>", ci_str, cn_str));
        }
        (Mi(id1), Mi(id2)) => {
            let ci_str = mi2ci(id1.to_string());
            let ci2_str = mi2ci(id2.to_string());
            mfrac_str.push_str(&format!("<apply><divide/>{}{}</apply>", ci_str, ci2_str));
        }
        (Mrow(num_exp), Mrow(denom_exp)) => {
            if num_exp.len() == 2 {
                if let (Mi(num_id), Mi(denom_id)) = (&num_exp[0], &denom_exp[0]) {
                    if num_id == "d" && denom_id == "d" {
                        if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                            let ci_str = mi2ci(id1.to_string());
                            let ci0_str = mi2ci(id0.to_string());
                            mfrac_str.push_str(&format!(
                                "<apply><diff/><bvar>{}</bvar>{}</apply>",
                                ci_str, ci0_str
                            ))
                        }
                    } else if num_id == "∂" && denom_id == "∂" {
                        if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                            let ci_str = mi2ci(id1.to_string());
                            let ci0_str = mi2ci(id0.to_string());
                            mfrac_str.push_str(&format!(
                                "<apply><partialdiff/><bvar>{}</bvar>{}</apply>",
                                ci_str, ci0_str
                            ))
                        }
                    }
                }
            } else if num_exp.len() == 3 && denom_exp.len() == 2 {
                if let (Mi(num_id), Mi(denom_id)) = (&num_exp[0], &denom_exp[0]) {
                    if num_id == "d" && denom_id == "d" {
                        if let (Mi(id0), GroupTuple(group), Mi(id1)) =
                            (&num_exp[1], &num_exp[2], &denom_exp[1])
                        {
                            let ci_str = mi2ci(id1.to_string());
                            let ci0_str = mi2ci(id0.to_string());
                            let g_str = parenthesis_group(group);
                            mfrac_str.push_str(&format!(
                                "<apply><diff/><bvar>{}</bvar><apply>{}{}</apply></apply>",
                                ci_str, ci0_str, g_str
                            ))
                        }
                    } else if num_id == "∂" && denom_id == "∂" {
                        if let (Mi(id0), GroupTuple(group), Mi(id1)) =
                            (&num_exp[1], &num_exp[2], &denom_exp[1])
                        {
                            let ci_str = mi2ci(id1.to_string());
                            let ci0_str = mi2ci(id0.to_string());
                            let g_str = parenthesis_group(group);
                            mfrac_str.push_str(&format!(
                                "<apply><partialdiff/><bvar>{}</bvar><apply>{}{}</apply></apply>",
                                ci_str, ci0_str, g_str
                            ))
                        }
                    }
                }
            }
        }
        (Mrow(num_exp), Mi(id)) => {
            println!("-------------");
            println!("num_exp.len()={}", &num_exp.len());
            let frac_num = content_for_times_or_function_of((&num_exp).to_vec());
            let ci_str = mi2ci(id.to_string());
            mfrac_str.push_str(&format!("<apply><divide/>{}{}</apply>", frac_num, ci_str));
            println!("mfrac_str = {}", mfrac_str);
        }
        _ => {
            panic!("Unhandled Mfrac")
        }
    }
    mfrac_str
}

/// Takes parsed  presentation MathML's Msub components and turns it into content MathML representation
fn parsed_msub(comp1: &Box<MathExpression>, comp2: &Box<MathExpression>) -> String {
    let mut msub_str = String::new();
    msub_str.push_str(&"<apply><selector/>".to_string());
    match (&**comp1, &**comp2) {
        (Mi(id1), Mi(id2)) => {
            let ci1_str = mi2ci(id1.to_string());
            let ci2_str = mi2ci(id2.to_string());
            msub_str.push_str(&format!("{}{}", ci1_str, ci2_str))
        }
        (Mi(id), Mn(num)) => {
            let ci_str = mi2ci(id.to_string());
            let cn_str = mn2cn(num.to_string());
            msub_str.push_str(&format!("{}{}", ci_str, cn_str))
        }
        (Mn(num1), Mn(num2)) => {
            let cn1_str = mn2cn(num1.to_string());
            let cn2_str = mn2cn(num2.to_string());
            msub_str.push_str(&format!("{}{}", cn1_str, cn2_str))
        }
        (Mrow(row1), Mrow(row2)) => {
            let (r1_str, count_row1_op) = counting_operators(&row1);
            msub_str.push_str(&r1_str.to_string());
            for r1 in row1.iter() {
                match r1 {
                    Mi(id) => msub_str.push_str(&mi2ci(id.to_string())),
                    Mn(num) => msub_str.push_str(&mn2cn(num.to_string())),
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

            let (r2_op_str, count_row2_op) = counting_operators(&row2);
            msub_str.push_str(&r2_op_str.to_string());
            for r2 in row2.iter() {
                match r2 {
                    Mi(id) => msub_str.push_str(&mi2ci(id.to_string())),
                    Mn(num) => msub_str.push_str(&mn2cn(num.to_string())),
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
                let comp = if_two_operation_exists((&group).to_vec());
                group_str.push_str(&comp.to_string());
            }
            _ => {
                panic!("Unhandled  operations count in grouping parenthesis")
            }
        }
    } else {
        println!("Unhandled nested grouping with parenthesis");
    }
    println!("group_str = {}", group_str);
    group_str
}

/// Handles one operation of either Mo("Add") and Mo("Subtract") within in between other MathExpression
fn if_one_operation_exists(after_equals: Vec<MathExpression>) -> String {
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
        let before_components = content_for_times_or_function_of(before_op);
        mrow_str.push_str(&before_components.to_string());
        let after_components = content_for_times_or_function_of(after_op);
        mrow_str.push_str(&after_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());
    } else if before_op.is_empty() && !after_op.is_empty() {
        let after_components = content_for_times_or_function_of(after_op);
        mrow_str.push_str(&after_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());
    }
    mrow_str
}

/// Handles incorporating <times/> or "function of" between MathExpression of vector by looking at it's length and type MathExpression
fn content_for_times_or_function_of(x: Vec<MathExpression>) -> String {
    let mut str_component = String::new();
    match x.len() {
        1 => {
            let comp = vec_mathexp(&x);
            str_component.push_str(&comp.to_string());
        }
        2 => {
            for (i, comp) in x.iter().enumerate() {
                if i > 0 {
                    if let (Mi(id1), Mi(id2)) = (&x[i - 1], &x[i]) {
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        str_component.push_str(&format!("<apply><times/>{}{}</apply>", ci1, ci2));
                    } else if let (GroupTuple(group), Mi(id)) = (&x[i - 1], &x[i]) {
                        let group_comp = parenthesis_group(&group);
                        let ci = mi2ci(id.to_string());
                        str_component
                            .push_str(&format!("<apply><times/>{}{}</apply>", group_comp, ci));
                    } else if let (Mi(id), Mo(Rparenthesis)) = (&x[i - 1], &x[i]) {
                        str_component.push_str(&mi2ci(id.to_string()));
                    } else if let (Mfrac(num, denom), Mi(id)) = (&x[i - 1], &x[i]) {
                        let mfrac_comp = parsed_mfrac(&num, &denom);
                        let ci = mi2ci(id.to_string());
                        str_component
                            .push_str(&format!("<apply><times/>{}{}</apply>", mfrac_comp, ci));
                    } else if let (Mi(id), GroupTuple(group)) = (&x[i - 1], &x[i]) {
                        let ci = mi2ci(id.to_string());
                        let group_comp = parenthesis_group(&group);
                        str_component.push_str(&format!("<apply>{}{}</apply>", ci, group_comp));
                    } else {
                        println!("Unhandled combination of MathExpression of length 2.");
                    }
                }
            }
        }
        3 => {
            for (i, comp) in x.iter().enumerate() {
                if i > 1 {
                    if let (Mi(id1), Mi(id2), Mfrac(num, denom)) = (&x[i - 2], &x[i - 1], &x[i]) {
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        let mfrac_comp = parsed_mfrac(&num, &denom);
                        str_component.push_str(&format!(
                            "<apply><times/>{}{}{}</apply>",
                            ci1, ci2, mfrac_comp
                        ));
                    } else if let (Mi(id1), Mi(id2), Mi(id3)) = (&x[i - 2], &x[i - 1], &x[i]) {
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        let ci3 = mi2ci(id3.to_string());
                        str_component
                            .push_str(&format!("<apply><times/>{}{}{}</apply>", ci1, id2, id3));
                    } else {
                        println!("Unhandled combination of MathExpression of length 3.");
                    }
                }
            }
        }
        4 => {
            for (i, comp) in x.iter().enumerate() {
                if i > 2 {
                    if let (Mi(id1), Mi(id2), GroupTuple(group), Mfrac(num, denom)) =
                        (&x[i - 3], &x[i - 2], &x[i - 1], &x[i])
                    {
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        let group_comp = parenthesis_group(&group);
                        let mfrac_comp = parsed_mfrac(&num, &denom);

                        let combine = format!("<apply>{}{}</apply>", ci2, group_comp);
                        str_component.push_str(&format!(
                            "<apply><times/>{}{}{}</apply>",
                            ci1, combine, mfrac_comp
                        ));
                    }
                } else {
                    println!("-----Unhandled");
                }
            }
        }
        _ => {
            panic!("Unhandled length inside a vector for incorporating times functionality or function of functionality")
        }
    }
    str_component
}

///Handles two operation of either Mo("Add") and Mo("Subtract") within in between other MathExpression
fn if_two_operation_exists(after_equals: Vec<MathExpression>) -> String {
    let mut mrow_str = String::new();
    let mut before_op: Vec<MathExpression> = Vec::new();
    let mut after_op: Vec<MathExpression> = Vec::new();
    let mut after_op_exists = false;
    let operations: HashSet<&str> = ["+", "-"].iter().copied().collect();
    for comps in after_equals.iter() {
        if let Mo(operation) = &comps {
            let op = format!("{operation}");
            if operations.contains(&op.as_str()) {
                if after_op_exists {
                    after_op.push(comps.clone());
                } else {
                    after_op_exists = true;
                }
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

        let new_before_components = content_for_times_or_function_of(before_new_op);
        mrow_str.push_str(&new_before_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());

        let new_after_components = content_for_times_or_function_of(after_new_op);
        mrow_str.push_str(&new_after_components.to_string());
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
    }

    if !before_equals.is_empty() && !after_equals.is_empty() {
        println!("before_equals = {:?}", before_equals);
        println!("after_equals={:?}", after_equals);
        mrow_str.push_str(&"<apply><eq/>".to_string());

        let (b_op_str, count_before_op) = counting_operators(&before_equals);
        mrow_str.push_str(&b_op_str.to_string());

        let before_component = vec_mathexp(&before_equals);
        println!("before_component = {}", before_component);
        mrow_str.push_str(&before_component.to_string());

        let (a_op_str, count_after_op) = counting_operators(&after_equals);
        mrow_str.push_str(&a_op_str.to_string());
        println!("count_after_op={}", count_after_op);

        let mut before_minus: Vec<MathExpression> = Vec::new();
        let mut after_minus: Vec<MathExpression> = Vec::new();
        let mut minus_count = 0;
        let operations: HashSet<&str> = ["+", "-"].iter().copied().collect();
        if count_after_op == 1 {
            let one_ops = if_one_operation_exists(after_equals);
            mrow_str.push_str(&one_ops.to_string());
        } else if count_after_op == 2 {
            let two_ops_after_equals = if_two_operation_exists(after_equals);
            mrow_str.push_str(&two_ops_after_equals.to_string());
        } else {
            let after_component = vec_mathexp(&after_equals);
            mrow_str.push_str(&after_component.to_string());
            /* if count_after_op > 1 {
                //for _ in 0..count_after_op {
                mrow_str.push_str(&"</apply>".to_string());
                //}
            } else {
            }*/
        }
        mrow_str.push_str(&"</apply>".to_string());
    } else {
        let (row_str, count_op) = counting_operators(&row);
        mrow_str.push_str(&row_str.to_string());
        let row_comp = vec_mathexp(&row);
        mrow_str.push_str(&row_comp.to_string());
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
            match &**over1 {
                Mi(id) => mover_str.push_str(&mi2ci(id.to_string())),
                Mrow(comp) => {
                    let (op_str, count_op) = counting_operators(&comp);
                    mover_str.push_str(&op_str.to_string());
                    for c in comp.iter() {
                        match c {
                            Mi(id) => mover_str.push_str(&mi2ci(id.to_string())),
                            Mn(num) => mover_str.push_str(&mn2cn(num.to_string())),
                            Mo(op) => {}
                            _ => {
                                panic!("Unhandled comp inside Mover")
                            }
                        }
                    }

                    if count_op > 0 {
                        for _ in 0..count_op {
                            mover_str.push_str(&"</apply>".to_string());
                        }
                    } else {
                    }
                }

                _ => {
                    panic!("Unhandled comps inside Mover when over term is ‾")
                }
            }

            mover_str.push_str(&"</apply>".to_string());
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

pub fn to_content_mathml(pmathml: Vec<Math>) -> String {
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
    //let input = "tests/test_c2p_mml/test4.xml";
    //let input = "tests/sir.xml";
    //let input = "tests/seir_eq1.xml";
    let input = "tests/seir_eq2.xml";
    //let input = "tests/test_c2p_mml/test6.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><minus/><ci>y</ci><apply><conjugate/><apply><plus/><ci>x</ci><cn>1</cn></apply></apply></apply></math>");
}
#[test]
fn test_content_mml_seir_eq2() {
    let input = "tests/seir_eq2.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><ci>E</ci></apply><apply><minus/><apply><times/><ci>β</ci><ci>S</ci><apply><divide/><ci>I</ci><ci>N</ci></apply></apply><apply><times/><apply><plus/><ci>μ</ci><ci>ϵ</ci></apply><ci>E</ci></apply></apply></apply></math>");
}

#[test]
fn test_content_mml_seirdv_eq2() {
    let input = "tests/seirdv_eq2.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>s</ci></apply><apply><minus/><apply><minus/><ci>ı</ci><apply><times/><ci>μ</ci><ci>S</ci></apply></apply><apply><times/><apply><divide/><apply><times/><ci>β</ci><ci>I</ci></apply><ci>N</ci></apply><ci>S</ci></apply></apply></apply></math>")
}
#[test]
fn test_content_hackathon2_scenario1_eq1() {
    let input = "tests/h2_scenario1_eq1.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><apply><ci>S</ci><ci>t</ci></apply></apply><apply><minus/><apply><times/><ci>β</ci><apply><ci>I</ci><ci>t</ci></apply><apply><divide/><apply><ci>S</ci><ci>t</ci></apply><ci>N</ci></apply></apply></apply></apply></math>")
}

//fn main() {
////let input = "../tests/test_c2p_mml/test4.xml";
////let input = "../tests/sir.xml";
////let input = "../tests/seir_eq1.xml";
////let input = "../tests/seir_eq2.xml";
//let input = "../tests/seir_eq3.xml";
////let input = "../tests/seir_eq4.xml";
//let mut contents = std::fs::read_to_string(input)
//.unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
//let mut vector_mml = Vec::<Math>::new();
//let (_, mut math) =
//parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
////let parsed_mml = get_mathml_asts_from_file("../tests/test_c2p_mml/test6.xml");
//println!("math={:?}", math);
////math.normalize();
//vector_mml.push(math);

//println!("vector_mml={:?}", vector_mml);
//let mml = to_content_mathml(vector_mml);
//println!("mml={:?}", mml);
////assert_eq!(mml, "<math><apply><minus/><ci>y</ci><apply><conjugate/><apply><plus/><ci>x</ci><cn>1</cn></apply></apply></apply></math>");
//}

/*
fn main() {
    let mathml_exp = get_mathml_asts_from_file("../tests/test_c2p_mml/test6.xml");
    println!("mathml_exp= {:?}", mathml_exp);
    let cmml = parsed_pmathml_2_cmathml(mathml_exp);
    println!("cmml={:}", cmml);
}
*/
