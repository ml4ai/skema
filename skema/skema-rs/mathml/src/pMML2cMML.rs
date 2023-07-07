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
            let op = format!("{operation}");
            if op == "+" {
                count += 1;
                op_str.push_str(&"<apply><plus/>".to_string());
            } else if op == "-" {
                count += 1;
                op_str.push_str(&"<apply><minus/>".to_string());
            } else if op == ")" || op == "(" {
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
                let mrow_comp = parse_nested_operations(row);
                exp_str.push_str(&mrow_comp.to_string());
            }
            GroupTuple(group) => {
                let group_comp = parenthesis_group(&mut group.to_vec());
                exp_str.push_str(&group_comp.to_string());
            }
            _ => {
                panic!("Unhandled components in vec_mathexp function")
            }
        }
    }
    exp_str
}

/// Takes parsed presentation MathML's Msup components and translates it into content MathML representation
fn parsed_msup(comp1: &Box<MathExpression>, comp2: &Box<MathExpression>) -> String {
    let mut msup_str = String::new();
    match (&**comp1, &**comp2) {
        (Mi(id), Mn(num)) => {
            let ci_str = mi2ci(id.to_string());
            let cn_str = mn2cn(num.to_string());
            msup_str.push_str(&format!("<apply><power/>{}{}</apply>", ci_str, cn_str));
        }
        (Mi(id1), Mi(id2)) => {
            let ci_str = mi2ci(id1.to_string());
            let ci2_str = mi2ci(id2.to_string());
            msup_str.push_str(&format!("<apply><power/>{}{}</apply>", ci_str, ci2_str));
        }
        (Mn(num1), Mn(num2)) => {
            let cn_str = mn2cn(num1.to_string());
            let cn2_str = mn2cn(num2.to_string());
            msup_str.push_str(&format!("<apply><power/>{}{}</apply>", cn_str, cn2_str));
        }
        (Mn(num), Mi(id)) => {
            let ci_str = mi2ci(id.to_string());
            let cn_str = mn2cn(num.to_string());
            msup_str.push_str(&format!("<apply><power/>{}{}</apply>", cn_str, ci_str));
        }
        (Mi(id), Mrow(row)) => {
            if id == "e" {
                msup_str.push_str(&"<apply><exp/>".to_string());
                let row_comp = minus_at_vector_position0(&mut row.to_vec());
                msup_str.push_str(&row_comp.to_string());
                msup_str.push_str(&"</apply>".to_string());
            }
        }
        _ => {
            panic!("Unhandled Msup")
        }
    }
    msup_str
}

/// Takes parsed  presentation MathML's Mfrac components and translates it into content MathML representation
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
                            let ci0_str = mi2ci(id0.to_string());
                            mfrac_str.push_str(&format!("<apply><diff/>{}</apply>", ci0_str))
                        }
                    } else if num_id == "∂" && denom_id == "∂" {
                        if let (Mi(id0), Mi(id1)) = (&num_exp[1], &denom_exp[1]) {
                            let ci0_str = mi2ci(id0.to_string());
                            mfrac_str.push_str(&format!("<apply><partialdiff/>{}</apply>", ci0_str))
                        }
                    }
                }
            } else if num_exp.len() == 3 && denom_exp.len() == 2 {
                if let (Mi(num_id), Mi(denom_id)) = (&num_exp[0], &denom_exp[0]) {
                    if num_id == "d" && denom_id == "d" {
                        if let (Mi(id0), GroupTuple(group), Mi(id1)) =
                            (&num_exp[1], &num_exp[2], &denom_exp[1])
                        {
                            let ci0_str = mi2ci(id0.to_string());
                            mfrac_str.push_str(&format!("<apply><diff/>{}</apply>", ci0_str))
                        }
                    } else if num_id == "∂" && denom_id == "∂" {
                        if let (Mi(id0), GroupTuple(group), Mi(id1)) =
                            (&num_exp[1], &num_exp[2], &denom_exp[1])
                        {
                            let ci0_str = mi2ci(id0.to_string());
                            mfrac_str.push_str(&format!("<apply><partialdiff/>{}</apply>", ci0_str))
                        }
                    }
                }
            } else {
                let (num_str, count_num_op) = counting_operators(&num_exp);
                let (denom_str, count_denom_op) = counting_operators(&denom_exp);
                if count_num_op == 1 && count_denom_op == 1 {
                    mfrac_str.push_str(&"<apply><divide/>");
                    mfrac_str.push_str(&num_str.to_string());
                    let (n_before_op, n_after_op) = if_one_operation_exists(num_exp.to_vec());
                    mfrac_str.push_str(&format!("{}{}", n_before_op, n_after_op));
                    mfrac_str.push_str(&denom_str.to_string());
                    let (d_before_op, d_after_op) = if_one_operation_exists(denom_exp.to_vec());
                    mfrac_str.push_str(&format!("{}{}", d_before_op, d_after_op));
                } else {
                    println!("Unhandled numerator Mrow and denominator Mrow in Mfrac");
                }
            }
        }
        (Mrow(num_exp), Mi(id)) => {
            let frac_num = content_for_times((&num_exp).to_vec());
            let ci_str = mi2ci(id.to_string());
            mfrac_str.push_str(&format!("<apply><divide/>{}{}</apply>", frac_num, ci_str));
        }
        _ => {
            panic!("Unhandled Mfrac")
        }
    }
    mfrac_str
}

/// Takes parsed  presentation MathML's Msub components and translates it into content MathML representation
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
                        panic!("Unhandled component inside Mover")
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

/// Takes parsed presentation MathML's Mover components and translates it into content MathML representation
/// Currently only handles "\bar" (<conjugate/> only, does not handles mean yet) and "\dot"
fn parsed_mover(over1: &Box<MathExpression>, over2: &Box<MathExpression>) -> String {
    let mut mover_str = String::new();

    if let Mo(over_op) = &**over2 {
        let over_term = format!("{over_op}");
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
    mover_str
}
/// Takes parsed presentation MathML's grouped parenthesis components and translates it into content MathML representation.
/// Currently cannot handles nested parenthesis.
fn parenthesis_group(group: &mut Vec<MathExpression>) -> String {
    let mut group_str = String::new();
    let mut count_groups = 0;
    let grouping = group.iter().for_each(|i| {
        if let GroupTuple(expression) = i {
            count_groups += 1;
        }
    });

    if count_groups == 0 {
        let group_comp = minus_at_vector_position0(&mut group.to_vec());
        group_str.push_str(&group_comp.to_string());
    } else {
        println!("Unhandled nested grouping with parenthesis");
    }
    group_str
}

/// Handles one operation of Mo("Add") and Mo("Subtract") in between other MathExpression
fn if_one_operation_exists(after_equals: Vec<MathExpression>) -> (String, String) {
    let mut before_str = String::new();
    let mut after_str = String::new();
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
    if !before_op.is_empty() && !after_op.is_empty() {
        let before_components = content_for_times(before_op);
        before_str.push_str(&before_components.to_string());
        let after_components = content_for_times(after_op);
        after_str.push_str(&after_components.to_string());
        after_str.push_str(&"</apply>".to_string());
    } else if before_op.is_empty() && !after_op.is_empty() {
        let after_components = content_for_times(after_op);
        after_str.push_str(&after_components.to_string());
        after_str.push_str(&"</apply>".to_string());
    } else if !before_op.is_empty() && after_op.is_empty() {
        let before_components = content_for_times(before_op);
        before_str.push_str(&before_components.to_string());
        before_str.push_str(&"</apply>".to_string());
    }
    (before_str, after_str)
}

///Handles two operation of Mo("Add") and Mo("Subtract") in between other MathExpression
fn if_two_operation_exists(after_equals: Vec<MathExpression>) -> (String, String) {
    let mut before_str = String::new();
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
    let mut before_new_op: Vec<MathExpression> = Vec::new();
    let mut after_new_op: Vec<MathExpression> = Vec::new();
    if !before_op.is_empty() && !after_op.is_empty() {
        let b_op = content_for_times(before_op);
        before_str.push_str(&b_op.to_string());

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
        let new_before_components = content_for_times(before_new_op);
        mrow_str.push_str(&new_before_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());

        let new_after_components = content_for_times(after_new_op);
        mrow_str.push_str(&new_after_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());
    } else if before_op.is_empty() && !after_op.is_empty() {
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

        let new_before_components = content_for_times(before_new_op);
        mrow_str.push_str(&new_before_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());

        let new_after_components = content_for_times(after_new_op);
        mrow_str.push_str(&new_after_components.to_string());
        mrow_str.push_str(&"</apply>".to_string());
    }
    (before_str, mrow_str)
}

/// Handles incorporating <times/> and parenthesized component (e.g. (x+1)) between MathExpression of vector by looking at it's length and type MathExpression
fn content_for_times(x: Vec<MathExpression>) -> String {
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
                        let group_comp = parenthesis_group(&mut group.to_vec());
                        let ci = mi2ci(id.to_string());
                        str_component
                            .push_str(&format!("<apply><times/>{}{}</apply>", group_comp, ci));
                    } else if let (Mi(id), Mo(Rparenthesis)) = (&x[i - 1], &x[i]) {
                        str_component.push_str(&mi2ci(id.to_string()));
                    } else if let (Mo(Lparenthesis), Mn(num)) = (&x[i - 1], &x[i]) {
                        str_component.push_str(&mn2cn(num.to_string()));
                    } else if let (Mo(Lparenthesis), Mi(id)) = (&x[i - 1], &x[i]) {
                        str_component.push_str(&mi2ci(id.to_string()));
                    } else if let (Msub(s1, s2), Mo(Rparenthesis)) = (&x[i - 1], &x[i]) {
                        let sub1_comp = parsed_msub(&s1, &s2);
                        str_component.push_str(&sub1_comp.to_string());
                    } else if let (Msub(s1, s2), Msub(s3, s4)) = (&x[i - 1], &x[i]) {
                        let sub1_comp = parsed_msub(&s1, &s2);
                        let sub2_comp = parsed_msub(&s3, &s4);
                        str_component.push_str(&format!(
                            "<apply><times/>{}{}</apply>",
                            sub1_comp, sub2_comp
                        ));
                    } else if let (Mfrac(num, denom), Mi(id)) = (&x[i - 1], &x[i]) {
                        let mfrac_comp = parsed_mfrac(&num, &denom);
                        let ci = mi2ci(id.to_string());
                        str_component
                            .push_str(&format!("<apply><times/>{}{}</apply>", mfrac_comp, ci));
                    } else if let (Mi(id), GroupTuple(group)) = (&x[i - 1], &x[i]) {
                        let ci = mi2ci(id.to_string());
                        let group_comp = parenthesis_group(&mut group.to_vec());

                        let (_, count_op) = counting_operators(&group.to_vec());
                        if count_op == 0 {
                            str_component.push_str(&mi2ci(id.to_string()));
                        } else {
                            str_component
                                .push_str(&format!("<apply><times/>{}{}</apply>", ci, group_comp));
                        }
                    } else if let (Mi(id), Mrow(row)) = (&x[i - 1], &x[i]) {
                        let ci = mi2ci(id.to_string());

                        let exp = vec_mathexp(&row);
                        str_component.push_str(&format!("<apply><times/>{}{}</apply>", ci, exp));
                    } else {
                        println!("Unhandled MathExpression combination of length 2.");
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
                            .push_str(&format!("<apply><times/>{}{}{}</apply>", ci1, ci2, ci3));
                    } else if let (Mi(id1), Mi(id2), GroupTuple(group)) =
                        (&x[i - 2], &x[i - 1], &x[i])
                    {
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        let group_comp = parenthesis_group(&mut group.to_vec());

                        let (_, count_op) = counting_operators(&group);
                        if count_op == 0 {
                            str_component
                                .push_str(&format!("<apply><times/>{}{}</apply>", ci1, ci2));
                        } else {
                            str_component.push_str(&format!(
                                "<apply><times/>{}{}{}</apply>",
                                ci1, ci2, group_comp
                            ));
                        }
                    } else {
                        println!("Unhandled MathExpression combination of length 3.");
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
                        let group_comp = parenthesis_group(&mut group.to_vec());
                        let mfrac_comp = parsed_mfrac(&num, &denom);

                        let (_, count_op) = counting_operators(&group);
                        if count_op == 0 {
                            str_component.push_str(&format!(
                                "<apply><times/>{}{}{}</apply>",
                                ci1, ci2, mfrac_comp
                            ));
                        } else {
                            str_component.push_str(&format!(
                                "<apply><times/>{}{}{}{}</apply>",
                                ci1, ci2, group_comp, mfrac_comp
                            ));
                        }
                    } else if let (GroupTuple(group1), Mi(id1), Mi(id2), GroupTuple(group2)) =
                        (&x[i - 3], &x[i - 2], &x[i - 1], &x[i])
                    {
                        let group1_comp = parenthesis_group(&mut group1.to_vec());
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        let group2_comp = parenthesis_group(&mut group2.to_vec());

                        let (_, count_op) = counting_operators(&group2);
                        if count_op == 0 {
                            str_component.push_str(&format!(
                                "<apply><times/>{}{}{}</apply>",
                                group1_comp, ci1, ci2
                            ));
                        } else {
                            str_component.push_str(&format!(
                                "<apply><times/>{}{}{}{}</apply>",
                                group1_comp, ci1, ci2, group2_comp
                            ));
                        }
                    } else if let (Mi(id1), Mi(id2), Mi(id3), GroupTuple(group)) =
                        (&x[i - 3], &x[i - 2], &x[i - 1], &x[i])
                    {
                        let ci1 = mi2ci(id1.to_string());
                        let ci2 = mi2ci(id2.to_string());
                        let ci3 = mi2ci(id3.to_string());
                        let group_comp = parenthesis_group(&mut group.to_vec());

                        let (_, count_op) = counting_operators(&group);
                        if count_op == 0 {
                            str_component
                                .push_str(&format!("<apply><times/>{}{}{}</apply>", ci1, ci2, ci3));
                        } else {
                            str_component.push_str(&format!(
                                "<apply><times/>{}{}{}{}</apply>",
                                ci1, ci2, ci3, group_comp
                            ));
                        }
                    } else {
                        println!("Unhandled MathExpression combination of length 4");
                    }
                }
            }
        }
        _ => {
            panic!("Unhandled length inside a vector for incorporating times functionality or function of functionality")
        }
    }
    str_component
}

/// Handles cases with minus operation at the beginning of the vector
fn minus_at_vector_position0(component: &mut Vec<MathExpression>) -> String {
    let mut mrow_str = String::new();

    if let Mo(operation) = &component[0] {
        let op = format!("{operation}");
        if op == "-" {
            component.remove(0);

            let (a_op_str, count_after_op) = counting_operators(&component);
            mrow_str.push_str(&a_op_str.to_string());
            mrow_str.push_str(&"<apply><minus/>".to_string());
            match count_after_op {
                0 => {
                    let (before_ops, _) = if_one_operation_exists(component.to_vec());
                    mrow_str.push_str(&before_ops.to_string());
                }
                1 => {
                    let (before_ops, after_ops) = if_one_operation_exists(component.to_vec());
                    mrow_str.push_str(&before_ops.to_string());
                    mrow_str.push_str(&"</apply>".to_string());
                    mrow_str.push_str(&after_ops.to_string());
                }
                2 => {
                    let (before_ops, after_ops) = if_two_operation_exists(component.to_vec());
                    mrow_str.push_str(&before_ops.to_string());
                    mrow_str.push_str(&"</apply>".to_string());
                    mrow_str.push_str(&after_ops.to_string());
                }
                _ => {
                    println!("Unhandled number of operations when <minus/> exists.");
                }
            }
            mrow_str.push_str(&"</apply>".to_string());
        } else if op == "(" {
            if let Mo(another_operation) = &component[1] {
                let op = format!("{another_operation}");
                if op == "-" {
                    component.remove(1);

                    let (a_op_str, count_after_op) = counting_operators(&component);
                    mrow_str.push_str(&a_op_str.to_string());
                    mrow_str.push_str(&"<apply><minus/>".to_string());
                    match count_after_op {
                        0 => {}
                        1 => {
                            let (before_ops, after_ops) =
                                if_one_operation_exists(component.to_vec());
                            mrow_str.push_str(&before_ops.to_string());
                            mrow_str.push_str(&"</apply>".to_string());
                            mrow_str.push_str(&after_ops.to_string());
                        }
                        2 => {
                            let (before_ops, after_ops) =
                                if_two_operation_exists(component.to_vec());
                            mrow_str.push_str(&before_ops.to_string());
                            mrow_str.push_str(&"</apply>".to_string());
                            mrow_str.push_str(&after_ops.to_string());
                        }
                        _ => {
                            println!("Unhandled number of operations when <minus/> exists.");
                        }
                    }
                }
            } else {
                let (a_op_str, count_after_op) = counting_operators(&component);
                mrow_str.push_str(&a_op_str.to_string());
                match count_after_op {
                    0 => {}
                    1 => {
                        let (before_ops, after_ops) = if_one_operation_exists(component.to_vec());
                        mrow_str.push_str(&before_ops.to_string());
                        mrow_str.push_str(&after_ops.to_string());
                    }
                    2 => {
                        let (before_ops, after_ops) = if_two_operation_exists(component.to_vec());
                        mrow_str.push_str(&before_ops.to_string());
                        mrow_str.push_str(&after_ops.to_string());
                    }
                    _ => {
                        let after_component = content_for_times(component.to_vec());
                        mrow_str.push_str(&after_component.to_string());
                    }
                }
            }
        } else {
            let (a_op_str, count_after_op) = counting_operators(&component);
            mrow_str.push_str(&a_op_str.to_string());
            match count_after_op {
                0 => {}
                1 => {
                    let (before_ops, after_ops) = if_one_operation_exists(component.to_vec());
                    mrow_str.push_str(&before_ops.to_string());
                    mrow_str.push_str(&after_ops.to_string());
                }
                2 => {
                    let (before_ops, after_ops) = if_two_operation_exists(component.to_vec());
                    mrow_str.push_str(&before_ops.to_string());
                    mrow_str.push_str(&after_ops.to_string());
                }
                _ => {
                    let after_component = content_for_times(component.to_vec());
                    mrow_str.push_str(&after_component.to_string());
                }
            }
        }
    } else {
        let (a_op_str, count_after_op) = counting_operators(&component);
        mrow_str.push_str(&a_op_str.to_string());
        match count_after_op {
            0 => {
                let row_comp = content_for_times(component.to_vec());
                mrow_str.push_str(&row_comp.to_string());
            }
            1 => {
                let (before_ops, after_ops) = if_one_operation_exists(component.to_vec());
                mrow_str.push_str(&before_ops.to_string());
                mrow_str.push_str(&after_ops.to_string());
            }
            2 => {
                let (before_ops, after_ops) = if_two_operation_exists(component.to_vec());
                mrow_str.push_str(&before_ops.to_string());
                mrow_str.push_str(&after_ops.to_string());
            }
            _ => {
                let after_component = content_for_times(component.to_vec());
                mrow_str.push_str(&after_component.to_string());
            }
        }

        mrow_str.push_str(&"</apply>".to_string());
    }
    mrow_str
}

/// Handles parsed nested operation components as well as nested MathExpression and translates to content
/// mathml
fn parse_nested_operations(row: &Vec<MathExpression>) -> String {
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
        mrow_str.push_str(&"<apply><eq/>".to_string());

        let (b_op_str, count_before_op) = counting_operators(&before_equals);
        mrow_str.push_str(&b_op_str.to_string());

        let before_component = content_for_times(before_equals);
        mrow_str.push_str(&before_component.to_string());

        let comp = minus_at_vector_position0(&mut after_equals.to_vec());

        mrow_str.push_str(&comp.to_string());
    } else {
        let comp = minus_at_vector_position0(&mut row.to_vec());
        mrow_str.push_str(&comp.to_string());
    }
    mrow_str
}

/// This is the main function that translates parsed presentation mathml to content mathml
pub fn to_content_mathml(pmathml: Vec<Math>) -> String {
    let mut cmathml = String::new();
    for pmml in pmathml.iter() {
        cmathml.push_str(&"<math>".to_string());
        let comp = &pmml.content;
        let components = parse_nested_operations(&comp);
        cmathml.push_str(&components.to_string());
    }
    cmathml.push_str(&"</math>".to_string());
    cmathml
}

#[test]
fn test_content_mml() {
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
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><ci>s</ci></apply><apply><minus/><apply><minus/><ci>ı</ci><apply><times/><ci>μ</ci><ci>S</ci></apply></apply><apply><times/><apply><divide/><apply><times/><ci>β</ci><ci>I</ci></apply><ci>N</ci></apply><ci>S</ci></apply></apply></apply></math>")
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
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><ci>S</ci></apply><apply><minus/><apply><times/><ci>β</ci><ci>I</ci><apply><divide/><ci>S</ci><ci>N</ci></apply></apply></apply></apply></math>")
}

#[test]
fn test_content_hackathon2_scenario1_eq2() {
    let input = "tests/h2_scenario1_eq2.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><ci>E</ci></apply><apply><minus/><apply><times/><ci>β</ci><ci>I</ci><apply><divide/><ci>S</ci><ci>N</ci></apply></apply><apply><times/><ci>δ</ci><ci>E</ci></apply></apply></apply></math>")
}

#[test]
fn test_content_hackathon2_scenario1_eq3() {
    let input = "tests/h2_scenario1_eq3.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><ci>I</ci></apply><apply><minus/><apply><minus/><apply><times/><ci>δ</ci><ci>E</ci></apply><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci><ci>I</ci></apply></apply><apply><times/><ci>α</ci><ci>ρ</ci><ci>I</ci></apply></apply></apply></math>")
}

#[test]
fn test_content_hackathon2_scenario1_eq4() {
    let input = "tests/h2_scenario1_eq4.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml,"<math><apply><eq/><apply><diff/><ci>R</ci></apply><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci><ci>I</ci></apply></apply></math>")
}

#[test]
fn test_content_hackathon2_scenario1_eq5() {
    let input = "tests/h2_scenario1_eq5.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(mml, "<math><apply><eq/><apply><diff/><ci>D</ci></apply><apply><times/><ci>α</ci><ci>ρ</ci><ci>I</ci></apply></apply></math>")
}

#[test]
fn test_content_hackathon2_scenario1_eq6() {
    let input = "tests/h2_scenario1_eq6.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(
        mml,"<math><apply><eq/><apply><diff/><ci>S</ci></apply><apply><plus/><apply><minus/><apply><times/><ci>β</ci><ci>I</ci><apply><divide/><ci>S</ci><ci>N</ci></apply></apply></apply><apply><times/><ci>ϵ</ci><ci>R</ci></apply></apply></apply></math>"  )
}

#[test]
fn test_content_hackathon2_scenario1_eq7() {
    let input = "tests/h2_scenario1_eq7.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(
        mml, "<math><apply><eq/><apply><diff/><ci>R</ci></apply><apply><minus/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci><ci>I</ci></apply><apply><times/><ci>ϵ</ci><ci>R</ci></apply></apply></apply></math>")
}

#[test]
fn test_content_hackathon2_scenario1_eq8() {
    let input = "tests/h2_scenario1_eq8.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(
        mml,
        "<math><apply><eq/><ci>β</ci><apply><times/><ci>κ</ci><ci>m</ci></apply></apply></math>"
    )
}

#[test]
fn test_content_hackathon2_scenario1_eq9() {
    let input = "tests/h2_scenario1_eq9.xml";
    let mut contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut vector_mml = Vec::<Math>::new();
    let (_, mut math) =
        parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    println!("math={:?}", math);
    vector_mml.push(math);
    let mml = to_content_mathml(vector_mml);
    println!("mml={:?}", mml);
    assert_eq!(
        mml,"<math><apply><eq/><ci>m</ci><apply><plus/><apply><divide/><apply><minus/><apply><selector/><ci>β</ci><ci>s</ci></apply><apply><selector/><ci>β</ci><ci>c</ci></apply></apply><apply><plus/><cn>1</cn><apply><exp/><apply><minus/><apply><times/><ci>k</ci><apply><plus/><apply><minus/><ci>t</ci></apply><apply><selector/><ci>t</ci><cn>0</cn></apply></apply></apply></apply></apply></apply></apply><apply><selector/><ci>β</ci><ci>c</ci></apply></apply></apply></math>"
    )
}
