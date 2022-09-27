use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_until, take_while},
    character::complete::{
        alpha1, alphanumeric0, anychar, line_ending, none_of, not_line_ending, space0,
    },
    character::is_alphanumeric,
    combinator::{opt, recognize, not, value},
    multi::{fold_many0, many0},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};
use serde::{Deserialize, Serialize};
use serde_json;
use strum_macros::Display; // used for macro on enums // used for macro on enums
use nom_locate::{position, LocatedSpan};
use std::collections::HashMap;

type Span<'a> = LocatedSpan<&'a str>;


#[derive(Debug, Display, Clone, Serialize, Deserialize)]
enum Statement {
    WholeLineComment(u32, String),
    Docstring { object_name: String, contents: Vec<String> },
    Other
}


/// Triple quoted strings
fn triple_quoted_string(input: Span) -> IResult<Span, Span> {
    alt((
        delimited(tag("'''"), take_until("'''"), tag("'''")),
        delimited(tag("\"\"\""), take_until("\"\"\""), tag("\"\"\"")),
    ))(input)
}

fn is_valid_name_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn name(input: Span) -> IResult<Span, Span> {
    take_while(is_valid_name_character)(input)
}


#[derive(Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Comments {
    pub whole_line_comments: Vec<(u32, String)>,
    pub docstrings: HashMap<String, Vec<String>>,
}

/// Whole line comments
fn whole_line_comment(input: Span) -> IResult<Span, Statement> {
    let (s, x) = delimited(tuple((space0, tag("#"))), not_line_ending, line_ending)(input)?;
    let (s, pos) = position(s)?;
    let line = pos.location_line() - 1;
    Ok((s, Statement::WholeLineComment(line, x.to_string())))
}

fn docstring(input: Span) -> IResult<Span, Statement> {
    let mut func_declaration = delimited(
        tuple((space0, tag("def "))),
        name,
        tuple((tag("("), take_until("):"), tag("):"), line_ending, space0)),
    );
    let (input, (func_name, docstring_contents, _, _)) = tuple((func_declaration, triple_quoted_string, space0, line_ending))(input)?;
    let object_name = func_name.to_string();
    let contents: Vec<String> = docstring_contents.to_string().split('\n').map(|x| {x.to_string()}).collect();
    Ok((input, Statement::Docstring { object_name, contents }))
}

fn comment(input: Span) -> IResult<Span, Statement> {
    alt((whole_line_comment, docstring))(input)
}

fn other(input: Span) -> IResult<Span, Statement> {
    let (s, _) = tuple((not_line_ending, line_ending))(input)?;
    Ok((s, Statement::Other))
}

fn statement(input: Span) -> IResult<Span, Statement> {
    alt((comment, other))(input)
}


fn comments(input: Span) -> IResult<Span, Comments> {
    fold_many0(statement, Comments::default, |mut acc: Comments, item| {
        match item {
            Statement::WholeLineComment(line, x) => {
                acc.whole_line_comments.push((line, x));
            }
            Statement::Docstring {object_name, contents} => {
                acc.docstrings.insert(object_name, contents);
            }
            _ => ()
        }
        acc
    })(input)
}

fn parse(input: Span) -> IResult<Span, Comments> {
    let (remaining_input, matched) = comments(input)?;
    let (s, pos) = position(remaining_input)?;

    Ok((remaining_input, matched))
}

#[test]
fn test_parser() {
    let contents = std::fs::read_to_string("tests/data/CHIME_SIR.py").unwrap();
    let span = Span::new(&contents);
    let result = comments(span);
    match result {
        Ok((_, c)) => println!("{}", serde_json::to_string(&c).unwrap()),
        _ => panic!("Error getting the comments"),
    };

}
