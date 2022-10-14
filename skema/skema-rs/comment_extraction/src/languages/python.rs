use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_until, take_while},
    character::complete::{char, line_ending, multispace0, not_line_ending, space0},
    multi::fold_many0,
    sequence::{delimited, tuple},
    IResult,
};
use nom_locate::{position, LocatedSpan};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, strum_macros::Display, Clone, Serialize, Deserialize)]
enum Statement {
    Comment {
        line_number: u32,
        contents: String,
    },
    Docstring {
        object_name: String,
        contents: Vec<String>,
    },
    Other,
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
    pub comments: Vec<(u32, String)>,
    pub docstrings: HashMap<String, Vec<String>>,
}

/// Whole line comments
fn whole_line_comment(input: Span) -> IResult<Span, Statement> {
    let (s, x) = delimited(
        tuple((multispace0, char('#'))),
        not_line_ending,
        line_ending,
    )(input)?;
    let (_, pos) = position(s)?;
    let line_number = pos.location_line() - 1;
    let contents = x.to_string();
    Ok((
        s,
        Statement::Comment {
            line_number,
            contents,
        },
    ))
}

/// Partial line comments
fn partial_line_comment(input: Span) -> IResult<Span, Statement> {
    let (s, (_, _, x, _)) =
        tuple((is_not("#\n\r"), char('#'), not_line_ending, line_ending))(input)?;
    let (_, pos) = position(s)?;
    let line_number = pos.location_line() - 1;
    let contents = x.to_string();
    Ok((
        s,
        Statement::Comment {
            line_number,
            contents,
        },
    ))
}

fn docstring(input: Span) -> IResult<Span, Statement> {
    let func_declaration = delimited(
        tuple((space0, tag("def "))),
        name,
        tuple((tag("("), take_until("):"), tag("):"), line_ending, space0)),
    );
    let (input, (func_name, docstring_contents, _, _)) =
        tuple((func_declaration, triple_quoted_string, space0, line_ending))(input)?;
    let object_name = func_name.to_string();
    let contents: Vec<String> = docstring_contents
        .to_string()
        .split('\n')
        .map(|x| x.to_string())
        .collect();
    Ok((
        input,
        Statement::Docstring {
            object_name,
            contents,
        },
    ))
}

fn comment(input: Span) -> IResult<Span, Statement> {
    alt((docstring, whole_line_comment, partial_line_comment))(input)
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
            Statement::Comment {
                line_number,
                contents,
            } => {
                acc.comments.push((line_number, contents));
            }
            Statement::Docstring {
                object_name,
                contents,
            } => {
                acc.docstrings.insert(object_name, contents);
            }
            _ => (),
        }
        acc
    })(input)
}

pub fn get_comments(src_file_path: &str) -> Comments {
    let contents = std::fs::read_to_string(src_file_path).unwrap();
    let span = Span::new(&contents);
    let (_, result) = comments(span).unwrap();
    result
}

#[test]
fn test_parser() {
    get_comments("tests/data/CHIME_SIR.py");
}
