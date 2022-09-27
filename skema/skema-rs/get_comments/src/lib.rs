use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_until, take_while},
    character::complete::{anychar, none_of, not_line_ending, line_ending, space0, alpha1, alphanumeric0},
    character::is_alphanumeric,
    combinator::{opt, recognize},
    multi::{fold_many0, many0},
    sequence::{pair, preceded, tuple, delimited},
    IResult,
};
use std::collections::HashMap;
use nom_locate::{position, LocatedSpan};

type Span<'a> = LocatedSpan<&'a str>;

/// Whole line comments
fn whole_line_comment(input: Span) -> IResult<Span, Span> {
    delimited(tuple((space0, tag("#"))), not_line_ending, line_ending)(input)
}

/// Triple quoted strings
fn triple_quoted_string(input:Span) -> IResult<Span, Span> {
    alt((delimited(tag("'''"), take_until("'''"), tag("'''")),
    delimited(tag("\"\"\""), take_until("\"\"\""), tag("\"\"\""))))(input)
}

fn is_valid_name_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn name(input:Span) -> IResult<Span, Span> {
    take_while(is_valid_name_character)(input)
}


#[derive(Debug, Default)]
struct Docstring {
    pub object_name: String,
    pub contents: Vec<String>
}

#[derive(Debug, PartialEq, Default)]
pub struct Comments {
    pub lineno: usize,
    pub whole_line_comments: HashMap<usize, String>,
    pub docstrings: HashMap<String, Vec<String>>
}

#[derive(Debug)]
enum Comment {
    String,
    Docstring,
}


fn docstring(input: Span) -> IResult<Span, Span> {
    let (s, (function_name)) = tuple(space0, tag("def "), name, tag("("), take_until())
    //preceded(tuple((tag("def "), name, tag("("), take_until("):"))), triple_quoted_string)(input)
}


fn comment(input: Span) -> IResult<Span, Span> {
    alt((whole_line_comment, docstring))(input)
}

fn comments(input: Span) -> IResult<Span, Comments> {
    fold_many0(
        comment,
        Comments::default,
        |mut acc: Comments, item| {
            acc.lineno += 1;
            acc.whole_line_comments.insert(acc.lineno, item.to_string());
            acc
        },
    )(input)
}

fn parse(input: Span) -> IResult<Span, Comments> {
    let (remaining_input, matched) = comments(input)?;
    let (s, pos) = position(remaining_input)?;
    dbg!(pos);

    Ok((
        remaining_input,
        matched
    ))
}

#[test]
fn test_parser() {
    let contents = std::fs::read_to_string("tests/data/python_example.py").unwrap();
    let span = Span::new(&contents);
    let state = comments(span);
    dbg!(state);
}
