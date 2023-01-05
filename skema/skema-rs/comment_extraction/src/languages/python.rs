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

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub line_number: u32,
    pub contents: String,
}

#[derive(Debug, strum_macros::Display, Clone, Serialize, Deserialize)]
enum Statement {
    Comment(Comment),
    Other,
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
        Statement::Comment(Comment {
            line_number,
            contents,
        }),
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
        Statement::Comment(Comment {
            line_number,
            contents,
        }),
    ))
}

fn comment(input: Span) -> IResult<Span, Statement> {
    alt((whole_line_comment, partial_line_comment))(input)
}

fn other(input: Span) -> IResult<Span, Statement> {
    let (s, _) = tuple((not_line_ending, line_ending))(input)?;
    Ok((s, Statement::Other))
}

fn statement(input: Span) -> IResult<Span, Statement> {
    alt((comment, other))(input)
}

fn comments(input: Span) -> IResult<Span, Vec<Comment>> {
    fold_many0(
        statement,
        Vec::<Comment>::new,
        |mut acc: Vec<Comment>, item| {
            match item {
                Statement::Comment(comment) => {
                    acc.push(comment);
                }
                _ => (),
            }
            acc
        },
    )(input)
}

pub fn get_comments(src_file_path: &str) -> Vec<Comment> {
    let contents = std::fs::read_to_string(src_file_path).unwrap();
    get_comments_from_string(&contents)
}

pub fn get_comments_from_string(source_code: &str) -> Vec<Comment> {
    let span = Span::new(&source_code);
    let (_, result) = comments(span).unwrap();
    result
}

#[test]
fn test_parser() {
    // Smokescreen test to see if the extraction works without crashing. May want to replace this
    // with a more rigorous test later.
    get_comments("tests/data/CHIME_SIR.py");
}
