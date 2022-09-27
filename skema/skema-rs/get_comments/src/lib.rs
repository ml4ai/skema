use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_until, take_while},
    character::complete::{anychar, none_of, not_line_ending, line_ending, space0},
    combinator::{opt, recognize},
    multi::fold_many0,
    sequence::{pair, preceded, tuple},
    IResult,
};

#[derive(Debug, PartialEq)]
pub struct Comment {
    pub comment: String,
}

#[derive(Debug, Default)]
struct ParserState {
    pub lineno: usize,
    pub comments: Vec<String>
}

fn whole_line_comment(input: &str) -> IResult<&str, &str> {
    preceded(pair(space0, tag("#")), recognize(not_line_ending))(input)
}

fn comments(input: &str) -> IResult<&str, ParserState> {
    fold_many0(
        whole_line_comment,
        ParserState::default,
        |mut acc: ParserState, item| {
            acc.lineno += 1;
            acc.comments.push(item.to_string());
            acc
        },
    )(input)
}

fn parse(input: &str) -> IResult<&str, ParserState> {
    let (remaining_input, matched) = comments(input)?;

    Ok((
        remaining_input,
        matched
    ))
}

#[test]
fn test_parser() {
    let contents = std::fs::read_to_string("tests/data/python_example.py").unwrap();
    let state = comments(&contents);
    dbg!(state);
}
