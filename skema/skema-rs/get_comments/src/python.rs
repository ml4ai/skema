use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take_until, take_while},
    character::complete::{anychar, none_of, not_line_ending, space0},
    combinator::opt,
    sequence::{pair, preceded, tuple},
    IResult,
};

#[derive(Debug, PartialEq)]
pub struct Comment {
    pub comment: String,
}

fn parse(input: &str) -> IResult<&str, Comment> {
    let (remaining_input, (_, _, matched)) =
        tuple((opt(space0), tag("#"), not_line_ending))(input)?;
    Ok((
        remaining_input,
        Comment {
            comment: matched.to_string(),
        },
    ))
}

pub fn test_parser() {
    //let contents = std::fs::read_to_string("../../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.py").unwrap();
    //assert_eq!(parse(&contents), Ok(("", Comment { comment: " this is a comment".to_string()})));
    assert_eq!(
        parse("   # This is a comment\n not this"),
        Ok((
            "\n not this",
            Comment {
                comment: " This is a comment".to_string()
            }
        ))
    );
}
