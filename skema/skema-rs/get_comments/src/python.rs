use nom::{
    bytes::complete::{tag, is_not},
    sequence::preceded,
    character::complete::anychar,
    IResult,
};

#[derive(Debug, PartialEq)]
pub struct Comment {
    pub comment: String,
}

fn comment(input: &str) -> IResult<&str, Comment> {
    let (remaining_input, matched) = preceded(tag("#"), is_not("\n\r"))(input)?;
    Ok((remaining_input, Comment {comment: matched.to_string()}))
}

pub fn test_parser() {
    assert_eq!(comment("# this is a comment"), Ok(("", Comment { comment: " this is a comment".to_string()})));
}
