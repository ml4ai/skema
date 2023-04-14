use crate::comments::{Comment, Comments};
use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_until},
    multi::fold_many0,
    sequence::delimited,
    IResult,
};
use nom_locate::LocatedSpan;

/// Parse C/C++ code and output the comments along with their line numbers.  

type Span<'a> = LocatedSpan<&'a str>;

/// find C and C++ style comments
fn parse_comment(input: Span) -> IResult<Span, Span> {
    alt((
        delimited(tag("/*"), take_until("*/"), tag("*/")), // C
        delimited(tag("//"), take_until("\n"), tag("\n")), // C++
    ))(input)
}

/// Move to next input element, bypassing quoted elements if found
fn parse_next(input: Span) -> IResult<Span, Span> {
    match alt((
        delimited(tag("\""), take_until("\""), tag("\"")), // quoted
        take(1usize),                                      // Move to next element
    ))(input)
    {
        Ok((i, _)) => Ok((i, "".into())),
        Err(e) => Err(e),
    }
}

/// Return a vector of comments and their line numbes
fn parse(s: Span) -> IResult<Span, Vec<Comment>> {
    fold_many0(
        alt((parse_comment, parse_next)),
        Vec::new,
        |mut acc: Vec<Comment>, span| {
            if span.fragment().len() > 0 {
                acc.push(Comment {
                    line_number: span.location_line().try_into().unwrap(),
                    contents: span.fragment().to_string(),
                });
            } else {
            }
            acc
        },
    )(s)
}

/// Parse C and C++ style comments from string input
fn process_string(s: &str) -> Comments {
    let (_, vec) = parse(LocatedSpan::new(s)).unwrap();
    Comments {
        comments: vec,
        ..Default::default()
    }
}

// Parse C and C++ style comments from file input
pub fn get_comments(file_path: &str) -> Comments {
    let string = std::fs::read_to_string(file_path).unwrap();
    process_string(&string)
}
