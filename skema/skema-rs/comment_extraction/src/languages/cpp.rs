use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_until},
    multi::fold_many0,
    sequence::delimited,
    IResult,
};
use nom_locate::LocatedSpan;
use serde::{Deserialize, Serialize};

/// Parse C/C++ code and output the comments along with their line numbers.  

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Comments {
    pub comments: Vec<(u32, String)>,
}

// find C and C++ style comments
fn parse_comment(input: Span) -> IResult<Span, Span> {
    alt((
        delimited(tag("/*"), take_until("*/"), tag("*/")), // C
        delimited(tag("//"), take_until("\n"), tag("\n")), // C++
    ))(input)
}

// Move to next input element, bypassing quoted elements if found
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

// Return a vector of comments and their line numbes
fn parse(s: Span) -> IResult<Span, Vec<(u32, String)>> {
    fold_many0(
        alt((parse_comment, parse_next)),
        Vec::new,
        |mut acc: Vec<(u32, String)>, span| {
            if span.fragment().len() > 0 {
                acc.push((span.location_line(), span.fragment().to_string()));
            } else {
            }
            acc
        },
    )(s)
}

// parse C and C++ style comments from string input
fn process_string(s: &str) -> Comments {
    match parse(LocatedSpan::new(s)) {
        Ok((_, vec)) => Comments { comments: vec },
        Err(e) => {
            println!("Error: {e:?}");
            Comments {
                comments: Vec::new(),
            }
        }
    }
}

// Parse C and C++ style comments from file input
pub fn get_comments(file_path: &str) -> Comments {
    let string = std::fs::read_to_string(file_path);
    match string {
        Ok(s) => process_string(&s),
        Err(e) => {
            println!("Error: {e:?}");
            Comments {
                comments: Vec::new(),
            }
        }
    }
}
