use crate::{
    ast::{MathExpression, Mi, Operator},
    parsing::{attribute, lparen, mi, mo, rparen, ws, IResult, ParseError, Span},
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alphanumeric1, multispace0, not_line_ending, one_of},
    combinator::{map, map_parser, opt, recognize, value},
    multi::many0,
    sequence::{delimited, pair, preceded, separated_pair, tuple},
};

use derive_new::new;

#[cfg(test)]
use crate::parsing::test_parser;

macro_rules! stag {
    ($tag:expr) => {{
        tuple((tag("<"), tag($tag), many0(attribute), tag(">")))
    }};
}

macro_rules! etag {
    ($tag:expr) => {{
        delimited(tag("</"), tag($tag), tag(">"))
    }};
}

/// A macro to help build tag parsers
macro_rules! tag_parser {
    ($tag:expr, $parser:expr) => {{
        ws(delimited(stag!($tag), $parser, etag!($tag)))
    }};
}

/// A macro to help build parsers for simple MathML elements (i.e., without further nesting).
macro_rules! elem0 {
    ($tag:expr) => {{
        let tag_end = concat!("</", $tag, ">");
        tag_parser!($tag, ws(take_until(tag_end)))
    }};
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
enum Type {
    Integer,
    Rational,
    Real,
    Complex,
    ComplexPolar,
    ComplexCartesian,
    Constant,
    Function,
    Vector,
    List,
    Set,
    Matrix,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
struct Ci {
    r#type: Option<Type>,
    content: MathExpression,
}

fn univariate_func(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), left, y, right)) = tuple((mi, mo, mi, mo))(input)?;
    if let (MathExpression::Mo(Operator::Lparen), MathExpression::Mo(Operator::Rparen)) =
        (left, right)
    {
        Ok((
            s,
            Ci::new(
                Some(Type::Function),
                MathExpression::Mi(Mi(x.trim().to_string())),
            ),
        ))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify univariate function".to_string(),
            input,
        )))
    }
}

#[test]
fn test_dsp() {
    test_parser(
        "<mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>",
        univariate_func,
        Ci::new(
            Some(Type::Function),
            MathExpression::Mi(Mi("S".to_string())),
        ),
    )

    //test_parser(
    //"<mfrac>
    //<mrow><mi>d</mi><mi>S</mi></mrow>
    //<mrow><mi>d</mi><mi>t</mi></mrow>
    //</mfrac>"

    //)
}
