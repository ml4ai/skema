use crate::{
    ast::{Derivative, MathExpression, Mi, Operator},
    parsing::{attribute, lparen, mi, mo, mrow, operator, rparen, ws, IResult, ParseError, Span},
    pratt_parsing::MathExpressionTree,
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
    let (s, (Mi(x), left, Mi(y), right)) = tuple((mi, mo, mi, mo))(input)?;
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

fn d(input: Span) -> IResult<()> {
    let (s, Mi(x)) = mi(input)?;
    if let "d" = x.as_ref() {
        Ok((s, ()))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify Mi('d')".to_string(),
            input,
        )))
    }
}

fn ci_func(input: Span) -> IResult<Ci> {
    let (s, x) = mi(input)?;
    Ok((s, Ci::new(Some(Type::Function), MathExpression::Mi(x))))
}

fn first_order_derivative_leibniz_notation(input: Span) -> IResult<(Derivative, Ci)> {
    let (s, _) = tuple((ws(stag!("mfrac")), ws(stag!("mrow")), ws(d)))(input)?;
    let (s, func) = ws(alt((univariate_func, ci_func)))(s)?;
    let (s, _) = tuple((
        ws(etag!("mrow")),
        ws(stag!("mrow")),
        ws(d),
        ws(mi),
        ws(etag!("mrow")),
        ws(etag!("mfrac")),
    ))(s)?;
    Ok((s, (Derivative::new(1, 1), func)))
}

fn ode(input: Span) -> IResult<MathExpressionTree> {
    // Recognize LHS derivative
    let (s, (derivative, ci)) = first_order_derivative_leibniz_notation(input)?;

    // Recognize equals sign
    let (s, _) = tuple((stag!("mo"), tag("="), etag!("mo")))(input)?;

    todo!()
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
    );

    test_parser(
        "<mfrac>
        <mrow><mi>d</mi><mi>S</mi></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>",
        first_order_derivative_leibniz_notation,
        (
            Derivative::new(1, 1),
            Ci::new(
                Some(Type::Function),
                MathExpression::Mi(Mi("S".to_string())),
            ),
        ),
    );

    // Test derivative with explicit time dependence 
    test_parser(
        "<mfrac>
        <mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>",
        first_order_derivative_leibniz_notation,
        (
            Derivative::new(1, 1),
            Ci::new(
                Some(Type::Function),
                MathExpression::Mi(Mi("S".to_string())),
            ),
        ),
    );
}
