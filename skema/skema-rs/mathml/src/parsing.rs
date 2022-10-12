use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mn, Mo, Mrow, Msqrt, Msub, Msup},
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::multispace0,
    combinator::{map, value},
    multi::many0,
    sequence::{delimited, pair, preceded},
};
use nom_locate::LocatedSpan;

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq)]
pub struct ParseError<'a> {
    span: Span<'a>,
    message: String,
}

impl<'a> ParseError<'a> {
    pub fn new(message: String, span: Span<'a>) -> Self {
        Self { message, span }
    }

    pub fn span(&self) -> &Span {
        &self.span
    }

    pub fn line(&self) -> u32 {
        self.span().location_line()
    }

    pub fn offset(&self) -> usize {
        self.span().location_offset()
    }
}

impl<'a> nom::error::ParseError<Span<'a>> for ParseError<'a> {
    fn from_error_kind(input: Span<'a>, kind: nom::error::ErrorKind) -> Self {
        Self::new(format!("Parse error {:?}", kind), input)
    }

    fn append(_input: Span<'a>, _kind: nom::error::ErrorKind, other: Self) -> Self {
        other
    }

    fn from_char(input: Span<'a>, c: char) -> Self {
        Self::new(format!("Unexpected character '{}'", c), input)
    }
}

impl<'a> nom::error::ContextError<Span<'a>> for ParseError<'a> {
    fn add_context(input: Span<'a>, ctx: &'static str, other: Self) -> Self {
        let message = format!("{}: {}", ctx, other.message);
        ParseError::new(message, input)
    }
}

pub type IResult<'a, O> = nom::IResult<Span<'a>, O, ParseError<'a>>;

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading
/// and trailing whitespace and comments, returning the output of `inner`.
fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(Span<'a>) -> IResult<O>
where
    F: FnMut(Span<'a>) -> IResult<O>,
{
    delimited(multispace0, inner, multispace0)
}

pub fn comment(input: Span) -> IResult<Span> {
    ws(delimited(tag("<!--"), take_until("-->"), tag("-->")))(input)
}

fn mi(input: Span) -> IResult<MathExpression> {
    let (s, element) = ws(delimited(tag("<mi>"), take_until("</mi>"), tag("</mi>")))(input)?;
    Ok((s, Mi(&element)))
}

fn mn(input: Span) -> IResult<MathExpression> {
    let (s, element) = ws(delimited(tag("<mn>"), take_until("</mn>"), tag("</mn>")))(input)?;
    Ok((s, Mn(&element)))
}

fn mo(input: Span) -> IResult<MathExpression> {
    let (s, element) = ws(delimited(tag("<mo>"), take_until("</mo>"), tag("</mo>")))(input)?;
    Ok((s, Mo(&element)))
}

fn mrow(input: Span) -> IResult<MathExpression> {
    let (s, elements) = ws(delimited(
        tag("<mrow>"),
        many0(math_expression),
        tag("</mrow>"),
    ))(input)?;
    Ok((s, Mrow(elements)))
}

fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = map(
        ws(delimited(
            tag("<mfrac>"),
            pair(math_expression, math_expression),
            tag("</mfrac>"),
        )),
        |(numerator, denominator)| Mfrac(Box::new(numerator), Box::new(denominator)),
    )(input)?;
    Ok((s, frac))
}

fn msup(input: Span) -> IResult<MathExpression> {
    let (s, expression) = map(
        ws(delimited(
            tag("<msup>"),
            pair(math_expression, math_expression),
            tag("</msup>"),
        )),
        |(base, superscript)| Msup(Box::new(base), Box::new(superscript)),
    )(input)?;
    Ok((s, expression))
}

fn msub(input: Span) -> IResult<MathExpression> {
    let (s, expression) = map(
        ws(delimited(
            tag("<msub>"),
            pair(math_expression, math_expression),
            tag("</msub>"),
        )),
        |(base, subscript)| Msub(Box::new(base), Box::new(subscript)),
    )(input)?;
    Ok((s, expression))
}

fn msqrt(input: Span) -> IResult<MathExpression> {
    let (s, expression) = map(
        ws(delimited(tag("<msqrt>"), math_expression, tag("</msqrt>"))),
        |contents| Msqrt(Box::new(contents)),
    )(input)?;
    Ok((s, expression))
}

fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((mi, mo, mn, msup, msub, msqrt, mfrac, mrow)))(input)
}

fn math(input: Span) -> IResult<Math> {
    let (s, elements) = ws(delimited(
        tag("<math>"),
        many0(math_expression),
        tag("</math>"),
    ))(input)?;
    Ok((s, Math { content: elements }))
}

pub fn parse(input: &str) -> IResult<Math> {
    let span = Span::new(input);
    let (remaining, math) = math(span)?;
    Ok((remaining, math))
}

/// A generic helper function for testing individual parsers
fn test_parser<'a, P, O>(input: &'a str, mut parser: P, output: O)
where
    P: FnMut(Span<'a>) -> IResult<'a, O>,
    O: std::cmp::PartialEq + std::fmt::Debug,
{
    assert_eq!(parser(Span::new(input)).unwrap().1, output);
}

#[test]
fn test_mi() {
    test_parser("<mi>x</mi>", mi, Mi("x"))
}

#[test]
fn test_mo() {
    test_parser("<mo>=</mo>", mo, Mo("="))
}

#[test]
fn test_mn() {
    test_parser("<mn>1</mn>", mn, Mn("1"));
}

#[test]
fn test_mrow() {
    test_parser(
        "<mrow><mo>-</mo><mi>b</mi></mrow>",
        mrow,
        Mrow(vec![Mo("-"), Mi("b")]),
    )
}

#[test]
fn test_mfrac() {
    let frac = mfrac(Span::new("<mfrac><mn>1</mn><mn>2</mn></mfrac>"))
        .unwrap()
        .1;
    assert_eq!(frac, Mfrac(Box::new(Mn("1")), Box::new(Mn("2"))),)
}

#[test]
fn test_math_expression() {
    test_parser(
        "<mrow><mo>-</mo><mi>b</mi></mrow>",
        math_expression,
        Mrow(vec![Mo("-"), Mi("b")]),
    )
}
