use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mn, Mo, Mrow, Msqrt, Msub, Msup},
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, multispace0},
    combinator::map,
    multi::many0,
    sequence::{delimited, pair, tuple},
};
use nom_locate::LocatedSpan;

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Eq)]
pub struct ParseError<'a> {
    span: Span<'a>,
    message: String,
}

/// We implement the ParseError trait here to support the Span type.
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
/// and trailing whitespace, returning the output of `inner`.
fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(Span<'a>) -> IResult<O>
where
    F: FnMut(Span<'a>) -> IResult<O>,
{
    delimited(multispace0, inner, multispace0)
}

///XML-style comments
pub fn comment(input: Span) -> IResult<Span> {
    delimited(tag("<!--"), take_until("-->"), tag("-->"))(input)
}

macro_rules! stag {
    ($tag:expr) => {{
        tuple((tag("<"), tag($tag), many0(alpha1), tag(">")))
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
        tag_parser!($tag, take_until(tag_end))
    }};
}

/// A macro to help build parsers for MathML elements with 1 argument.
macro_rules! elem1 {
    ($tag:expr, $t:ident) => {{
        map(tag_parser!($tag, math_expression), |x| $t(Box::new(x)))
    }};
}

/// A macro to help build parsers for MathML elements with 2 arguments.
macro_rules! elem2 {
    ($tag:expr, $t:ident) => {{
        map(
            tag_parser!($tag, pair(math_expression, math_expression)),
            |(x, y)| $t(Box::new(x), Box::new(y)),
        )
    }};
}

/// A macro to help build parsers for MathML elements with zero or more arguments.
macro_rules! elem_many0 {
    ($tag:expr) => {{
        tag_parser!($tag, many0(math_expression))
    }};
}

/// Identifiers
fn mi(input: Span) -> IResult<MathExpression> {
    let (s, element) = elem0!("mi")(input)?;
    Ok((s, Mi(&element)))
}

fn mn(input: Span) -> IResult<MathExpression> {
    let (s, element) = elem0!("mn")(input)?;
    Ok((s, Mn(&element)))
}

fn mo(input: Span) -> IResult<MathExpression> {
    let (s, element) = elem0!("mo")(input)?;
    Ok((s, Mo(&element)))
}

fn mrow(input: Span) -> IResult<MathExpression> {
    let (s, elements) = elem_many0!("mrow")(input)?;
    Ok((s, Mrow(elements)))
}

fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = elem2!("mfrac", Mfrac)(input)?;
    Ok((s, frac))
}

fn msup(input: Span) -> IResult<MathExpression> {
    let (s, expression) = elem2!("msup", Msup)(input)?;
    Ok((s, expression))
}

fn msub(input: Span) -> IResult<MathExpression> {
    let (s, expression) = elem2!("msub", Msub)(input)?;
    Ok((s, expression))
}

fn msqrt(input: Span) -> IResult<MathExpression> {
    let (s, expression) = elem1!("msqrt", Msqrt)(input)?;
    Ok((s, expression))
}

fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((mi, mo, mn, msup, msub, msqrt, mfrac, mrow)))(input)
}

fn math(input: Span) -> IResult<Math> {
    let (s, elements) = elem_many0!("math")(input)?;
    Ok((s, Math { content: elements }))
}

pub fn parse(input: &str) -> IResult<Math> {
    let span = Span::new(input);
    let (remaining, math) = math(span)?;
    Ok((remaining, math))
}

/// A generic helper function for testing individual parsers
#[cfg(test)]
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
