use crate::ast::{
    operator::Operator,
    Math, MathExpression,
    MathExpression::{
        Mfrac, Mn, Mo, MoLine, Mover, Mspace, Msqrt, Mstyle, Msub, Msubsup, Msup, Mtext, Munder,
    },
    Mi, Mrow,
};

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alphanumeric1, multispace0, not_line_ending},
    combinator::{cut, map, map_parser, opt, peek, recognize, value},
    multi::many0,
    sequence::{delimited, pair, preceded, separated_pair, tuple},
};
use nom::{character::complete::char as nom_char, error::context};

use nom_locate::LocatedSpan;
use std::str::FromStr;

pub type Span<'a> = LocatedSpan<&'a str>;

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

    pub fn append_message(&mut self, msg: &str) {
        self.message.push_str(&format!("\nERROR: {}", msg));
    }
}

/// Further trait implementation for Span
impl<'a> nom::error::ParseError<Span<'a>> for ParseError<'a> {
    fn from_error_kind(input: Span<'a>, kind: nom::error::ErrorKind) -> Self {
        Self::new(format!("Parse error {kind:?}"), input)
    }

    fn append(_input: Span<'a>, _kind: nom::error::ErrorKind, other: Self) -> Self {
        other
    }

    fn from_char(input: Span<'a>, c: char) -> Self {
        Self::new(format!("Unexpected character '{c}'"), input)
    }
}

/// Implementing ContextError to support Span
impl<'a> nom::error::ContextError<Span<'a>> for ParseError<'a> {
    fn add_context(input: Span<'a>, ctx: &'static str, other: Self) -> Self {
        let message = format!("{}: {}", ctx, other.message);
        ParseError::new(message, input)
    }
}

/// Redefine IResult, filling in the first generic type parameter with Span, for increased brevity.
pub type IResult<'a, O> = nom::IResult<Span<'a>, O, ParseError<'a>>;

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading
/// and trailing whitespace, returning the output of `inner`.
pub fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(Span<'a>) -> IResult<O>
where
    F: FnMut(Span<'a>) -> IResult<O>,
{
    delimited(multispace0, inner, multispace0)
}

///Quoted string
pub fn quoted_string(input: Span) -> IResult<Span> {
    delimited(tag("\""), take_until("\""), tag("\""))(input)
}

pub fn attribute(input: Span) -> IResult<(&str, &str)> {
    let (s, (key, value)) = ws(separated_pair(alphanumeric1, ws(tag("=")), quoted_string))(input)?;
    Ok((s, (&key, &value)))
}

#[macro_export]
macro_rules! stag {
    ($tag:expr) => {{
        ws(tuple((tag("<"), tag($tag), many0(attribute), tag(">"))))
    }};
}

#[macro_export]
macro_rules! etag {
    ($tag:expr) => {{
        ws(delimited(tag("</"), tag($tag), tag(">")))
    }};
}

/// A macro to help build tag parsers
#[macro_export]
macro_rules! tag_parser {
    ($tag:expr, $parser:expr) => {{
        ws(delimited(stag!($tag), $parser, etag!($tag)))
    }};
}

/// A macro to help build parsers for simple MathML elements (i.e., without further nesting).
#[macro_export]
macro_rules! elem0 {
    ($tag:expr) => {{
        let tag_end = concat!("</", $tag, ">");
        tag_parser!($tag, ws(take_until(tag_end)))
    }};
}

/// A macro to help build parsers for MathML elements with 1 argument.
#[macro_export]
macro_rules! elem1 {
    ($tag:expr, $t:ident) => {{
        map(tag_parser!($tag, math_expression), |x| $t(Box::new(x)))
    }};
}

/// A macro to help build parsers for MathML elements with 2 arguments.
#[macro_export]
macro_rules! elem2 {
    ($tag:expr, $t:ident) => {{
        map(
            tag_parser!($tag, pair(math_expression, math_expression)),
            |(x, y)| $t(Box::new(x), Box::new(y)),
        )
    }};
}

/// A macro to help build parsers for MathML elements with 3 arguments.
#[macro_export]
macro_rules! elem3 {
    ($tag:expr, $t:ident) => {{
        map(
            tag_parser!(
                $tag,
                tuple((math_expression, math_expression, math_expression))
            ),
            |(x, y, z)| $t(Box::new(x), Box::new(y), Box::new(z)),
        )
    }};
}

/// A macro to help build parsers for MathML elements with zero or more arguments.
#[macro_export]
macro_rules! elem_many0 {
    ($tag:expr) => {{
        tag_parser!($tag, many0(math_expression))
    }};
}

/// Identifiers
pub fn mi(input: Span) -> IResult<Mi> {
    let (s, element) = elem0!("mi")(input)?;
    Ok((s, Mi(element.trim().to_string())))
}

/// Numbers
pub fn mn(input: Span) -> IResult<MathExpression> {
    let (s, element) = elem0!("mn")(input)?;
    Ok((s, Mn(element.trim().to_string())))
}

pub fn add(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Add, ws(tag("+")))(input)?;
    Ok((s, op))
}

pub fn subtract(input: Span) -> IResult<Operator> {
    let (s, op) = value(
        Operator::Subtract,
        alt((tag("-"), tag("−"), tag("&#x2212;"))),
    )(input)?;
    Ok((s, op))
}

pub fn equals(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Equals, ws(tag("=")))(input)?;
    Ok((s, op))
}

pub fn lparen(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Lparen, alt((ws(tag("(")), ws(tag("[")))))(input)?;
    Ok((s, op))
}

pub fn rparen(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Rparen, alt((ws(tag(")")), ws(tag("]")))))(input)?;
    Ok((s, op))
}

pub fn comma(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Comma, ws(tag(",")))(input)?;
    Ok((s, op))
}

fn operator_other(input: Span) -> IResult<Operator> {
    let (s, consumed) = ws(recognize(not_line_ending))(input)?;
    let op = Operator::Other(consumed.to_string());
    Ok((s, op))
}

pub fn operator(input: Span) -> IResult<Operator> {
    let (s, op) = alt((add, subtract, equals, lparen, rparen, operator_other))(input)?;
    Ok((s, op))
}

#[test]
fn test_operator() {
    let (_, op) = operator(Span::new("-")).unwrap();
    assert_eq!(op, Operator::Subtract);
}

/// Operators
pub fn mo(input: Span) -> IResult<MathExpression> {
    let (s, op) = ws(delimited(
        stag!("mo"),
        map_parser(recognize(take_until("</mo>")), operator),
        etag!("mo"),
    ))(input)?;
    Ok((s, Mo(op)))
}

/// Rows
pub fn mrow(input: Span) -> IResult<Mrow> {
    let (s, elements) = ws(delimited(
        stag!("mrow"),
        many0(math_expression),
        etag!("mrow"),
    ))(input)?;
    Ok((s, Mrow(elements)))
}

/// Fractions
pub fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = elem2!("mfrac", Mfrac)(input)?;
    Ok((s, frac))
}

/// Superscripts
pub fn msup(input: Span) -> IResult<MathExpression> {
    let (s, expression) = elem2!("msup", Msup)(input)?;
    Ok((s, expression))
}

/// Subscripts
pub fn msub(input: Span) -> IResult<MathExpression> {
    let (s, expression) = elem2!("msub", Msub)(input)?;
    Ok((s, expression))
}

/// Square roots
pub fn msqrt(input: Span) -> IResult<MathExpression> {
    let (s, expression) = elem1!("msqrt", Msqrt)(input)?;
    Ok((s, expression))
}

// Underscripts
fn munder(input: Span) -> IResult<MathExpression> {
    let (s, underscript) = elem2!("munder", Munder)(input)?;
    Ok((s, underscript))
}

// Overscripts
pub fn mover(input: Span) -> IResult<MathExpression> {
    let (s, overscript) = elem2!("mover", Mover)(input)?;
    Ok((s, overscript))
}

// Subscript-superscript Pair
pub fn msubsup(input: Span) -> IResult<MathExpression> {
    let (s, subsup) = elem3!("msubsup", Msubsup)(input)?;
    Ok((s, subsup))
}

//Text
fn mtext(input: Span) -> IResult<MathExpression> {
    let (s, element) = elem0!("mtext")(input)?;
    Ok((s, Mtext(element.trim().to_string())))
}

//mstyle
fn mstyle(input: Span) -> IResult<MathExpression> {
    let (s, elements) = elem_many0!("mstyle")(input)?;
    Ok((s, Mstyle(elements)))
}

// function for xml
pub fn xml_declaration(input: Span) -> IResult<()> {
    let (s, _contents) = ws(delimited(tag("<?"), take_until("?>"), tag("?>")))(input)?;
    Ok((s, ()))
}

//mspace
fn mspace(input: Span) -> IResult<MathExpression> {
    let (s, element) = ws(delimited(tag("<mspace"), take_until("/>"), tag("/>")))(input)?;
    Ok((s, Mspace(element.to_string())))
}

// Some xml have <mo .../>
fn mo_line(input: Span) -> IResult<MathExpression> {
    let (s, element) = ws(delimited(tag("<mo"), take_until("/>"), tag("/>")))(input)?;
    Ok((s, MoLine(element.to_string())))
}

/// Math expressions
pub fn math_expression(input: Span) -> IResult<MathExpression> {
    // Lookahead for next open tag
    let tag_name = peek(delimited(
        multispace0,
        delimited(
            nom_char('<'),
            take_until(">"),
            alt((tag(">"), tag("/>"))), // Matches both self-closing and regular tags
        ),
        multispace0,
    ))(input)
    .map(|(_, tag_name)| {
        let tag_name_string = tag_name.to_string();
        let mut split_tag_name = tag_name_string.split_whitespace(); // We only want the tag name and no attributes
        split_tag_name.next().unwrap().to_string()
    })?;

    if tag_name.contains('/') {
        // Found a closing tag! This means no more math expressions, but is not wrong.
        // We want the parent combinator to still continue to try and parse the remaining input
        mn(input)
    } else {
        match tag_name.as_str() {
            "mi" => context("FAILED TO PARSE <mi>", cut(ws(map(mi, MathExpression::Mi))))(input),
            "mn" => context("FAILED TO PARSE <mn>", cut(ws(mn)))(input),
            "msup" => context("FAILED TO PARSE <msup>", cut(ws(msup)))(input),
            "msub" => context("FAILED TO PARSE <msub>", cut(ws(msub)))(input),
            "msqrt" => context("FAILED TO PARSE <msqrt>", cut(ws(msqrt)))(input),
            "mfrac" => context("FAILED TO PARSE <mfrac>", cut(ws(mfrac)))(input),
            "mrow" => context(
                "FAILED TO PARSE <mrow>",
                cut(map(mrow, MathExpression::Mrow)),
            )(input),
            "munder" => context("FAILED TO PARSE <munder>", cut(ws(munder)))(input),
            "mover" => context("FAILED TO PARSE <mover>", cut(ws(mover)))(input),
            "msubsup" => context("FAILED TO PARSE <msubsup>", cut(ws(msubsup)))(input),
            "mtext" => context("FAILED TO PARSE <mtext>", cut(ws(mtext)))(input),
            "mstyle" => context("FAILED TO PARSE <mstyle>", cut(ws(mstyle)))(input),
            "mspace" => context("FAILED TO PARSE <mspace>", cut(ws(mspace)))(input),
            "mo" => context("FAILED TO PARSE <mo>", cut(ws(alt((mo, mo_line)))))(input),
            _ => {
                println!("Something went wrong. We grabbed a {} tag", tag_name);
                context("SOMETHING WENT WRONG. WE SHOULDN'T BE HERE.", cut(mn))(input)
            }
        }
    }
}

/// testing MathML documents
pub fn math(input: Span) -> IResult<Math> {
    let (s, elements) = preceded(opt(xml_declaration), elem_many0!("math"))(input)?;
    Ok((s, Math { content: elements }))
}

/// The `parse` function is part of the public API. It takes a string and returns a Math object.
pub fn parse(input: &str) -> IResult<Math> {
    let span = Span::new(input);
    let (remaining, math) = math(span)?;
    Ok((remaining, math))
}

impl FromStr for Math {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        math(s.into())
            .map(|(_, math)| math)
            .map_err(|err| err.to_string())
    }
}

pub trait FromFile<T: FromStr> {
    fn from_file(filepath: &str) -> T {
        let file_contents = std::fs::read_to_string(filepath)
            .unwrap_or_else(|_| panic!("{}", "Unable to read file {filepath}!"));
        file_contents
            .parse::<T>()
            .unwrap_or_else(|_| panic!("{}", "Unable to parse file {filepath}!"))
    }
}

/// A generic helper function for testing individual parsers.
#[cfg(test)]
pub fn test_parser<'a, P, O>(input: &'a str, mut parser: P, output: O)
where
    P: FnMut(Span<'a>) -> IResult<'a, O>,
    O: std::cmp::PartialEq + std::fmt::Debug,
{
    let (s, o) = parser(Span::new(input)).unwrap();
    assert_eq!(s.fragment(), &"");
    assert_eq!(o, output);
}

#[test]
fn test_mi() {
    test_parser("<mi k=\"v\" m1=\"n\">x</mi>", mi, Mi("x".to_string()))
}

#[test]
fn test_mo() {
    test_parser("<mo>=</mo>", mo, Mo(Operator::Equals));
    test_parser("<mo>+</mo>", mo, Mo(Operator::Add));
    test_parser("<mo>-</mo>", mo, Mo(Operator::Subtract));
}

#[test]
fn test_mn() {
    test_parser("<mn>1</mn>", mn, Mn("1".to_string()));
}

#[test]
fn test_mrow() {
    test_parser(
        "<mrow><mo>-</mo><mi>b</mi></mrow>",
        mrow,
        Mrow(vec![
            Mo(Operator::Subtract),
            MathExpression::Mi(Mi("b".to_string())),
        ]),
    );
}

#[test]
fn test_attribute() {
    test_parser("key=\"value\"", attribute, ("key", "value"))
}

#[test]
fn test_mfrac() {
    let frac = mfrac(Span::new("<mfrac><mn>1</mn><mn>2</mn></mfrac>"))
        .unwrap()
        .1;
    assert_eq!(
        frac,
        Mfrac(Box::new(Mn("1".to_string())), Box::new(Mn("2".to_string()))),
    )
}

#[test]
fn test_math_expression() {
    test_parser(
        "<mrow><mo>-</mo><mi>b</mi></mrow>",
        math_expression,
        MathExpression::Mrow(Mrow(vec![
            Mo(Operator::Subtract),
            MathExpression::Mi(Mi("b".to_string())),
        ])),
    )
}

#[test]
fn test_mover() {
    test_parser(
        "<mover><mi>x</mi><mo>¯</mo></mover>",
        mover,
        Mover(
            Box::new(MathExpression::Mi(Mi("x".to_string()))),
            Box::new(Mo(Operator::Other("¯".to_string()))),
        ),
    )
}

#[test]
fn test_mtext() {
    test_parser("<mtext>if</mtext>", mtext, Mtext("if".to_string()));
}

#[test]
fn test_mstyle() {
    test_parser(
        "<mstyle><mo>∑</mo><mi>I</mi></mstyle>",
        mstyle,
        Mstyle(vec![
            Mo(Operator::Other("∑".to_string())),
            MathExpression::Mi(Mi("I".to_string())),
        ]),
    )
}

#[test]
fn test_mspace() {
    test_parser(
        "<mspace width=\"1em\"/>",
        mspace,
        Mspace(" width=\"1em\"".to_string()),
    );
}

#[test]
fn test_moline() {
    test_parser(
        "<mo fence=\"true\" stretchy=\"true\" symmetric=\"true\"/>",
        mo_line,
        MoLine(" fence=\"true\" stretchy=\"true\" symmetric=\"true\"".to_string()),
    );
}

#[test]
fn test_math() {
    test_parser(
        "<math>
            <mrow>
                <mo>-</mo>
                <mi>b</mi>
            </mrow>
        </math>",
        math,
        Math {
            content: vec![MathExpression::Mrow(Mrow(vec![
                Mo(Operator::Subtract),
                MathExpression::Mi(Mi("b".to_string())),
            ]))],
        },
    )
}
#[test]
fn test_mathml_parser() {
    let eqn = std::fs::read_to_string("tests/test01.xml").unwrap();
    test_parser(
        &eqn,
        math,
        Math {
            content: vec![
                Munder(
                    Box::new(Mo(Operator::Other("sup".to_string()))),
                    Box::new(MathExpression::Mrow(Mrow(vec![
                        Mn("0".to_string()),
                        Mo(Operator::Other("≤".to_string())),
                        MathExpression::Mi(Mi("t".to_string())),
                        Mo(Operator::Other("≤".to_string())),
                        Msub(
                            Box::new(MathExpression::Mi(Mi("T".to_string()))),
                            Box::new(Mn("0".to_string())),
                        ),
                    ]))),
                ),
                Mo(Operator::Other("‖".to_string())),
                Msup(
                    Box::new(MathExpression::Mrow(Mrow(vec![Mover(
                        Box::new(MathExpression::Mi(Mi("ρ".to_string()))),
                        Box::new(Mo(Operator::Other("~".to_string()))),
                    )]))),
                    Box::new(MathExpression::Mi(Mi("R".to_string()))),
                ),
                Msup(
                    Box::new(MathExpression::Mrow(Mrow(vec![Mover(
                        Box::new(MathExpression::Mi(Mi("x".to_string()))),
                        Box::new(Mo(Operator::Other("¯".to_string()))),
                    )]))),
                    Box::new(MathExpression::Mi(Mi("a".to_string()))),
                ),
                Msub(
                    Box::new(Mo(Operator::Other("‖".to_string()))),
                    Box::new(MathExpression::Mrow(Mrow(vec![
                        Msup(
                            Box::new(MathExpression::Mi(Mi("L".to_string()))),
                            Box::new(Mn("1".to_string())),
                        ),
                        Mo(Operator::Other("∩".to_string())),
                        Msup(
                            Box::new(MathExpression::Mi(Mi("L".to_string()))),
                            Box::new(MathExpression::Mi(Mi("∞".to_string()))),
                        ),
                    ]))),
                ),
                Mo(Operator::Other("≤".to_string())),
                MathExpression::Mi(Mi("C".to_string())),
            ],
        },
    )
}

// Exporting macros
pub(crate) use elem2;
pub(crate) use elem_many0;
pub(crate) use etag;
pub(crate) use stag;
pub(crate) use tag_parser;
