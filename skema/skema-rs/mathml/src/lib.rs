//#![cfg(feature = "alloc")]

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{anychar, char, multispace0, none_of},
    combinator::{map, map_opt, map_res, value, verify},
    error::ParseError,
    multi::{fold_many0, separated_list0},
    number::complete::double,
    sequence::{delimited, preceded, separated_pair},
    IResult, Parser,
};
use nom::{multi::many0, sequence::pair};
use nom_locate::{position, LocatedSpan};

use std::collections::HashMap;

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Clone)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Str(String),
    Num(f64),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum MathExpression {
    Mi(String),
    Mo(String),
    Mrow(Vec<MathExpression>),
    Mfrac(MathExpression, MathExpression),
    Msub(MathExpression, MathExpression),
}

fn mi(input: Span) -> IResult<Span, MathExpression> {
    let (s, element) = delimited(tag("<mi>"), take_until("</mi>"), tag("</mi>"))(input)?;
    let element = element.to_string();
    Ok((s, MathExpression::Mi(element)))
}

#[test]
fn test_mi() {
    assert_eq!(
        mi(Span::new("<mi>x</mi>")).unwrap().1,
        MathExpression::Mi("x".to_string())
    );
    //test_equality(mi, "<mi>x</mi>", MathExpression::Mi("x".to_string()));
}

fn mo(input: Span) -> IResult<Span, MathExpression> {
    let (s, element) = delimited(tag("<mo>"), take_until("</mo>"), tag("</mo>"))(input)?;
    let element = element.to_string();
    Ok((s, MathExpression::Mo(element)))
}

fn math_expression(input: Span) -> IResult<Span, MathExpression> {
    alt((mi, mo))(input)
}

#[test]
fn test_mo() {
    assert_eq!(
        mo(Span::new("<mo>=</mo>")).unwrap().1,
        MathExpression::Mo("=".to_string())
    );
}

fn mrow(input: Span) -> IResult<Span, MathExpression> {
    let (s, elements) = delimited(tag("<mrow>"), many0(math_expression), tag("</mrow>"))(input)?;
    Ok((s, MathExpression::Mrow(elements)))
}

fn mfrac(input: Span) -> IResult<Span, MathExpression> {
    let (s, (numerator, denominator)) = delimited(
        tag("<mfrac>"),
        pair(math_expression, math_expression),
        tag("</mfrac>"),
    )(input)?;
    Ok((s, MathExpression::Mfrac(numerator, denominator)))
}

#[derive(Debug, PartialEq)]
struct Math {
    content: String,
}

fn math(input: Span) -> IResult<Span, Math> {
    let (s, element) = delimited(tag("<math>"), take_until("</math>"), tag("</math>"))(input)?;
    let element = element.to_string();
    Ok((s, Math { content: element }))
}

#[test]
fn test_math() {
    assert_eq!(
        math(Span::new("<math>Content</math>")).unwrap().1,
        Math {
            content: "Content".to_string()
        }
    );
}
