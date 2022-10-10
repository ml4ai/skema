pub mod ast;
use ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mo, Mrow, Msub},
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    combinator::map,
    multi::many0,
    sequence::{delimited, pair, preceded},
    IResult,
};
use nom_locate::{position, LocatedSpan};

use std::collections::HashMap;

type Span<'a> = LocatedSpan<&'a str>;

fn mi(input: Span) -> IResult<Span, MathExpression> {
    let (s, element) = delimited(tag("<mi>"), take_until("</mi>"), tag("</mi>"))(input)?;
    //let element = element.to_string();
    Ok((s, Mi(&element)))
}

fn mo(input: Span) -> IResult<Span, MathExpression> {
    let (s, element) = delimited(tag("<mo>"), take_until("</mo>"), tag("</mo>"))(input)?;
    //let element = element.to_string();
    Ok((s, Mo(&element)))
}

fn mrow(input: Span) -> IResult<Span, MathExpression> {
    let (s, elements) = delimited(tag("<mrow>"), many0(math_expression), tag("</mrow>"))(input)?;
    Ok((s, Mrow(elements)))
}

fn math_expression(input: Span) -> IResult<Span, MathExpression> {
    alt((mi, mo, mrow))(input)
}

fn mfrac(input: Span) -> IResult<Span, MathExpression> {
    let (s, result) = map(
        delimited(
            tag("<mfrac>"),
            pair(math_expression, math_expression),
            tag("</mfrac>"),
        ),
        |(numerator, denominator)| Mfrac(Box::new(numerator), Box::new(denominator)),
    )(input)?;
    Ok((s, result))
}

fn math(input: Span) -> IResult<Span, Math> {
    let (s, elements) = delimited(tag("<math>"), many0(math_expression), tag("</math>"))(input)?;
    Ok((s, Math { content: elements }))
}

fn parse(input: &str) -> IResult<Span, Math> {
    let span = Span::new(input);
    let (remaining, math) = math(span)?;
    Ok((remaining, math))
}

#[test]
fn test_mi() {
    assert_eq!(mi(Span::new("<mi>x</mi>")).unwrap().1, Mi("x"));
}

#[test]
fn test_mo() {
    assert_eq!(mo(Span::new("<mo>=</mo>")).unwrap().1, Mo("="));
}

#[test]
fn test_mrow() {
    assert_eq!(
        mrow(Span::new("<mrow><mo>-</mo><mi>b</mi></mrow>"))
            .unwrap()
            .1,
        Mrow(vec![Mo("-"), Mi("b")])
    )
}

#[test]
fn test_math_expression() {
    assert_eq!(
        math_expression(Span::new("<mrow><mo>-</mo><mi>b</mi></mrow>"))
            .unwrap()
            .1,
        Mrow(vec![Mo("-"), Mi("b")])
    )
}

#[test]
fn test_parser() {
    assert_eq!(
        parse("<math><mrow><mo>-</mo><mi>b</mi></mrow></math>")
            .unwrap()
            .1,
        Math {
            content: vec![Mrow(vec![Mo("-"), Mi("b")])]
        }
    )
}
