// The line below is required for recursive enum definitions to work.
#![cfg(feature = "alloc")]

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    multi::many0,
    sequence::{delimited, pair},
    IResult,
};
use nom_locate::{position, LocatedSpan};

type Span<'a> = LocatedSpan<&'a str>;

fn test_equality<P, O>(input: &str, parser: P, output: O) {
    assert_eq!(P(Span::new("<mo>=</mo>")).unwrap().1, O);
}
#[derive(Debug, PartialEq)]
enum MathExpression {
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
    //assert_eq!(
    //mi(Span::new("<mi>x</mi>")).unwrap().1,
    //MathExpression::Mi("x".to_string())
    //);
    test_equality(mi, "<mi>x</mi>", MathExpression::Mi("x".to_string()));
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
