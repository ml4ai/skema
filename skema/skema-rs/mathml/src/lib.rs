use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    combinator::value,
    multi::many0,
    sequence::delimited,
    IResult,
};
use nom_locate::{position, LocatedSpan};

type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq)]
struct Mi(String);

#[derive(Debug, PartialEq)]
struct Mo(String);

#[derive(Debug, PartialEq)]
enum MathExpression {
    Mi(Mi),
    Mo(Mo),
}

fn mi(input: Span) -> IResult<Span, Mi> {
    let (s, element) = delimited(tag("<mi>"), take_until("</mi>"), tag("</mi>"))(input)?;
    let element = element.to_string();
    Ok((s, Mi(element)))
}

#[test]
fn test_mi() {
    assert_eq!(mi(Span::new("<mi>x</mi>")).unwrap().1, Mi("x".to_string()));
}

fn mo(input: Span) -> IResult<Span, Mo> {
    let (s, element) = delimited(tag("<mo>"), take_until("</mo>"), tag("</mo>"))(input)?;
    let element = element.to_string();
    Ok((s, Mo(element)))
}

#[test]
fn test_mo() {
    assert_eq!(mo(Span::new("<mo>=</mo>")).unwrap().1, Mo("=".to_string()));
}

#[derive(Debug, PartialEq)]
struct Mrow(Vec<MathExpression>);

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
