use nom::{
    bytes::complete::{tag, take_until},
    sequence::delimited,
    IResult,
};
use nom_locate::{position, LocatedSpan};

type Span<'a> = LocatedSpan<&'a str>;

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
