//! This module contains parsers that perform some amount of preliminary domain-specific
//! interpretation of presentation MathML (e.g., globbing S(t) to an identifier S of type
//! 'function'). This is in contrast to the `generic_mathml.rs` module that contains parsers that
//! do not attempt to perform any interpretation but instead simply preserve the original MathML
//! document structure.

use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, Math, MathExpression, Mi, Mrow, Type,
    },
    parsers::generic_mathml::{
        add, attribute, elem_many0, equals, etag, lparen, mi, mn, msqrt, msub, msubsup, msup,
        rparen, stag, subtract, tag_parser, ws, xml_declaration, IResult, ParseError, Span,
    },
};

use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt},
    multi::{many0, many1},
    sequence::{delimited, pair, preceded, terminated, tuple},
};

/// Function to parse operators. This function differs from the one in parsers::generic_mathml by
/// disallowing operators besides +, -, =, (, and ).
pub fn operator(input: Span) -> IResult<Operator> {
    let (s, op) = ws(delimited(
        stag!("mo"),
        alt((add, subtract, equals, lparen, rparen)),
        etag!("mo"),
    ))(input)?;
    Ok((s, op))
}

fn parenthesized_identifier(input: Span) -> IResult<()> {
    let mo_lparen = delimited(stag!("mo"), lparen, etag!("mo"));
    let mo_rparen = delimited(stag!("mo"), rparen, etag!("mo"));
    let (s, _) = delimited(mo_lparen, mi, mo_rparen)(input)?;
    Ok((s, ()))
}

/// Parse content identifiers corresponding to univariate functions.
/// Example: S(t)
pub fn ci_univariate_func(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), _pi)) = tuple((mi, parenthesized_identifier))(input)?;
    Ok((
        s,
        Ci::new(
            Some(Type::Function),
            Box::new(MathExpression::Mi(Mi(x.trim().to_string()))),
        ),
    ))
}

/// Parse content identifier for Msub
pub fn ci_subscript(input: Span) -> IResult<Ci> {
    let (s, x) = msub(input)?;
    Ok((s, Ci::new(None, Box::new(x))))
}

/// Parse content identifier for Msup
pub fn ci_superscript(input: Span) -> IResult<Ci> {
    let (s, x) = msup(input)?;
    Ok((s, Ci::new(None, Box::new(x))))
}

/// Parse the identifier 'd'
fn d(input: Span) -> IResult<Operator> {
    let (s, Mi(x)) = mi(input)?;
    if let "d" = x.as_ref() {
        Ok((s, Operator::Exponential))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify Mi('d')".to_string(),
            input,
        )))
    }
}

/// Parse a content identifier of unknown type.
pub fn ci_unknown(input: Span) -> IResult<Ci> {
    let (s, x) = mi(input)?;
    Ok((s, Ci::new(None, Box::new(MathExpression::Mi(x)))))
}

/// Parse a first-order ordinary derivative written in Leibniz notation.
pub fn first_order_derivative_leibniz_notation(input: Span) -> IResult<(Derivative, Ci)> {
    let (s, _) = tuple((stag!("mfrac"), stag!("mrow"), d))(input)?;
    let (s, func) = ws(alt((
        ci_univariate_func,
        map(ci_unknown, |Ci { content, .. }| Ci {
            r#type: Some(Type::Function),
            content,
        }),
    )))(s)?;
    let (s, _) = tuple((
        etag!("mrow"),
        stag!("mrow"),
        d,
        mi,
        etag!("mrow"),
        etag!("mfrac"),
    ))(s)?;
    Ok((s, (Derivative::new(1, 1), func)))
}

pub fn newtonian_derivative(input: Span) -> IResult<(Derivative, Ci)> {
    // Get number of dots to recognize the order of the derivative
    let n_dots = delimited(
        stag!("mo"),
        map(many1(nom::character::complete::char('˙')), |x| {
            x.len() as u8
        }),
        etag!("mo"),
    );

    let (s, (x, order)) = terminated(
        delimited(
            stag!("mover"),
            pair(
                map(ci_unknown, |Ci { content, .. }| Ci {
                    r#type: Some(Type::Function),
                    content,
                }),
                n_dots,
            ),
            etag!("mover"),
        ),
        opt(parenthesized_identifier),
    )(input)?;

    Ok((s, (Derivative::new(order, 1), x)))
}

fn exp(input: Span) -> IResult<()> {
    let (s, Mi(x)) = mi(input)?;
    if let "e" = x.as_ref() {
        Ok((s, ()))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify Mi('e')".to_string(),
            input,
        )))
    }
}

pub fn exponential(input: Span) -> IResult<(Operator, MathExpression)> {
    let (s, x) = delimited(stag!("msup"), pair(exp, math_expression), etag!("msup"))(input)?;
    //let (s, x) = delimited(stag!("msup"), pair(exp, math_expression), etag!("msup"))(input)?;
    let (_, comp) = x;
    Ok((s, (Operator::Exponential, comp)))
}

// We reimplement the mfrac and mrow parsers in this file (instead of importing them from
// the generic_mathml module) to work with the specialized version of the math_expression parser
// (also in this file).
pub fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = ws(map(
        tag_parser!("mfrac", pair(math_expression, math_expression)),
        |(x, y)| MathExpression::Mfrac(Box::new(x), Box::new(y)),
    ))(input)?;

    Ok((s, frac))
}

pub fn mrow(input: Span) -> IResult<Mrow> {
    let (s, elements) = ws(delimited(
        stag!("mrow"),
        many0(math_expression),
        etag!("mrow"),
    ))(input)?;
    Ok((s, Mrow(elements)))
}

/// Parser for math expressions. This varies from the one in the generic_mathml module, since it
/// assumes that expressions such as S(t) are actually univariate functions.
pub fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((
        map(ci_univariate_func, MathExpression::Ci),
        map(ci_subscript, MathExpression::Ci),
        map(ci_superscript, MathExpression::Ci),
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(Type::Function),
                content,
            })
        }),
        map(operator, MathExpression::Mo),
        mn,
        msqrt,
        mfrac,
        map(mrow, MathExpression::Mrow),
        msubsup,
    )))(input)
}

/// Parser for interpreted math expressions.
/// testing MathML documents
pub fn interpreted_math(input: Span) -> IResult<Math> {
    let (s, elements) = preceded(opt(xml_declaration), elem_many0!("math"))(input)?;
    Ok((s, Math { content: elements }))
}
