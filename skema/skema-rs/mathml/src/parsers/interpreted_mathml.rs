//! This module contains parsers that perform some amount of preliminary domain-specific
//! interpretation of presentation MathML (e.g., globbing S(t) to an identifier S of type
//! 'function'). This is in contrast to the `generic_mathml.rs` module that contains parsers that
//! do not attempt to perform any interpretation but instead simply preserve the original MathML
//! document structure.

use crate::{
    ast::{
        operator::{Derivative, Domain, Operator, Sum},
        Ci, CiType, Cn, Math, MathExpression, Mi, Mrow,
    },
    parsers::{
        generic_mathml::{
            add, attribute, elem_many0, equals, etag, lparen, mi, mn, msqrt, msub, msubsup, msup,
            rparen, stag, subtract, tag_parser, ws, xml_declaration, IResult, ParseError, Span,
        },
        math_expression_tree::MathExpressionTree,
    },
};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::{many0, many1, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
};

/// A macro to recognize specific mi elements.
macro_rules! mi {
    ($parser:expr) => {{
        tag_parser!("mi", $parser)
    }};
}

/// A macro to recognize specific mo elements.
macro_rules! mo {
    ($parser:expr) => {{
        tag_parser!("mo", $parser)
    }};
}

/// A macro to recognize mrows with specific contents
macro_rules! mrow {
    ($parser:expr) => {{
        tag_parser!("mrow", $parser)
    }};
}

/// A macro to recognize mover elements with specific contents
macro_rules! mover {
    ($parser:expr) => {{
        tag_parser!("mover", $parser)
    }};
}

/// A macro to recognize munder elements with specific contents
macro_rules! munder {
    ($parser:expr) => {{
        tag_parser!("munder", $parser)
    }};
}

/// A macro to recognize mfrac elements with specific contents
macro_rules! mfrac {
    ($parser:expr) => {{
        tag_parser!("mfrac", $parser)
    }};
}

/// Function to parse operators. This function differs from the one in parsers::generic_mathml by
/// disallowing operators besides +, -, =, (, and ).
pub fn operator(input: Span) -> IResult<Operator> {
    let (s, op) = mo!(alt((add, subtract, equals, lparen, rparen)))(input)?;
    Ok((s, op))
}

fn parenthesized_identifier(input: Span) -> IResult<()> {
    let mo_lparen = mo!(lparen);
    let mo_rparen = mo!(rparen);
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
            Some(CiType::Function),
            Box::new(MathExpression::Mi(Mi(x.trim().to_string()))),
        ),
    ))
}

/// Parse content identifier for Msub
pub fn ci_subscript(input: Span) -> IResult<Ci> {
    let (s, x) = msub(input)?;
    Ok((s, Ci::new(None, Box::new(x))))
}

/// Parse the identifier 'd'
fn d(input: Span) -> IResult<()> {
    let (s, _) = mi!(tag("d"))(input)?;
    Ok((s, ()))
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
            r#type: Some(CiType::Function),
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
    let n_dots = mo!(map(many1(char('˙')), |x| { x.len() as u8 }));

    let (s, (x, order)) = terminated(
        mover!(pair(
            map(ci_unknown, |Ci { content, .. }| Ci {
                r#type: Some(CiType::Function),
                content,
            }),
            n_dots,
        )),
        opt(parenthesized_identifier),
    )(input)?;

    Ok((s, (Derivative::new(order, 1), x)))
}

// We reimplement the mfrac and mrow parsers in this file (instead of importing them from
// the generic_mathml module) to work with the specialized version of the math_expression parser
// (also in this file).
pub fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = ws(map(
        mfrac!(pair(math_expression, math_expression)),
        |(x, y)| MathExpression::Mfrac(Box::new(x), Box::new(y)),
    ))(input)?;

    Ok((s, frac))
}

pub fn mrow(input: Span) -> IResult<Mrow> {
    let (s, elements) = ws(mrow!(many0(math_expression)))(input)?;
    Ok((s, Mrow(elements)))
}

/// Parser for math expressions. This varies from the one in the generic_mathml module, since it
/// assumes that expressions such as S(t) are actually univariate functions.
pub fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((
        map(ci_univariate_func, MathExpression::Ci),
        map(ci_subscript, MathExpression::Ci),
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(CiType::Function),
                content,
            })
        }),
        map(operator, MathExpression::Mo),
        mn,
        msup,
        msub,
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

fn summation(input: Span) -> IResult<Sum> {
    let summation_symbol = tag_parser!("mo", tag("∑"));
    let comma = mo!(tag(","));

    let to_ci = |x| {
        MathExpressionTree::Atom(Box::new(MathExpression::Ci(Ci::new(
            None,
            Box::new(MathExpression::Mi(x)),
        ))))
    };

    let domain = mrow!(separated_pair(
        mi,
        mo!(tag("=")),
        map(separated_list1(comma, mi), move |xs| {
            xs.into_iter()
                .map(to_ci)
                .collect::<Vec<MathExpressionTree>>()
        })
    ));

    let (s, (bvar, domain_elements)) = munder!(preceded(summation_symbol, domain))(input)?;

    let args = vec![
        to_ci(bvar.clone()),
        MathExpressionTree::Cons(Operator::Set, domain_elements),
    ];

    Ok((
        s,
        Sum::new(
            vec![Ci::new(None, Box::new(MathExpression::Mi(bvar)))],
            Domain::Condition(Box::new(MathExpressionTree::Cons(Operator::In, args))),
        ),
    ))
}

#[test]
fn test_summation() {
    let input = "
        <munder>
            <mo>∑</mo>
            <mrow>
            <mi>X</mi>
            <mo>=</mo>
            <mi>W</mi>
            <mo>,</mo>
            <mi>A</mi>
            <mo>,</mo>
            <mi>D</mi>
            </mrow>
        </munder>";

    let (_, summation) = summation(input.into()).unwrap();
    println!("{summation}");
}
