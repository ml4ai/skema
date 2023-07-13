//! This module contains parsers that perform some amount of preliminary domain-specific
//! interpretation of presentation MathML (e.g., globbing S(t) to an identifier S of type
//! 'function'). This is in contrast to the `generic_mathml.rs` module that contains parsers that
//! do not attempt to perform any interpretation but instead simply preserve the original MathML
//! document structure.

use crate::{
    ast::{
        operator::{Derivative, Domain, Operator, Sum},
        Ci, CiType, Math, MathExpression, Mi, Mrow,
    },
    parsers::{
        generic_mathml::{
            add, attribute, elem_many0, equals, etag, lparen, mfrac, mi, mn, mo, msqrt, msub,
            msubsup, msup, rparen, stag, subtract, tag_parser, ws, xml_declaration, IResult, Span,
        },
        math_expression_tree::{Atom, MathExpressionTree},
        specific_tag_macros::{math, mfrac, mi, mo, mover, mrow, munder},
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
pub fn first_order_derivative_leibniz_notation(input: Span) -> IResult<MathExpressionTree> {
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

    let tree = MathExpressionTree::new_cons(
        Operator::Derivative(Derivative::new(1, 1)),
        vec![func.into()],
    );

    Ok((s, tree))
}

pub fn newtonian_derivative(input: Span) -> IResult<MathExpressionTree> {
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

    let tree = MathExpressionTree::new_cons(
        Operator::Derivative(Derivative::new(order, 1)),
        vec![x.into()],
    );

    Ok((s, tree))
}

// We reimplement the mfrac and mrow parsers in this file (instead of importing them from
// the generic_mathml module) to work with the specialized version of the math_expression parser
// (also in this file).
pub fn fraction(input: Span) -> IResult<MathExpressionTree> {
    let (s, (numerator, denominator)) =
        mfrac!(pair(math_expression_tree, math_expression_tree))(input)?;

    let tree = MathExpressionTree::new_cons(Operator::Divide, vec![numerator, denominator]);
    Ok((s, tree))
}

pub fn mrow(input: Span) -> IResult<MathExpressionTree> {
    let (s, elements) = mrow!(pmml_elements)(input)?;
    let tree = MathExpressionTree::from(elements);
    Ok((s, tree))
}

pub fn pmml_elements(input: Span) -> IResult<Vec<MathExpression>> {
    let (s, elements) = many1(alt((
        map(ci, MathExpression::Ci),
        mo,
        mn,
        map(ci_univariate_func, MathExpression::Ci),
        mfrac,
    )))(input)?;
    Ok((s, elements))
}

/// Parse a first order ODE with a single derivative term on the LHS.
pub fn first_order_ode(input: Span) -> IResult<MathExpressionTree> {
    // Recognize LHS derivative
    let (s, lhs) = alt((
        first_order_derivative_leibniz_notation,
        newtonian_derivative,
    ))(input)?;

    // Recognize equals sign
    let (s, _) = mo!(tag("="))(s)?;

    // Recognize other tokens
    let (s, remaining_tokens) = pmml_elements(s)?;

    let tree = MathExpressionTree::new_cons(Operator::Equals, vec![lhs, remaining_tokens.into()]);

    Ok((s, tree))
}

fn ci(input: Span) -> IResult<Ci> {
    let (s, res) = alt((ci_univariate_func, ci_subscript, ci_unknown))(input)?;
    Ok((s, res))
}

/// Parser for math expressions. This varies from the one in the generic_mathml module, since it
/// assumes that expressions such as S(t) are actually univariate functions.
pub fn math_expression_tree(input: Span) -> IResult<MathExpressionTree> {
    ws(alt((
        first_order_ode,
        fraction,
        map(ci_univariate_func, |x| x.into()),
        map(ci_subscript, |x| x.into()),
        map(ci_unknown, |Ci { content, .. }| {
            Ci::new(Some(CiType::Function), content).into()
        }),
        map(pmml_elements, MathExpressionTree::from),
        mrow,
        //msubsup,
    )))(input)
}

/// Parser for interpreted math expressions.
/// testing MathML documents
pub fn math_expression_tree_document(input: Span) -> IResult<MathExpressionTree> {
    let (s, tree) = preceded(opt(xml_declaration), math!(math_expression_tree))(input)?;
    Ok((s, tree))
}

fn summation(input: Span) -> IResult<Sum> {
    let summation_symbol = tag_parser!("mo", tag("∑"));
    let comma = mo!(tag(","));

    let to_ci =
        |x| MathExpressionTree::Atom(Atom::Ci(Ci::new(None, Box::new(MathExpression::Mi(x)))));

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
