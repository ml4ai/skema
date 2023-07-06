use crate::{
    ast::{Ci, Derivative, MathExpression, Mi, Operator, Type},
    parsers::math_expression_tree::MathExpressionTree,
    parsing::{attribute, etag, math_expression, mi, mo, stag, ws, IResult, ParseError, Span},
};
use derive_new::new;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::one_of,
    combinator::{map, value},
    multi::{many0, many1},
    sequence::{delimited, tuple},
};

#[cfg(test)]
use crate::parsing::test_parser;

/// First order ordinary differential equation.
/// This assumes that the left hand side of the equation consists solely of a derivative expressed
/// in Leibniz or Newtonian notation.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct FirstOrderODE {
    pub lhs_var: Ci,
    pub rhs: MathExpressionTree,
}

// Operators, including surrounding tags.
fn add(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Add, ws(tag("+")))(input)?;
    Ok((s, op))
}

fn subtract(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Subtract, ws(one_of("-−")))(input)?;
    Ok((s, op))
}

fn equals(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Equals, ws(tag("=")))(input)?;
    Ok((s, op))
}

pub fn lparen(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Lparen, ws(tag("(")))(input)?;
    Ok((s, op))
}

pub fn rparen(input: Span) -> IResult<Operator> {
    let (s, op) = value(Operator::Rparen, ws(tag(")")))(input)?;
    Ok((s, op))
}

pub fn operator(input: Span) -> IResult<Operator> {
    let (s, op) = ws(delimited(
        stag!("mo"),
        alt((add, subtract, equals, lparen, rparen)),
        etag!("mo"),
    ))(input)?;
    Ok((s, op))
}

fn ci_univariate_func(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), left, Mi(y), right)) = tuple((mi, mo, mi, mo))(input)?;
    if let (MathExpression::Mo(Operator::Lparen), MathExpression::Mo(Operator::Rparen)) =
        (left, right)
    {
        Ok((
            s,
            Ci::new(
                Some(Type::Function),
                MathExpression::Mi(Mi(x.trim().to_string())),
            ),
        ))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify univariate function".to_string(),
            input,
        )))
    }
}

fn d(input: Span) -> IResult<()> {
    let (s, Mi(x)) = mi(input)?;
    if let "d" = x.as_ref() {
        Ok((s, ()))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify Mi('d')".to_string(),
            input,
        )))
    }
}

// Parse an content identifier of unknown type.
fn ci_unknown(input: Span) -> IResult<Ci> {
    let (s, x) = mi(input)?;
    Ok((s, Ci::new(None, MathExpression::Mi(x))))
}

fn first_order_derivative_leibniz_notation(input: Span) -> IResult<(Derivative, Ci)> {
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

fn newtonian_derivative(input: Span) -> IResult<(Derivative, Ci)> {
    let (s, (_, _, _, _)) = tuple((stag!("mover"), mi, mo, etag!("mover")))(input)?;
    todo!()
    //let (s, func) = ws(alt((univariate_func, ci_func)))(s)?;
    //let (s, _) = tuple((
    //etag!("mrow"),
    //stag!("mrow"),
    //d,
    //mi,
    //etag!("mrow"),
    //etag!("mfrac"),
    //))(s)?;
    //Ok((s, (Derivative::new(1, 1), func)))
}

/// First order ordinary differential equation.
pub fn first_order_ode(input: Span) -> IResult<FirstOrderODE> {
    let (s, _) = stag!("math")(input)?;

    // Recognize LHS derivative
    let (s, (_, ci)) = first_order_derivative_leibniz_notation(s)?;

    // Recognize equals sign
    let (s, _) = delimited(stag!("mo"), equals, etag!("mo"))(s)?;

    // Recognize other tokens
    let (s, remaining_tokens) = many1(alt((
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::new_ci(Box::new(Ci {
                r#type: Some(Type::Function),
                content,
            }))
        }),
        map(operator, MathExpression::Mo),
        math_expression,
    )))(s)?;

    let (s, _) = etag!("math")(s)?;

    let ode = FirstOrderODE {
        lhs_var: ci,
        rhs: MathExpressionTree::from(remaining_tokens),
    };

    Ok((s, ode))
}

#[test]
fn test_ci_univariate_func() {
    test_parser(
        "<mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>",
        ci_univariate_func,
        Ci::new(
            Some(Type::Function),
            MathExpression::Mi(Mi("S".to_string())),
        ),
    );
}

#[test]
fn test_first_order_derivative_leibniz_notation_with_implicit_time_dependence() {
    test_parser(
        "<mfrac>
        <mrow><mi>d</mi><mi>S</mi></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>",
        first_order_derivative_leibniz_notation,
        (
            Derivative::new(1, 1),
            Ci::new(
                Some(Type::Function),
                MathExpression::Mi(Mi("S".to_string())),
            ),
        ),
    );
}

#[test]
fn test_first_order_derivative_leibniz_notation_with_explicit_time_dependence() {
    test_parser(
        "<mfrac>
        <mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>",
        first_order_derivative_leibniz_notation,
        (
            Derivative::new(1, 1),
            Ci::new(
                Some(Type::Function),
                MathExpression::Mi(Mi("S".to_string())),
            ),
        ),
    );
}

#[test]
fn test_first_order_ode() {
    let (_, FirstOrderODE { lhs_var, rhs }) = first_order_ode(
        "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>-</mo>
        <mi>β</mi>
        <mi>S</mi>
        <mi>I</mi>
    </math>
    "
        .into(),
    )
    .unwrap();

    println!("LHS specie: {lhs_var}, RHS: {rhs}");
}
