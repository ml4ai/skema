use crate::{
    ast::{Ci, Derivative, MathExpression, Mi, Operator, Type},
    parsing::{attribute, etag, math_expression, mi, mo, stag, ws, IResult, ParseError, Span},
    pratt_parsing::Token,
};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::one_of,
    combinator::{map, value},
    multi::{many0, many1},
    sequence::{delimited, tuple},
};

#[cfg(test)]
use crate::{parsing::test_parser, pratt_parsing::MathExpressionTree};

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

fn univariate_func(input: Span) -> IResult<Ci> {
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

fn ci_func(input: Span) -> IResult<Ci> {
    let (s, x) = mi(input)?;
    Ok((s, Ci::new(Some(Type::Function), MathExpression::Mi(x))))
}

fn first_order_derivative_leibniz_notation(input: Span) -> IResult<(Derivative, Ci)> {
    let (s, _) = tuple((stag!("mfrac"), stag!("mrow"), d))(input)?;
    let (s, func) = ws(alt((univariate_func, ci_func)))(s)?;
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

/// First order ordinary differential equation.
pub fn ode(input: Span) -> IResult<Vec<MathExpression>> {
    let (s, _) = stag!("math")(input)?;

    // Recognize LHS derivative
    let (s, (derivative, ci)) = first_order_derivative_leibniz_notation(s)?;

    // Recognize equals sign
    let (s, _) = delimited(stag!("mo"), equals, etag!("mo"))(s)?;

    // If we get here, initialize the token list
    let mut tokens = vec![
        MathExpression::Mo(Operator::Derivative(derivative)),
        MathExpression::Ci(Box::new(ci)),
        MathExpression::Mo(Operator::Equals),
    ];

    // Recognize other tokens
    let (s, remaining_tokens) = many1(alt((
        map(ci_func, |x| MathExpression::new_ci(Box::new(x))),
        map(operator, MathExpression::Mo),
        math_expression,
    )))(s)?;

    let (s, _) = etag!("math")(s)?;

    tokens.extend(remaining_tokens);

    Ok((s, tokens))
}

#[test]
fn test_dsp() {
    test_parser(
        "<mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>",
        univariate_func,
        Ci::new(
            Some(Type::Function),
            MathExpression::Mi(Mi("S".to_string())),
        ),
    );

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

    // Test derivative with explicit time dependence
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
fn test_ode() {
    // Test ODE
    let (s, tokens) = ode("
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
    .into())
    .unwrap();

    let s_expression = MathExpressionTree::from(tokens);
    println!("{s_expression}");
}
