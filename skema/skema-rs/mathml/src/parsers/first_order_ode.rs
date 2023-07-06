use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, MathExpression, Mi, Type,
    },
    parsers::generic_mathml::{
        add, attribute, equals, etag, lparen, math_expression, mi, mo, rparen, stag, subtract, ws,
        IResult, ParseError, Span,
    },
    parsers::math_expression_tree::MathExpressionTree,
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
use crate::parsers::generic_mathml::test_parser;

/// First order ordinary differential equation.
/// This assumes that the left hand side of the equation consists solely of a derivative expressed
/// in Leibniz or Newtonian notation.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct FirstOrderODE {
    /// The variable/univariate function) on the LHS of the equation that is being
    /// differentiated. This variable may be referred to as a 'specie', 'state', or 'vertex' in the
    /// context of discussions about Petri Nets and RegNets.
    pub lhs_var: Ci,

    /// An expression tree corresponding to the RHS of the ODE.
    pub rhs: MathExpressionTree,
}

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

/// Parse content identifiers corresponding to univariate functions.
/// Example: S(t)
fn ci_univariate_func(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), left, Mi(_), right)) = tuple((mi, mo, mi, mo))(input)?;
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

/// Parse the identifier 'd'
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

/// Parse a content identifier of unknown type.
fn ci_unknown(input: Span) -> IResult<Ci> {
    let (s, x) = mi(input)?;
    Ok((s, Ci::new(None, MathExpression::Mi(x))))
}

/// Parse a first-order ordinary derivative written in Leibniz notation.
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

//fn newtonian_derivative(input: Span) -> IResult<(Derivative, Ci)> {
//let (s, (_, _, _, _)) = tuple((stag!("mover"), mi, mo, etag!("mover")))(input)?;
//todo!()
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
//}

/// Parse a first order ODE with a single derivative term on the LHS.
pub fn first_order_ode(input: Span) -> IResult<FirstOrderODE> {
    let (s, _) = stag!("math")(input)?;

    // Recognize LHS derivative
    let (s, (_, ci)) = first_order_derivative_leibniz_notation(s)?;

    // Recognize equals sign
    let (s, _) = delimited(stag!("mo"), equals, etag!("mo"))(s)?;

    // Recognize other tokens
    let (s, remaining_tokens) = many1(alt((
        map(ci_univariate_func, |x| MathExpression::new_ci(Box::new(x))),
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
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>-</mo>
        <mi>β</mi>
        <mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";

    println!("Input: {input}");
    let (_, FirstOrderODE { lhs_var, rhs }) = first_order_ode(input.into()).unwrap();

    println!("Output:\n");
    println!("\tLHS var: {lhs_var}");
    assert_eq!(lhs_var.to_string(), "S");
    println!("\tRHS: {rhs}");
    assert_eq!(rhs.to_string(), "(* (* (- β) S) I)");
    println!()
}
