use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, MathExpression, Mi, Mrow, Type,
    },
    parsers::generic_mathml::{
        add, attribute, equals, etag, lparen, mi, mn, msqrt, msub, msubsup, msup, rparen, stag,
        subtract, tag_parser, ws, IResult, ParseError, Span,
    },
    parsers::math_expression_tree::MathExpressionTree,
};
use derive_new::new;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt},
    error::Error,
    multi::{many0, many1},
    sequence::{delimited, pair, terminated, tuple},
};
use std::str::FromStr;

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

impl FirstOrderODE {
    pub fn to_cmml(&self) -> String {
        let lhs_expression_tree = MathExpressionTree::Cons(
            Operator::Derivative(Derivative::new(1, 1)),
            vec![MathExpressionTree::Atom(MathExpression::Ci(
                self.lhs_var.clone(),
            ))],
        );
        let combined = MathExpressionTree::Cons(
            Operator::Equals,
            vec![lhs_expression_tree, self.rhs.clone()],
        );
        combined.to_cmml()
    }
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

fn parenthesized_identifier(input: Span) -> IResult<()> {
    let mo_lparen = delimited(stag!("mo"), lparen, etag!("mo"));
    let mo_rparen = delimited(stag!("mo"), rparen, etag!("mo"));
    let (s, _) = delimited(mo_lparen, mi, mo_rparen)(input)?;
    Ok((s, ()))
}

/// Parse content identifiers corresponding to univariate functions.
/// Example: S(t)
fn ci_univariate_func(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), _pi)) = tuple((mi, parenthesized_identifier))(input)?;
    Ok((
        s,
        Ci::new(
            Some(Type::Function),
            Box::new(MathExpression::Mi(Mi(x.trim().to_string()))),
        ),
    ))
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
    Ok((s, Ci::new(None, Box::new(MathExpression::Mi(x)))))
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

fn newtonian_derivative(input: Span) -> IResult<(Derivative, Ci)> {
    // Get number of dots to recognize the order of the derivative
    println!("FLAG1");
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
    println!("FLAG2, {x:?}");

    Ok((s, (Derivative::new(order, 1), x)))
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
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(Type::Function),
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

/// Parse a first order ODE with a single derivative term on the LHS.
pub fn first_order_ode(input: Span) -> IResult<FirstOrderODE> {
    let (s, _) = stag!("math")(input)?;

    // Recognize LHS derivative
    let (s, (_, ci)) = alt((
        first_order_derivative_leibniz_notation,
        newtonian_derivative,
    ))(s)?;

    // Recognize equals sign
    let (s, _) = delimited(stag!("mo"), equals, etag!("mo"))(s)?;

    // Recognize other tokens
    let (s, remaining_tokens) = many1(alt((
        map(ci_univariate_func, MathExpression::Ci),
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(Type::Function),
                content,
            })
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

impl FromStr for FirstOrderODE {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ode = first_order_ode(s.into()).unwrap().1;
        Ok(ode)
    }
}

#[test]
fn test_ci_univariate_func() {
    test_parser(
        "<mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>",
        ci_univariate_func,
        Ci::new(
            Some(Type::Function),
            Box::new(MathExpression::Mi(Mi("S".to_string()))),
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
                Box::new(MathExpression::Mi(Mi("S".to_string()))),
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
                Box::new(MathExpression::Mi(Mi("S".to_string()))),
            ),
        ),
    );
}

#[test]
fn test_first_order_ode() {
    // ASKEM Hackathon 2, scenario 1, equation 1.
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>-</mo>
        <mi>β</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mfrac><mrow><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mi>N</mi></mfrac>
    </math>
    ";

    let FirstOrderODE { lhs_var, rhs } = input.parse::<FirstOrderODE>().unwrap();

    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(/ (* (* (- β) I) S) N)");

    // ASKEM Hackathon 2, scenario 1, equation 1, but with Newtonian derivative notation.
    let input = "
    <math>
        <mover><mi>S</mi><mo>˙</mo></mover><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>=</mo>
        <mo>-</mo>
        <mi>β</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mfrac><mrow><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mi>N</mi></mfrac>
    </math>
    ";

    let FirstOrderODE { lhs_var, rhs } = input.parse::<FirstOrderODE>().unwrap();

    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(/ (* (* (- β) I) S) N)");
}
