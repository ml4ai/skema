use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, MathExpression, Mi, Mrow, Type,
    },
    parsers::generic_mathml::{
        add, attribute, elem2, equals, etag, lparen, mi, mn, mo, mover, msqrt, msub, msubsup, msup,
        rparen, stag, subtract, tag_parser, ws, IResult, ParseError, Span,
    },
    parsers::math_expression_tree::MathExpressionTree,
};
use derive_new::new;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::map,
    error::Error,
    multi::{many0, many1},
    sequence::{delimited, pair, tuple},
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
        let lhs_expression_tree = MathExpressionTree::new_cons(
            Operator::Derivative(Derivative::new(1, 1)),
            vec![MathExpressionTree::new_atom(MathExpression::new_ci(
                Box::new(self.lhs_var.clone()),
            ))],
        );
        let combined = MathExpressionTree::new_cons(
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

/// Fractions
pub fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = ws(map(
        tag_parser!("mfrac", pair(math_expression, math_expression)),
        |(x, y)| MathExpression::Mfrac(Box::new(x), Box::new(y)),
    ))(input)?;

    //let (s, frac) = elem2!("mfrac", Mfrac)(input)?;
    Ok((s, frac))
}

/// Rows
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
        map(ci_univariate_func, |x| MathExpression::new_ci(Box::new(x))),
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::new_ci(Box::new(Ci {
                r#type: Some(Type::Function),
                content,
            }))
        }),
        map(operator, MathExpression::Mo),
        //map(mi, MathExpression::Mi),
        mn,
        msup,
        msub,
        msqrt,
        mfrac,
        map(mrow, MathExpression::Mrow),
        mover,
        msubsup,
    )))(input)
}

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

    let FirstOrderODE { lhs_var, rhs } = input.parse::<FirstOrderODE>().unwrap();

    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(* (* (- β) S) I)");

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
    assert_eq!(rhs.to_string(), "(* (* (- β) S) I)");
}
