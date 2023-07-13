use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, CiType, MathExpression,
    },
    parsers::{
        generic_mathml::{attribute, equals, etag, stag, ws, IResult, Span},
        interpreted_mathml::{
            ci_univariate_func, ci_unknown, first_order_derivative_leibniz_notation,
            math_expression, newtonian_derivative, operator,
        },
        math_expression_tree::MathExpressionTree,
    },
};

use derive_new::new;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::map,
    error::Error,
    multi::{many0, many1},
    sequence::{delimited, tuple},
};
use std::str::FromStr;

#[cfg(test)]
use crate::{ast::Mi, parsers::generic_mathml::test_parser};

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
                r#type: Some(CiType::Function),
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

impl FirstOrderODE {
    pub fn to_cmml(&self) -> String {
        let lhs_expression_tree = MathExpressionTree::Cons(
            Operator::Derivative(Derivative::new(1, 1)),
            vec![MathExpressionTree::Atom(Box::new(MathExpression::Ci(
                self.lhs_var.clone(),
            )))],
        );
        let combined = MathExpressionTree::Cons(
            Operator::Equals,
            vec![lhs_expression_tree, self.rhs.clone()],
        );
        combined.to_cmml()
    }
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
            Some(CiType::Function),
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
                Some(CiType::Function),
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
                Some(CiType::Function),
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
