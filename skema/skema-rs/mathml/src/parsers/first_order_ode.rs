use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, CiType, MathExpression,
    },
    parsers::{
        generic_mathml::{attribute, equals, etag, stag, ws, IResult, Span},
        interpreted_mathml::{
            ci_univariate_func, ci_unknown, first_order_derivative_leibniz_notation,
            math_expression_tree, newtonian_derivative, operator,
        },
        math_expression_tree::{Atom, MathExpressionTree},
        specific_tag_macros::{mi, mo},
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
//pub fn first_order_ode(input: Span) -> IResult<FirstOrderODE> {
//let (s, _) = stag!("math")(input)?;

//// Recognize LHS derivative
//let (s, lhs) = alt((
//first_order_derivative_leibniz_notation,
//newtonian_derivative,
//))(s)?;

//// Recognize equals sign
//let (s, _) = mo!(tag("="))(s)?;

//// Recognize other tokens
//let (s, remaining_tokens) = math_expression_tree(s)?;

//let (s, _) = etag!("math")(s)?;

//let ode = FirstOrderODE {
//lhs_var: ci,
//rhs: MathExpressionTree::from(remaining_tokens),
//};

//Ok((s, ode))
//}

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
        MathExpressionTree::new_cons(
            Operator::Derivative(Derivative::new(1, 1)),
            vec![Ci::new(
                Some(CiType::Function),
                Box::new(MathExpression::Mi(Mi("S".to_string()))),
            )
            .into()],
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
        MathExpressionTree::new_cons(
            Operator::Derivative(Derivative::new(1, 1)),
            vec![Ci::new(
                Some(CiType::Function),
                Box::new(MathExpression::Mi(Mi("S".to_string()))),
            )
            .into()],
        ),
    );
}
