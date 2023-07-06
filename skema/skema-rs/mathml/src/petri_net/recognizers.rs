//! This module contains functions for predicate testing of MathML elements, as well as extracting
//! semantically-meaningful objects from MathML expressions.

use crate::ast::{
    Derivative, MathExpression,
    MathExpression::{Mfrac, Mn, Mo, Mover, Mrow, Msub},
    Mi, Operator,
};
use crate::petri_net::{Polarity, Var};

/// Check if fraction is a derivative of a single-variable function expressed in Leibniz notation,
/// and if so, return a derivative operator and the identifier of the function.
pub fn recognize_leibniz_differential_operator<'a>(
    numerator: &Box<MathExpression>,
    denominator: &Box<MathExpression>,
) -> Result<(Operator, MathExpression), &'a str> {
    let mut numerator_contains_d = false;
    let mut denominator_contains_d = false;

    let mut numerator_contains_partial = false;
    let mut denominator_contains_partial = false;
    let mut function_candidate: Option<MathExpression> = None;

    // Check if numerator is an mrow
    if let MathExpression::Mrow(num_expressions) = &**numerator {
        // Check if first element of numerator is an mi
        if let MathExpression::Mi(Mi(num_id)) = &num_expressions.0[0] {
            // Check if mi contains 'd'
            if num_id == "d" {
                numerator_contains_d = true;
            }

            if num_id == "∂" {
                numerator_contains_partial = true;
            }

            // Gather the second identifier as a potential candidate function.
            function_candidate = Some(num_expressions.0[1].clone());
        }
    }

    if let MathExpression::Mrow(denom_expressions) = &**denominator {
        // Check if first element of denominator is an mi
        if let MathExpression::Mi(Mi(denom_id)) = &denom_expressions.0[0] {
            // Check if mi contains 'd'
            if denom_id == "d" {
                denominator_contains_d = true;
            }
            if denom_id == "∂" {
                denominator_contains_partial = true;
            }
        }
    }

    if (numerator_contains_d && denominator_contains_d)
        || (numerator_contains_partial && denominator_contains_partial)
    {
        Ok((
            Operator::new_derivative(Derivative::new(1, 1)),
            function_candidate.unwrap(),
        ))
    } else {
        Err("This Mfrac does not correspond to a derivative in Leibniz notation")
    }
}

/// Predicate testing whether a MathML operator (Mo) is a subtraction or addition.
pub fn is_add_or_subtract_operator(element: &MathExpression) -> bool {
    if let MathExpression::Mo(operator) = element {
        return Operator::Add == *operator || Operator::Subtract == *operator;
    }
    false
}

/// Get polarity from a Mo element.
pub fn get_polarity(element: &MathExpression) -> Polarity {
    if let MathExpression::Mo(op) = element {
        if *op == Operator::Subtract {
            Polarity::Negative
        } else if *op == Operator::Add {
            Polarity::Positive
        } else {
            panic!("Unhandled operator!");
        }
    } else {
        panic!("Element must be of type Mo!");
    }
}

/// Get variable corresponding to specie.
pub fn get_specie_var(expression: &MathExpression) -> Var {
    // Check if expression is an mfrac
    match expression {
        Mfrac(numerator, denominator) => {
            if let Ok(_) = recognize_leibniz_differential_operator(numerator, denominator) {
                mfrac_leibniz_to_specie(numerator, denominator)
            } else {
                panic!("Expression is an mfrac but not a Leibniz differential operator!");
            }
        }
        // Translate MathML :mover as Newton dot notation
        Mover(base, overscript) => {
            // Check if overscript is ˙
            if let Mo(id) = &**overscript {
                if *id == Operator::Other("˙".to_string()) {
                    Var(*base.clone())
                } else {
                    panic!("Overscript is not ˙, unhandled case!")
                }
            } else {
                panic!("Found an overscript that is not an Mo, aborting!");
            }
        }
        _ => panic!("Unhandled case!"),
    }
}

/// Predicate testing whether a MathML elm could be interpreted as a Var.
/// TODO: This currently permits Mn -> MathML numerical literals.
///     Perhaps useful to represent constant coefficients?
///     But should those be Vars?
pub fn is_var_candidate(element: &MathExpression) -> bool {
    match element {
        MathExpression::Mi(_x) => true,
        Mn(_x) => true,
        Msub(_x1, _x2) => true,
        _ => false,
    }
}

/// Translate a MathML mfrac (fraction) as an expression of a Leibniz differential operator.
/// In this case, the values after the 'd' or '∂' in the numerator are interpreted as
/// the Var tangent.
/// TODO: possibly generalize to accommodate superscripts for higher order derivatives;
///       although likely that would be an msup, so still the "first" elm of the numerator,
///       with the second (and beyond) elm(s) being the Var.
fn mfrac_leibniz_to_specie(
    numerator: &Box<MathExpression>,
    _denominator: &Box<MathExpression>,
) -> Var {
    // Check if numerator is an mrow
    if let Mrow(num_expressions) = &**numerator {
        // We assume here that the numerator is of the form dX where X is the variable of interest.
        if num_expressions.0.len() == 2 {
            let expression = &num_expressions.0[1];
            Var(expression.clone())
        } else {
            panic!(
                "More than two elements in the numerator, cannot extract specie from Leibniz differential operator!");
        }
    } else {
        panic!("Unable to extract specie from Leibniz differential operator!");
    }
}
