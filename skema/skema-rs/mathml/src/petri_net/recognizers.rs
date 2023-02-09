use crate::ast::{
    MathExpression,
    MathExpression::{Mfrac, Mi, Mn, Mo, Mover, Mrow, Msub},
    Operator,
};
use crate::petri_net::{Polarity, Var};

/// Check if fraction is a derivative expressed in Leibniz notation
pub fn is_leibniz_diff_operator(
    numerator: &Box<MathExpression>,
    denominator: &Box<MathExpression>,
) -> bool {
    let mut numerator_contains_d = false;
    let mut denominator_contains_d = false;

    let mut numerator_contains_partial = false;
    let mut denominator_contains_partial = false;

    // Check if numerator is an mrow
    if let MathExpression::Mrow(num_expressions) = &**numerator {
        // Check if first element of numerator is an mi
        if let MathExpression::Mi(num_id) = &num_expressions[0] {
            // Check if mi contains 'd'
            if num_id == "d" {
                numerator_contains_d = true;
            }

            if num_id == "∂" {
                numerator_contains_partial = true;
            }
        }
    }

    if let MathExpression::Mrow(denom_expressions) = &**denominator {
        // Check if first element of denominator is an mi
        if let MathExpression::Mi(denom_id) = &denom_expressions[0] {
            // Check if mi contains 'd'
            if denom_id == "d" {
                denominator_contains_d = true;
            }
            if denom_id == "∂" {
                denominator_contains_partial = true;
            }
        }
    }

    (numerator_contains_d && denominator_contains_d)
        || (numerator_contains_partial && denominator_contains_partial)
}

/// Predicate testing whether a MathML operator (:mo) is a subtraction or addition.
pub fn is_add_or_subtract_operator(element: &MathExpression) -> bool {
    if let MathExpression::Mo(operator) = element {
        return Operator::Add == *operator || Operator::Subtract == *operator;
    }
    false
}

pub fn get_polarity(element: &MathExpression) -> Polarity {
    if let MathExpression::Mo(op) = element {
        if *op == Operator::Subtract {
            Polarity::negative
        } else if *op == Operator::Add {
            Polarity::positive
        } else {
            panic!("Unhandled operator!");
        }
    } else {
        panic!("Element must be of type Mo!");
    }
}

/// Get specie_var
pub fn get_specie_var(expression: &MathExpression) -> Var {
    // Check if expression is an mfrac
    match expression {
        Mfrac(numerator, denominator) => {
            if is_leibniz_diff_operator(numerator, denominator) {
                let specie = mfrac_leibniz_to_specie(numerator, denominator);
                return specie;
            } else {
                panic!("Expression is an mfrac but not a Leibniz diff operator!");
            }
        }
        // Translate MathML :mover as Newton dot notation
        Mover(base, overscript) => {
            // Check if overscript is ˙
            if let Mo(id) = &**overscript {
                if *id == Operator::Other("˙".to_string()) {
                    return Var(*base.clone());
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
/// TODO: This currently permits :mn -> MathML numerical literals.
///     Perhaps useful to represent constant coefficients?
///     But should those be Vars?
pub fn is_var_candidate(element: &MathExpression) -> bool {
    match element {
        Mi(x) => true,
        Mn(x) => true,
        Msub(x1, x2) => true,
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
    denominator: &Box<MathExpression>,
) -> Var {
    // Check if numerator is an mrow
    if let Mrow(num_expressions) = &**numerator {
        // We assume here that the numerator is of the form dX where X is the variable of interest.
        if num_expressions.len() == 2 {
            let expression = &num_expressions[1];
            return Var(expression.clone());
        } else {
            panic!(
                "More than two elements in the numerator, cannot extract specie from Leibniz differential operator!");
        }
    } else {
        panic!("Unable to extract specie from Leibniz differential operator!");
    }
}
