use crate::{
    ast::{MathExpression, Operator},
    petri_net::Polarity,
};

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
            Polarity::sub
        } else if *op == Operator::Add {
            Polarity::add
        } else {
            panic!("Unhandled operator!");
        }
    } else {
        panic!("Element must be of type Mo!");
    }
}
