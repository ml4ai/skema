///! Program to parse MathML and convert it to a Petri Net
use clap::Parser;
use mathml::{
    ast::{Math, MathExpression, Operator},
    parsing::parse,
};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,

    /// Whether to normalize the output (collapse redundant mrows, collapse subscripts, etc.)
    #[arg(long, default_value_t = false)]
    normalize: bool,
}

/// Representation of a "named" variable
/// Here, 'variable' is intended to mean a symbolic name for a value.
/// Variables could be names of species (states) or rate (parameters).
#[derive(Debug, Eq, PartialEq, Default, Hash)]
struct Var {
    name: String,
    sub: (String, Option<Vec<String>>),
}

impl Var {
    fn new(name: &str) -> Var {
        return Var {
            name: name.to_string(),
            ..Default::default()
        };
    }

    //fn from_var_candidate
}

/// Represents the Tangent var of an ODE.
/// This is perhaps not really needed, although it at least introduces a type.
#[derive(Debug, Eq, PartialEq, Default)]
struct Tangent(Var);

#[derive(Debug, Eq, PartialEq)]
enum Polarity {
    add,
    sub,
}

/// A product of rate and species that is added or subtracted.
/// THere should just be one rate, but since we're parsing and there could
/// be noise, this accommodates possibly reading several names of things
/// that should be combined into a single rate.
#[derive(Debug, Eq, PartialEq)]
struct Term {
    rate: Vec<Var>,
    species: Vec<Var>,
    polarity: Polarity,
}

/// A single ODE equation.
#[derive(Debug, Eq, PartialEq, Default)]
struct Eqn {
    /// The left-hand-side is the tangent.
    lhs: Tangent,

    /// The right-hand-side of the equation, a sequence of Terms.
    rhs: Vec<Term>,

    /// Collects all of the Vars that appear in the rhs.
    rhs_vars: HashSet<Var>,
}

/// A collection of Eqns, indexed by the Var of the lhs Tangent.
#[derive(Debug, Eq, PartialEq, Default)]
struct EqnDict {
    eqns: HashMap<Var, Eqn>,

    /// The set of all Vars across the eqns that are interpreted as species.
    species: HashSet<Var>,

    /// The set of all Vars across the eqns that are interpreted as rates.
    rates: HashSet<Var>,
}

/// Check if fraction is Leibniz notation
fn is_leibniz_diff_op(numerator: &Box<MathExpression>, denominator: &Box<MathExpression>) -> bool {
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

fn var_candidate_to_var(expression: &MathExpression) -> Var {
    match expression {
                MathExpression::Mi(identifier) => return Var::new(&identifier),
                MathExpression::Msub(base, subscript) => {
                    // Get identifier from base
                    let base_identifier: String;

                    // Check that the base is an <mi> element
                    if let MathExpression::Mi(identifier) = &**base {
                        base_identifier = identifier.to_string();
                    } else {
                        panic!("Unhandled case!");
                    }

                    // Handle cases where the subscript is an Mi or an Mrow.
                    match &**subscript {
                        MathExpression::Mi(sub_identifier) => {
                            return Var {
                                name: base_identifier.to_string(),
                                sub: (sub_identifier.to_string(), None),
                            }
                        }
                        _ => {
                            panic!("For now, we only handle the case where the subscript is an Mi");
                        }
                    }
                }
                _ => panic!("For now, we only handle the case where there is a single 'Var' after the 'd' in the numerator of the Leibniz differential operator."),
            }
}
/// Translate a MathML mfrac (fraction) as an expression of a Leibniz differential operator.
/// In this case, the values after the 'd' or '∂' in the numerator are interpreted as
/// the Var tangent.
/// TODO: possibly generalize to accommodate superscripts for higher order derivatives;
///       although likely that would be an msup, so still the "first" elm of the numerator,
///       with the second (and beyond) elm(s) being the Var.
fn mfrac_leibniz_to_var(numerator: &Box<MathExpression>, denominator: &Box<MathExpression>) -> Var {
    // Check if numerator is an mrow
    if let MathExpression::Mrow(num_expressions) = &**numerator {
        // We assume here that the numerator is of the form dX where X is the variable of interest.
        if num_expressions.len() == 2 {
            let expression = &num_expressions[1];
            return var_candidate_to_var(expression);
        } else {
            panic!(
                "More than two elements in the numerator, cannot convert this Leibniz differential operator to a Var!");
        }
    } else {
        panic!("Unable to convert Leibniz differential operator to Var!");
    }
}

/// Translate MathML :mover as Newton dot notation
//fn mover_to_var(base: &Box<MathExpression>, overscript) -> Var {
// Check if the mover
//}

/// Get tangent
fn get_tangent(expressions: &[MathExpression]) -> Tangent {
    // Check if first expression is an mfrac

    match &expressions[0] {
        MathExpression::Mfrac(numerator, denominator) => {
            if is_leibniz_diff_op(numerator, denominator) {
                let var = mfrac_leibniz_to_var(numerator, denominator);
            }
        }
        MathExpression::Mover(base, overscript) => {
            // Check if overscript is ˙
            if let MathExpression::Mo(id) = &**overscript {
                if *id == Operator::Other("˙".to_string()) {
                    //var
                } else {
                    panic!("Overscript is not ˙, unhandled case!")
                }
            }
            //let var = mover_to_var(base, overscript);
        }
        _ => panic!("Unhandled case!"),
    }

    Tangent::default()
}

/// Converts a MathML AST to an Eqn
fn mml_to_eqn(ast: Math) -> Eqn {
    let mut equals_index = 0;

    // Check if the first element is an mrow
    if let MathExpression::Mrow(expr) = &ast.content[0] {
        // Get the index of the equals term
        for (i, term) in (*expr).iter().enumerate() {
            if let MathExpression::Mo(Operator::Equals) = term {
                if equals_index == 0 {
                    equals_index = i;
                } else {
                    panic!(
                        "Found multiple equals signs, does not look like an equation: {:?}",
                        ast
                    )
                }
            }
        }

        let lhs = &expr[0..equals_index];
        let rhs = &expr[equals_index + 1..];

        let tangent = get_tangent(lhs);
    }

    Eqn::default()
}

/// Translate list of MathML ASTs to EqnDict containing a collection of MAK ODE equations.
fn mathml_asts_to_eqn_dict(mathml_asts: Vec<Math>) -> EqnDict {
    let eqns = Vec::<Eqn>::new();
    let vars = HashSet::<Var>::new();
    let species = HashSet::<Var>::new();

    for ast in mathml_asts {
        let eqn = mml_to_eqn(ast);
    }

    EqnDict::default()
}

fn main() {
    let args = Cli::parse();
    let f = File::open(&args.input).unwrap();
    let lines = io::BufReader::new(f).lines();

    let mut mathml_asts = Vec::<Math>::new();

    for line in lines {
        if let Ok(l) = line {
            if let Some('#') = &l.chars().nth(0) {
                // Ignore lines starting with '#'
            } else {
                // Parse MathML into AST
                let (_, math) = parse(&l).unwrap_or_else(|_| panic!("Unable to parse line {}!", l));
                mathml_asts.push(math);
            }
        }
    }

    let eqn_dict = mathml_asts_to_eqn_dict(mathml_asts);
}
