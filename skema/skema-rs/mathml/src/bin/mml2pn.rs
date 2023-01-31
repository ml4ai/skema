///! Program to parse MathML and convert it to a Petri Net
use clap::Parser;
use mathml::{
    ast::{Math, MathExpression, Operator},
    parsing::parse,
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead},
};

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
#[derive(Debug, Eq, PartialEq, Default, Hash, Clone)]
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
}

/// Represents the Tangent var of an ODE.
/// This is perhaps not really needed, although it at least introduces a type.
#[derive(Debug, Eq, PartialEq, Default, Clone)]
struct Tangent(Var);

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
enum Polarity {
    add,
    sub,
}

/// A product of rate and species that is added or subtracted.
/// THere should just be one rate, but since we're parsing and there could
/// be noise, this accommodates possibly reading several names of things
/// that should be combined into a single rate.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Term {
    rate: Vec<Vec<MathExpressionOrVarOrTerm>>,
    species: Vec<Vec<MathExpressionOrVarOrTerm>>,
    polarity: Polarity,
}

/// A single ODE equation.
#[derive(Debug, Eq, PartialEq, Default, Clone)]
struct Eqn {
    /// The left-hand-side is the tangent.
    lhs: Tangent,

    /// The right-hand-side of the equation, a sequence of Terms.
    rhs: Vec<Vec<MathExpressionOrVarOrTerm>>,

    /// Collects all of the Vars that appear in the rhs.
    rhs_vars: HashSet<MathExpressionOrVarOrTerm>,
}

/// A collection of Eqns, indexed by the Var of the lhs Tangent.
#[derive(Debug, Eq, PartialEq, Default)]
struct EqnDict {
    eqns: HashMap<Var, Eqn>,

    /// The set of all Vars across the eqns that are interpreted as species.
    species: HashSet<MathExpressionOrVarOrTerm>,

    /// The set of all Vars across the eqns that are interpreted as rates.
    rates: HashSet<MathExpressionOrVarOrTerm>,
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

/// Get tangent
fn get_tangent(expressions: &[MathExpression]) -> Tangent {
    // Check if first expression is an mfrac

    match &expressions[0] {
        MathExpression::Mfrac(numerator, denominator) => {
            if is_leibniz_diff_op(numerator, denominator) {
                let var = mfrac_leibniz_to_var(numerator, denominator);
                return Tangent(var);
            } else {
                panic!("Expression is an mfrac but not a Leibniz diff operator!");
            }
        }
        // Translate MathML :mover as Newton dot notation
        MathExpression::Mover(base, overscript) => {
            // Check if overscript is ˙
            if let MathExpression::Mo(id) = &**overscript {
                if *id == Operator::Other("˙".to_string()) {
                    return Tangent(var_candidate_to_var(base));
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

/// Predicate testing whether a MathML operator (:mo) is a subtraction or addition.
fn is_sum_or_sub(element: &MathExpression) -> bool {
    if let MathExpression::Mo(operator) = element {
        if Operator::Add == *operator || Operator::Subtract == *operator {
            return true;
        }
    }
    return false;
}

/// Predicate testing whether a MathML elm could be interpreted as a Var.
/// TODO: This currently permits :mn -> MathML numerical literals.
///     Perhaps useful to represent constant coefficients?
///     But should those be Vars?
fn is_var_candidate(element: &MathExpression) -> bool {
    match element {
        MathExpression::Mi(x) => true,
        MathExpression::Mn(x) => true,
        MathExpression::Msub(x1, x2) => true,
        _ => false,
    }
}
/// Walk rhs sequence of MathML, identifying subsequences of elms that represent Terms
fn group_rhs(rhs: &[MathExpression]) -> Vec<Vec<&MathExpression>> {
    let mut terms_math_expressions = Vec::<Vec<&MathExpression>>::new();
    let mut current_term = Vec::<&MathExpression>::new();

    for element in rhs.iter() {
        if is_sum_or_sub(element) {
            if current_term.is_empty() {
                current_term.push(element);
            } else {
                terms_math_expressions.push(current_term);
                current_term = vec![element];
            }
        } else if is_var_candidate(element) {
            current_term.push(element);
        } else {
            panic!("Unhandled rhs element {:?}, rhs {:?}", element, rhs);
        }
    }
    if !current_term.is_empty() {
        terms_math_expressions.push(current_term);
    }
    terms_math_expressions
}

/// MathExpression or Var
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
enum MathExpressionOrVarOrTerm {
    MathExpression(MathExpression),
    Var(Var),
    Term(Term),
}

/// Identify elements in grouped rhs elements that are Var candidates and translate them to Vars.
fn rhs_groups_to_vars(
    rhs_groups: Vec<Vec<&MathExpression>>,
) -> (
    Vec<Vec<MathExpressionOrVarOrTerm>>,
    HashSet<MathExpressionOrVarOrTerm>,
) {
    let mut var_set = HashSet::<MathExpressionOrVarOrTerm>::new();
    let mut new_groups = Vec::<Vec<MathExpressionOrVarOrTerm>>::new();

    for group in rhs_groups {
        let mut new_group = Vec::new();
        for element in group {
            if is_var_candidate(element) {
                let var = var_candidate_to_var(element);
                new_group.push(MathExpressionOrVarOrTerm::Var(var.clone()));
                var_set.insert(MathExpressionOrVarOrTerm::Var(var.clone()));
            } else {
                new_group.push(MathExpressionOrVarOrTerm::MathExpression(element.clone()));
            }
            new_groups.push(new_group.clone());
        }
    }
    (new_groups, var_set)
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
        let rhs_groups = group_rhs(rhs);

        let (new_groups, var_set) = rhs_groups_to_vars(rhs_groups);

        // NOTE: Here the rhs represents an intermediate step of having
        //       grouped the rhs into components of terms and translated
        //       var candidates into Vars, but not yet fully translated
        //       into a Tuple of Terms. This intermediate step must be
        //       finished before all equations have been collection, after
        //       which we can identify the tangent Vars that in turn
        //       are the basis for inferring which Vars are species vs rates
        //       Once species and rates are distinguished, then Terms can
        //       be formed.

        let eqn = Eqn {
            lhs: tangent,
            rhs: new_groups,
            rhs_vars: var_set,
        };
        eqn
    } else {
        panic!("Does not look like Eqn: {:?}", ast)
    }
}

/// Translate list of MathML ASTs to EqnDict containing a collection of MAK ODE equations.
fn mathml_asts_to_eqn_dict(mathml_asts: Vec<Math>) -> EqnDict {
    let mut eqns = Vec::<Eqn>::new();
    let mut vars = HashSet::<MathExpressionOrVarOrTerm>::new();
    let mut species = HashSet::<MathExpressionOrVarOrTerm>::new();

    for ast in mathml_asts {
        let eqn = mml_to_eqn(ast);
        vars.extend(eqn.rhs_vars.clone());
        species.insert(MathExpressionOrVarOrTerm::Var(Var::new(&eqn.lhs.0.name)));
        eqns.push(eqn);
    }

    let rates = vars.difference(&species).cloned().collect();
    let mut eqn_dict = HashMap::<Var, Eqn>::new();

    for mut eqn in eqns {
        let mut terms = Vec::new();
        for rhs_group in eqn.rhs.clone() {
            let mut term_rate = Vec::new();
            let mut term_species = Vec::new();
            let mut term_polarity = Polarity::add;

            for element in rhs_group {
                match element {
                    MathExpressionOrVarOrTerm::Var(var) => {
                        let var = MathExpressionOrVarOrTerm::Var(var.clone());
                        if species.contains(&var) {
                            term_species.push(var);
                        } else {
                            term_rate.push(var);
                        }
                    }
                    MathExpressionOrVarOrTerm::MathExpression(expression) => {
                        if let MathExpression::Mo(op) = expression {
                            if op == Operator::Subtract {
                                term_polarity = Polarity::sub;
                            } else if op == Operator::Add {
                                term_polarity = Polarity::add;
                            }
                        }
                    }
                    _ => panic!("Unexpected rhs element: {:?}", element),
                }
                let term = Term {
                    rate: vec![term_rate.clone()],
                    species: vec![term_species.clone()],
                    polarity: term_polarity.clone(),
                };
                terms.push(MathExpressionOrVarOrTerm::Term(term));
                eqn.rhs = vec![terms.clone()];
            }
        }
        eqn_dict.insert(Var::new(&eqn.lhs.0.name), eqn);
    }

    EqnDict {
        eqns: eqn_dict,
        species: species,
        rates: rates,
    }
}

fn export_eqn_dict_json(eqn_dict: EqnDict) {
    dbg!(eqn_dict);
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
    export_eqn_dict_json(eqn_dict)
}
