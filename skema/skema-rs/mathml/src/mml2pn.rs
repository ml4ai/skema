use crate::petri_net::{
    recognizers::{get_polarity, is_add_or_subtract_operator, is_leibniz_diff_operator},
    PetriNet, Polarity, Specie, Tangent, Transition,
};
use crate::{
    ast::{Math, MathExpression, Operator},
    parsing::parse,
};
use std::collections::{HashMap, HashSet};

/// Representation of a "named" variable
/// Here, 'variable' is intended to mean a symbolic name for a value.
/// Variables could be names of species (states) or rate (parameters).
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Var(MathExpression);

/// A product of rate and species that is added or subtracted.
/// THere should just be one rate, but since we're parsing and there could
/// be noise, this accommodates possibly reading several names of things
/// that should be combined into a single rate.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Default)]
struct Term {
    polarity: Polarity,
    vars: Vec<Var>,
    rates: Vec<Var>,
    species: Vec<Var>,
}

/// A single ODE equation.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct Eqn {
    /// The left-hand-side is the tangent.
    lhs: Tangent,

    /// The right-hand-side of the equation, a sequence of Terms.
    rhs: Vec<Term>,
}

#[derive(Debug, Eq, PartialEq, Default)]
struct TermToEdgesMap {
    inward: Vec<Var>,
    outward: Vec<Var>,
}

/// A collection of Eqns, indexed by the Var of the lhs Tangent.
#[derive(Debug, Eq, PartialEq, Default)]
pub struct EqnDict {
    eqns: HashMap<Var, Eqn>,

    /// The set of all Vars across the eqns that are interpreted as species.
    species: HashSet<Var>,

    /// The set of all Vars across the eqns that are interpreted as rates.
    rates: HashSet<Var>,

    /// Term to equation map
    term_to_eqn_map: HashMap<Term, HashMap<Polarity, Vec<Var>>>,

    /// Term to edges map
    term_to_edges_map: TermToEdgesMap,
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
) -> Specie {
    // Check if numerator is an mrow
    if let MathExpression::Mrow(num_expressions) = &**numerator {
        // We assume here that the numerator is of the form dX where X is the variable of interest.
        if num_expressions.len() == 2 {
            let expression = &num_expressions[1];
            return Specie(expression.clone());
        } else {
            panic!(
                "More than two elements in the numerator, cannot extract specie from Leibniz differential operator!");
        }
    } else {
        panic!("Unable to extract specie from Leibniz differential operator!");
    }
}

/// Get tangent
fn get_tangent(expression: &MathExpression) -> Tangent {
    // Check if expression is an mfrac
    match expression {
        MathExpression::Mfrac(numerator, denominator) => {
            if is_leibniz_diff_operator(numerator, denominator) {
                let specie = mfrac_leibniz_to_specie(numerator, denominator);
                return Tangent(specie);
            } else {
                panic!("Expression is an mfrac but not a Leibniz diff operator!");
            }
        }
        // Translate MathML :mover as Newton dot notation
        MathExpression::Mover(base, overscript) => {
            // Check if overscript is ˙
            if let MathExpression::Mo(id) = &**overscript {
                if *id == Operator::Other("˙".to_string()) {
                    return Tangent(Specie(*base.clone()));
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
fn is_var_candidate(element: &MathExpression) -> bool {
    match element {
        MathExpression::Mi(x) => true,
        MathExpression::Mn(x) => true,
        MathExpression::Msub(x1, x2) => true,
        _ => false,
    }
}

/// MathExpression or Var
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
enum MathExpressionOrVarOrTerm {
    MathExpression(MathExpression),
    Var(Var),
    Term(Term),
}

/// Walk rhs sequence of MathML, identifying subsequences of elms that represent Terms
fn group_rhs(rhs: &[MathExpression]) -> (Vec<Term>, HashSet<Var>) {
    let mut terms = Vec::<Term>::new();
    let mut current_term = Term::default();
    let mut vars = HashSet::<Var>::new();

    for element in rhs.iter() {
        if is_add_or_subtract_operator(element) {
            if current_term.vars.is_empty() {
                current_term.polarity = get_polarity(element);
            } else {
                terms.push(current_term);
                current_term = Term {
                    vars: vec![],
                    ..Default::default()
                };
            }
        } else if is_var_candidate(element) {
            current_term.vars.push(Var(element.clone()));
            vars.insert(Var(element.clone()));
        } else {
            panic!("Unhandled rhs element {:?}, rhs {:?}", element, rhs);
        }
    }
    if !current_term.vars.is_empty() {
        terms.push(current_term);
    }
    (terms, vars)
}

/// Converts a MathML AST to an Eqn
fn mml_to_eqn(ast: Math) -> (Eqn, HashSet<Var>) {
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

        if equals_index != 1 {
            panic!("We do not handle the case where there is more than one term on the LHS of an equation!");
        }
        let lhs = &expr[0];
        let rhs = &expr[equals_index + 1..];

        let tangent = get_tangent(lhs);
        let (terms, vars) = group_rhs(rhs);

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
            rhs: terms,
        };
        (eqn, vars)
    } else {
        panic!("Does not look like Eqn: {:?}", ast)
    }
}

/// Translate list of MathML ASTs to EqnDict containing a collection of MAK ODE equations.
pub fn mathml_asts_to_eqn_dict(mathml_asts: Vec<Math>) -> EqnDict {
    let mut eqns = Vec::<Eqn>::new();
    let mut vars = HashSet::<Var>::new();
    let mut species = HashSet::<Var>::new();
    for ast in mathml_asts {
        let (eqn, rhs_vars) = mml_to_eqn(ast);
        vars.extend(rhs_vars.clone());
        species.insert(Var(eqn.clone().lhs.0 .0));
        eqns.push(eqn.clone());
    }

    let rates = vars.difference(&species).cloned().collect();

    for mut eqn in &mut eqns {
        for mut term in &mut eqn.rhs {
            let mut rate_vars = Vec::<Var>::new();
            let mut specie_vars = Vec::<Var>::new();

            for var in term.vars.clone() {
                if species.contains(&var) {
                    &specie_vars.push(var);
                } else {
                    &rate_vars.push(var);
                }
            }
            term.species.append(&mut specie_vars);
            term.rates.append(&mut rate_vars);
        }
    }

    let mut eqn_dict = HashMap::<Var, Eqn>::new();
    let mut term_to_eqn_map = HashMap::<Term, HashMap<Polarity, Vec<Var>>>::new();

    for eqn in &mut eqns {
        eqn_dict.insert(Var(eqn.clone().lhs.0 .0), eqn.clone());

        // Link terms to equations
        for term in &mut eqn.rhs {
            if term_to_eqn_map.contains_key(&term) {
                term_to_eqn_map
                    .entry(term.clone())
                    .or_insert(HashMap::from([(
                        term.polarity.clone(),
                        vec![Var(eqn.lhs.0 .0.clone())],
                    )]));
                //.unwrap()
                //.push(Var(eqn.lhs.0 .0.clone()));
            } else {
                let var = Var(eqn.lhs.0 .0.clone());
                let polarity_dict = HashMap::from([(term.polarity.clone(), vec![var])]);
                term_to_eqn_map.insert(term.clone(), polarity_dict);
            }
            //dbg!(&term_to_eqn_map);
            //term_to_eqn_map
            //.entry(term.clone())
            //.and_modify(|e| {
            //*e = HashMap::from([(term.polarity.clone(), vec![Var(eqn.lhs.0 .0.clone())])])
            //})
            //.or_insert(HashMap::from([(
            //term.polarity.clone(),
            //vec![Var(eqn.clone().lhs.0 .0)],
            //)]));
        }
    }

    EqnDict {
        eqns: eqn_dict,
        species: species,
        rates: rates,
        term_to_eqn_map: term_to_eqn_map,
        ..Default::default()
    }
}

pub fn wire_pn(eqn_dict: &mut EqnDict) {
    println!("Wiring PN");
    for (term, in_out_flow_dict) in eqn_dict.term_to_eqn_map.iter() {
        let mut in_list = Vec::<Var>::new();
        let mut out_list = Vec::<Var>::new();
        for var in &term.vars {
            in_list.push(var.clone());
            if !in_out_flow_dict[&Polarity::sub].contains(&var) {
                out_list.push(var.clone());
            }
        }

        for var in &in_out_flow_dict[&Polarity::add] {
            out_list.push(var.clone());
        }

        eqn_dict.term_to_edges_map = TermToEdgesMap {
            inward: in_list,
            outward: out_list,
        }
    }
}

pub fn export_eqn_dict_json(eqn_dict: &mut EqnDict) {
    wire_pn(eqn_dict);
    //dbg!(&eqn_dict);
}
