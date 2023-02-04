use crate::petri_net::{
    recognizers::{get_polarity, is_add_or_subtract_operator, is_leibniz_diff_operator},
    PetriNet, Polarity, Rate, Specie, Tangent, Transition,
};
use crate::{
    ast::{
        Math, MathExpression,
        MathExpression::{Mfrac, Mi, Mn, Mo, Mover, Mrow, Msub, Munder},
        Operator,
    },
    parsing::parse,
};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::{
    fs::File,
    io::{self, BufRead},
};

// MathML to Petri Net algorithm
// - Identify which variables are rates and which ones are species.
//   - Perform one pass over the equations in the ODE system to collect all the variables.
//     - Group the variables on the RHS into terms by splitting the RHS by + and - operators.
//   - The variables on the LHSes are the tangent variables (i.e., the ones whose
//     derivatives are being taken). The variables on the RHSes that correspond to variables on the
//     LHS are species. The rest are rates.

/// Representation of a "named" variable
/// Here, 'variable' is intended to mean a symbolic name for a value.
/// Variables could be names of species (states) or rate (parameters).
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Var(MathExpression);

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Exponent(i32);

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.to_string())
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Eqn {
    lhs_specie: Var,
    terms: Vec<Term>,
}

/// A product of rate and species that is added or subtracted.
/// THere should just be one rate, but since we're parsing and there could
/// be noise, this accommodates possibly reading several names of things
/// that should be combined into a single rate.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Default)]
struct Term {
    polarity: Polarity,
    species: Vec<Specie>,
    vars: Vec<Var>,
}

//#[derive(Debug, Eq, PartialEq, Default)]
//struct TermToEdgesMap {
//inward: Vec<Var>,
//outward: Vec<Var>,
//}

///// The set of all Vars across the eqns that are interpreted as species.
//species: HashSet<Var>,

///// The set of all Vars across the eqns that are interpreted as rates.
//rates: HashSet<Var>,

///// Term to equation map
//term_to_eqn_map: HashMap<Term, HashMap<Polarity, Vec<Var>>>,

///// Term to edges map
//term_to_edges_map: TermToEdgesMap,
//}

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

/// Get tangent
fn get_specie_var(expression: &MathExpression) -> Var {
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
fn is_var_candidate(element: &MathExpression) -> bool {
    match element {
        Mi(x) => true,
        Mn(x) => true,
        Msub(x1, x2) => true,
        _ => false,
    }
}

// Walk rhs sequence of MathML, identifying subsequences of elms that represent Terms

//// Link terms to equations
//for term in &eqn.rhs {
//term_to_eqn_map
//.entry(term.clone())
//.and_modify(|e1| {
//e1.entry(term.polarity.clone()).and_modify(|e2| {
//e2.push(Var(eqn.lhs.0 .0.clone()));
//});
//})
//.or_insert_with_key(|k| {
//let other_polarity = if k.polarity == Polarity::sub {
//Polarity::add
//} else {
//Polarity::sub
//};
//HashMap::from([
//(k.polarity.clone(), vec![Var(eqn.clone().lhs.0 .0)]),
//(other_polarity, vec![]),
//])
//});
//}
//}

//EqnDict {
//eqns: eqn_dict,
//species: species,
//rates: rates,
//term_to_eqn_map: term_to_eqn_map,
//..Default::default()
//}
//}

//pub fn wire_pn(eqn_dict: &mut EqnDict) {
//println!("Wiring PN");
//for (term, in_out_flow_dict) in eqn_dict.term_to_eqn_map.iter() {
//let mut in_list = Vec::<Var>::new();
//let mut out_list = Vec::<Var>::new();
//for specie in &term.species {
//in_list.push(specie.clone());
//if !in_out_flow_dict[&Polarity::sub].contains(specie) {
//out_list.push(specie.clone());
//}
//}

//for specie in &in_out_flow_dict[&Polarity::add] {
//out_list.push(specie.clone());
//}

//eqn_dict.term_to_edges_map = TermToEdgesMap {
//inward: in_list,
//outward: out_list,
//};
//}
//dbg!(&eqn_dict.term_to_edges_map);
//}

pub fn get_mathml_asts_from_file(filepath: &str) -> Vec<Math> {
    let f = File::open(filepath).unwrap();
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
    mathml_asts

    //let mut eqn_dict = mathml_asts_to_eqn_dict(mathml_asts);
    //export_eqn_dict_json(&mut eqn_dict)
}

/// Group the variables in the equations by the =, +, and - operators, and collect the variables.
fn group_by_operators(
    ast: Math,
    species: &mut HashSet<Var>,
    vars: &mut HashSet<Var>,
    eqns: &mut HashMap<Var, Vec<Term>>,
) {
    // Check if there is exactly one element in the AST
    if ast.content.len() != 1 {
        panic!("We cannot handle expressions with more than one top-level MathExpression yet!");
    }

    let mut terms = Vec::<Term>::new();
    let mut current_term = Term::default();
    let mut lhs_specie: Option<Var> = None;

    let mut equals_index = 0;
    // Check if the first element is an mrow
    if let Mrow(expr_1) = &ast.content[0] {
        // Get the index of the equals term
        for (i, expr_2) in (*expr_1).iter().enumerate() {
            if let Mo(Operator::Equals) = expr_2 {
                equals_index = i;
                let lhs = &expr_1[0];
                lhs_specie = Some(get_specie_var(lhs));
            }
        }

        // Iterate over MathExpressions in the RHS
        for (i, expr_2) in expr_1[equals_index + 1..].iter().enumerate() {
            if is_add_or_subtract_operator(expr_2) {
                if current_term.vars.is_empty() {
                    current_term.polarity = get_polarity(expr_2);
                } else {
                    terms.push(current_term);
                    current_term = Term {
                        vars: vec![],
                        polarity: get_polarity(expr_2),
                        ..Default::default()
                    };
                }
            } else if is_var_candidate(expr_2) {
                current_term.vars.push(Var(expr_2.clone()));
                vars.insert(Var(expr_2.clone()));
            } else {
                panic!("Unhandled rhs element {:?}", expr_2);
            }
        }
        if !current_term.vars.is_empty() {
            terms.push(current_term);
        }
    }

    let lhs_specie = lhs_specie.expect("Unable to determine the specie on the LHS!");
    species.insert(lhs_specie.clone());
    eqns.insert(lhs_specie, terms);
}

// M(S) is the set of monomials
// m: T -> M(S)
// \dot{x_i} = \sum_{y} f(x_i, y)m(y)
// e(i, y): T -> N --- exponent of species i in monomial corresponding to transition y.
// For each transition y, draw e(i, y) arrows from specie x_i to transition y.
// Finally, for each transition y, draw n_i(y) = f(x_i) + e(i, y) arrows from y to x_i
// Algorithm:
// - Identify monomials (i.e. products of species). Each monomial corresponds to a transition (TODO: Check if this is right)
// - When a monomial is identified, get the exponents of the species in it and store it in a data
//  structure.
#[test]
fn test_simple_sir_v1() {
    let mathml_asts = get_mathml_asts_from_file("../../mml2pn/mml/simple_sir_v1/mml_list.txt");
    let mut species = HashSet::<Var>::new();
    let mut vars = HashSet::<Var>::new();
    let mut eqns = HashMap::<Var, Vec<Term>>::new();
    let f = HashMap::<(Var, String), Var>::new();
    for (i, ast) in mathml_asts.into_iter().enumerate() {
        let terms = group_by_operators(ast, &mut species, &mut vars, &mut eqns);
    }
    let rates: HashSet<&Var> = vars.difference(&species).collect();

    let mut monomials = HashSet::<HashMap<Specie, Exponent>>::new();
    let mut exponents = HashMap::<Specie, HashMap<Transition, i32>>::new();
    let mut term_to_rate_map = HashMap::<Term, Rate>::new();

    for (lhs_specie, terms) in eqns.iter() {
        for mut term in terms {
            let rate: Option<Rate> = None;
            let mut monomial = Monomial::default();
            for var in term.vars.clone() {
                if rates.contains(&var) {
                    let rate = Some(Rate(var.0));
                } else {
                    monomial.0.insert(Specie(var.0), Exponent(1));
                }
            }
            term_to_rate_map.insert(
                term.clone(),
                rate.expect(&format!("Unable to find rate in term {:?}", term)),
            );
            monomials.insert(&monomial);
        }
    }
}

//pub fn export_eqn_dict_json(eqn_dict: &mut EqnDict) {
//wire_pn(eqn_dict);
//dbg!(&eqn_dict);
//}
