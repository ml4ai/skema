use crate::petri_net::{
    recognizers::{get_polarity, get_specie_var, is_add_or_subtract_operator, is_var_candidate},
    Polarity, Rate, Specie, Var,
};
use crate::{
    acset,
    ast::{
        Math,
        MathExpression::{Mn, Mo, Mrow},
        Operator,
    },
    parsing::parse,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::{
    fs::File,
    io::{self, BufRead},
};

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
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.to_string())
    }
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

// MathML to Petri Net algorithm
// - Identify which variables are rates and which ones are species.
//   - Perform one pass over the equations in the ODE system to collect all the variables.
//     - Group the variables on the RHS into terms by splitting the RHS by + and - operators.
//   - The variables on the LHSes are the tangent variables (i.e., the ones whose
//     derivatives are being taken). The variables on the RHSes that correspond to variables on the
//     LHS are species. The rest are rates.
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

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Clone)]
struct Exponent(isize);

/// A transition in a Petri net
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Transition(usize);

/// Coefficient of monomials in an ODE corresponding to a Petri net.
/// For example, in the following equation: \dot{S} = -βSI,
/// The coefficient of the βSI monomial on the RHS is (-1).
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Clone, Default)]
struct Coefficient(isize);

impl fmt::Display for Coefficient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Default, Ord, PartialOrd)]
struct Monomial((Rate, BTreeMap<Specie, Exponent>));

impl Monomial {
    fn new(vars: HashSet<Var>) -> Monomial {
        let mut m = Monomial::default();
        m.0 .0 = Rate(Mn("1".to_string()));
        for var in vars {
            m.0 .1.insert(Specie(var.0), Exponent(0));
        }
        m
    }
}

impl fmt::Display for Exponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.0 .0);
        for (specie, exponent) in &self.0 .1 {
            write!(f, " {}^{} ", specie, exponent);
        }
        write!(f, ")");
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Default, Ord, PartialOrd)]
struct Monomials(BTreeSet<Monomial>);

#[derive(Debug, Eq, PartialEq, Hash, Clone, Default, Ord, PartialOrd)]
struct Coefficients(BTreeMap<Specie, BTreeMap<Monomial, Coefficient>>);

#[derive(Debug, Eq, PartialEq, Hash, Clone, Default, Ord, PartialOrd)]
struct Exponents(BTreeMap<Specie, BTreeMap<Monomial, Exponent>>);

// Equation to Petri net algorithm (taken from https://arxiv.org/pdf/2206.03269.pdf)
//
// M(S) is the set of monomials
// m: T -> M(S)
// \dot{x_i} = \sum_{y} f(i, y)m(y)
// f(i, y) are integers such that f(i, y) + e(i, y) is a natural number.
// e(i, y): T -> N --- exponent of species i in monomial corresponding to transition y.
// For each transition y, draw e(i, y) arrows from specie x_i to transition y.
// Finally, for each transition y, draw n(i, y) = f(i, y) + e(i, y) arrows from y to x_i
// Algorithm:
// - Identify monomials (i.e. products of species and rates). I assume each monomial corresponds to
//   a transition.
// - When a monomial is identified, get the exponents of the species in it and store it in a data
//   structure.
#[test]
fn test_simple_sir_v1() {
    let mathml_asts = get_mathml_asts_from_file("../../mml2pn/mml/simple_sir_v1/mml_list.txt");
    let mut specie_vars = HashSet::<Var>::new();
    let mut vars = HashSet::<Var>::new();
    let mut eqns = HashMap::<Var, Vec<Term>>::new();

    for ast in mathml_asts.into_iter() {
        let _ = group_by_operators(ast, &mut specie_vars, &mut vars, &mut eqns);
    }
    let rate_vars: HashSet<&Var> = vars.difference(&specie_vars).collect();

    let mut species = BTreeSet::<Specie>::new();
    let mut monomials = Monomials::default();
    // Construct exponents table e(i, y) and coefficient table f(i, y)
    let mut exponents = Exponents::default();
    let mut coefficients = Coefficients::default();

    for (lhs_specie, terms) in eqns {
        for term in terms {
            let mut monomial = Monomial::new(specie_vars.clone());
            for var in term.vars.clone() {
                if rate_vars.contains(&var) {
                    monomial.0 .0 = Rate(var.0);
                } else {
                    // TODO: Generalize this to when the coefficients aren't just 1 and -1.
                    monomial.0 .1.insert(Specie(var.0), Exponent(1));
                }
            }
            let mut coefficient = Coefficient(1);
            if term.polarity == Polarity::sub {
                coefficient.0 = -1;
            }

            if !monomials.0.contains(&monomial) {
                monomials.0.insert(monomial.clone());
            }

            let specie = Specie(lhs_specie.0.clone());
            species.insert(specie.clone());

            coefficients
                .0
                .entry(specie.clone())
                .and_modify(|e| {
                    e.entry(monomial.clone()).or_insert(coefficient.clone());
                })
                .or_insert(BTreeMap::from([(monomial.clone(), coefficient.clone())]));
        }
    }

    // Construct the ACSet for TA2
    let mut acset = acset::ACSet::default();
    acset.S = species
        .clone()
        .into_iter()
        .enumerate()
        .map(|(i, x)| acset::Specie {
            sname: x.to_string(),
            uid: i,
        })
        .collect();
    for (i, monomial) in monomials.0.iter().enumerate() {
        acset.T.push(acset::Transition {
            tname: monomial.0 .0.to_string(),
        });
        for (specie, exponent) in monomial.0 .1.clone() {
            let n_arrows = exponent.0;
            for _ in 0..n_arrows {
                acset.I.push(acset::InputArc {
                    it: i + 1,
                    is: species.iter().position(|x| x == &specie).unwrap() + 1,
                });
            }
            exponents
                .0
                .entry(specie)
                .or_insert(BTreeMap::from([(monomial.clone(), exponent.clone())]));
        }
    }

    for specie_var in specie_vars {
        let specie = Specie(specie_var.0);
        for (i, monomial) in monomials.0.iter().enumerate() {
            let coefficient = coefficients
                .0
                .get_mut(&specie)
                .unwrap()
                .entry(monomial.clone())
                .or_insert(Coefficient(0))
                .0;

            let exponent = exponents
                .0
                .entry(specie.clone())
                .or_default()
                .entry(monomial.clone())
                .or_insert(Exponent(0));

            let narrows = coefficient + exponent.0;
            for _ in 0..narrows {
                acset.O.push(acset::OutputArc {
                    ot: i + 1,
                    os: species.iter().position(|x| x == &specie).unwrap() + 1,
                });
            }
        }
    }

    println!("{}", serde_json::to_string(&acset).unwrap());
}
