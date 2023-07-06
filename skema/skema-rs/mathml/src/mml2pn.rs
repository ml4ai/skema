use crate::acset;
pub use crate::acset::ACSet;
use crate::petri_net::{
    recognizers::{get_polarity, get_specie_var, is_add_or_subtract_operator, is_var_candidate},
    Polarity, Rate, Specie, Var,
};
use crate::{
    ast::{
        Math, MathExpression,
        MathExpression::{Mn, Mo},
        Mrow, Operator,
    },
    parsing::parse,
};
use petgraph::Graph;
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
            if let Some('#') = &l.chars().next() {
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
        write!(f, "{}", self.0)
    }
}

/// A product of rate and species that is added or subtracted.
/// THere should just be one rate, but since we're parsing and there could
/// be noise, this accommodates possibly reading several names of things
/// that should be combined into a single rate.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Default)]
pub struct Term {
    pub polarity: Polarity,
    pub species: Vec<Specie>,
    pub vars: Vec<Var>,
}

// MathML to Petri Net algorithm
// - Identify which variables are rates and which ones are species.
//   - Perform one pass over the equations in the ODE system to collect all the variables.
//     - Group the variables on the RHS into terms by splitting the RHS by + and - operators.
//   - The variables on the LHSes are the tangent variables (i.e., the ones whose
//     derivatives are being taken). The variables on the RHSes that correspond to variables on the
//     LHS are species. The rest are rates.
/// Group the variables in the equations by the =, +, and - operators, and collect the variables.
pub fn group_by_operators(
    ast: Math,
    species: &mut HashSet<Var>,
    vars: &mut HashSet<Var>,
    eqns: &mut HashMap<Var, Vec<Term>>,
) {
    let expressions = if ast.content.len() == 1 {
        if let MathExpression::Mrow(Mrow(exprs)) = &ast.content[0] {
            exprs
        } else {
            panic!("Exactly one top-level MathExpression found, but it is not an Mrow! We cannot handle this case.");
        }
    } else {
        &ast.content
    };

    let mut terms = Vec::<Term>::new();
    let mut current_term = Term::default();
    let mut lhs_specie: Option<Var> = None;
    let mut equals_index = 0;

    // Get the index of the equals term
    for (i, expr) in (*expressions).iter().enumerate() {
        if let Mo(Operator::Equals) = expr {
            equals_index = i;
            let lhs = &expressions[0];
            lhs_specie = Some(get_specie_var(lhs));
        }
    }

    // Iterate over MathExpressions in the RHS
    for (_i, expr) in expressions[equals_index + 1..].iter().enumerate() {
        if is_add_or_subtract_operator(expr) {
            if current_term.vars.is_empty() {
                current_term.polarity = get_polarity(expr);
            } else {
                terms.push(current_term);
                current_term = Term {
                    vars: vec![],
                    polarity: get_polarity(expr),
                    ..Default::default()
                };
            }
        } else if is_var_candidate(expr) {
            current_term.vars.push(Var(expr.clone()));
            vars.insert(Var(expr.clone()));
        } else {
            panic!("Unhandled rhs element {:?}", expr);
        }
    }
    if !current_term.vars.is_empty() {
        terms.push(current_term);
    }

    let lhs_specie = lhs_specie.expect("Unable to determine the specie on the LHS!");
    species.insert(lhs_specie.clone());
    eqns.insert(lhs_specie, terms);
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Clone)]
struct Exponent(isize);

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

#[derive(Debug, Clone)]
pub enum PetriNetElement {
    Specie(acset::Specie),
    Transition(acset::ACSetTransition),
}

impl From<Vec<Math>> for acset::ACSet {
    fn from(mathml_asts: Vec<Math>) -> Self {
        let mut specie_vars = HashSet::<Var>::new();
        let mut vars = HashSet::<Var>::new();
        let mut eqns = HashMap::<Var, Vec<Term>>::new();

        for ast in mathml_asts.into_iter() {
            group_by_operators(ast, &mut specie_vars, &mut vars, &mut eqns);
        }

        // Get the rate variables
        let rate_vars: HashSet<&Var> = vars.difference(&specie_vars).collect();

        let mut species = BTreeSet::<Specie>::new();
        let mut monomials = Monomials::default();
        // Initialize coefficient table f(i, y)
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
                if term.polarity == Polarity::Negative {
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
        // We increment indices by 1 wherever necessary in order to facilitate interoperability with Julia.
        let mut acset = acset::ACSet::default();

        // Collect the species for the ACSet
        acset.S = species
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, x)| acset::Specie {
                sname: x.to_string(),
                uid: i,
            })
            .collect();

        // Initialize exponents table e(i, y)
        let mut exponents = Exponents::default();

        for (i, monomial) in monomials.0.iter().enumerate() {
            acset.T.push(acset::ACSetTransition {
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
        acset
    }
}

impl acset::ACSet {
    /// Equation to Petri net algorithm (taken from https://arxiv.org/pdf/2206.03269.pdf)
    ///
    /// M(S) is the set of monomials
    /// m: T -> M(S)
    /// The ODEs corresponding to a Petri net can be written as follows:
    ///     \dot{x_i} = \sum_{y} f(i, y)m(y)
    /// where f(i, y) are integers such that f(i, y) + e(i, y) is a natural number.
    ///
    /// e(i, y): T -> N is a function representing the exponent of species i in monomial corresponding to transition y.
    /// For each transition y, draw e(i, y) arrows from specie x_i to transition y.
    /// Finally, for each transition y, draw n(i, y) = f(i, y) + e(i, y) arrows from y to x_i.
    ///
    /// This function returns a JSON serializable representation of an ACSet for TA2 teams to
    /// consume.
    pub fn from_file(filepath: &str) -> acset::ACSet {
        let mathml_asts = get_mathml_asts_from_file(filepath);
        acset::ACSet::from(mathml_asts)
    }

    /// Construct a graph object from an ACSet
    pub fn to_graph(&self) -> Graph<PetriNetElement, usize> {
        let mut graph = Graph::<PetriNetElement, usize>::new();
        let mut specie_indices = Vec::new();
        let mut transition_indices = Vec::new();
        for specie in &self.S {
            let node_index = graph.add_node(PetriNetElement::Specie(specie.clone()));
            specie_indices.push(node_index);
        }
        for transition in &self.T {
            let node_index = graph.add_node(PetriNetElement::Transition(transition.clone()));
            transition_indices.push(node_index);
        }

        for input_arc in &self.I {
            let source = specie_indices[input_arc.is - 1];
            let destination = transition_indices[input_arc.it - 1];
            if let Some(e) = graph.find_edge(source, destination) {
                graph[e] += 1;
            } else {
                graph.add_edge(source, destination, 1);
            }
        }
        for output_arc in &self.O {
            let source = transition_indices[output_arc.ot - 1];
            let destination = specie_indices[output_arc.os - 1];

            if let Some(e) = graph.find_edge(source, destination) {
                graph[e] += 1;
            } else {
                graph.add_edge(source, destination, 1);
            }
        }
        graph
    }

    /// Construct a graph from an ACSet object and output it to DOT format for visualization and
    /// debugging.
    pub fn to_dot(&self) -> String {
        let graph = self.to_graph();
        let mut dot: String = "".to_owned();
        dot.push_str("digraph {\n");
        for node in graph.node_indices() {
            dot.push_str(&format!("\t {} ", node.index()));
            let (label, shape, color) = {
                match &graph[node] {
                    PetriNetElement::Specie(specie) => (&specie.sname, "circle", "dodgerblue3"),
                    PetriNetElement::Transition(transition) => {
                        (&transition.tname, "square", "darkorange2")
                    }
                }
            };
            dot.push_str(&format!(
                "[ label = \"{}\" , shape = {}, color = {}]\n",
                label, shape, color
            ));
        }

        for edge in graph.edge_indices() {
            let (src, dest) = graph.edge_endpoints(edge).unwrap();
            dot.push_str(&format!(
                "\t {} -> {} [ label = \"{}\" ]\n ",
                src.index(),
                dest.index(),
                graph[edge]
            ));
        }
        dot.push('}');
        dot
    }
}

/// Helper function for testing equality of ACSets, since the order of the edges is not guaranteed
/// to be preserved in roundtrip serialization.
#[cfg(test)]
fn test_acset_equality(mut acset_1: ACSet, mut acset_2: ACSet) -> bool {
    acset_1.I.sort();
    acset_1.O.sort();
    acset_2.O.sort();
    acset_2.I.sort();
    acset_1 == acset_2
}

#[test]
fn test_simple_sir_v1() {
    let acset_1 = acset::ACSet::from_file("../../../data/mml2pn_inputs/simple_sir_v1/mml_list.txt");
    let acset_2: acset::ACSet =
        serde_json::from_str(&std::fs::read_to_string("tests/simple_sir_v1_acset.json").unwrap())
            .unwrap();
    assert!(test_acset_equality(acset_1, acset_2));
}

#[test]
fn test_simple_sir_v2() {
    let acset_1 = acset::ACSet::from_file("../../../data/mml2pn_inputs/simple_sir_v2/mml_list.txt");
    let acset_2: acset::ACSet =
        serde_json::from_str(&std::fs::read_to_string("tests/simple_sir_v2_acset.json").unwrap())
            .unwrap();
    assert!(test_acset_equality(acset_1, acset_2));
}

#[test]
fn test_simple_sir_v3() {
    let acset_1 = acset::ACSet::from_file("../../../data/mml2pn_inputs/simple_sir_v3/mml_list.txt");
    let acset_2: acset::ACSet =
        serde_json::from_str(&std::fs::read_to_string("tests/simple_sir_v3_acset.json").unwrap())
            .unwrap();
    assert!(test_acset_equality(acset_1, acset_2));
}
