use crate::ast::operator::Operator::{Add, Divide, Multiply, Subtract};
use crate::parsers::math_expression_tree::MathExpressionTree::Atom;
use crate::parsers::math_expression_tree::MathExpressionTree::Cons;
use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, MathExpression, Type,
    },
    parsers::{
        generic_mathml::{attribute, equals, etag, stag, ws, IResult, Span},
        interpreted_mathml::{
            ci_univariate_func, ci_unknown, first_order_derivative_leibniz_notation,
            math_expression, newtonian_derivative, operator,
        },
        math_expression_tree::MathExpressionTree,
    },
};

use derive_new::new;

use nom::error::context;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::map,
    multi::{many0, many1},
    sequence::{delimited, tuple},
};

use std::fs::File;
use std::io::{BufRead, BufReader};
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
pub fn first_order_ode(input: Span) -> IResult<FirstOrderODE> {
    let (s, _) = context("MISSING STARTING <math> TAG.", stag!("math"))(input)?;

    // Recognize LHS derivative
    let (s, (_, ci)) = context(
        "INVALID LHS DERIVATIVE.",
        alt((
            first_order_derivative_leibniz_notation,
            newtonian_derivative,
        )),
    )(s)?;

    // Recognize equals sign
    let (s, _) = context(
        "MISSING EQUALS SIGN.",
        delimited(stag!("mo"), equals, etag!("mo")),
    )(s)?;

    // Recognize other tokens
    let (s, remaining_tokens) = context(
        "INVALID RHS.",
        many1(alt((
            map(ci_univariate_func, MathExpression::Ci),
            map(ci_unknown, |Ci { content, .. }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Function),
                    content,
                })
            }),
            map(operator, MathExpression::Mo),
            math_expression,
        ))),
    )(s)?;

    let (s, _) = context("INVALID ENDING MATH TAG", etag!("math"))(s)?;

    let ode = FirstOrderODE {
        lhs_var: ci,
        rhs: MathExpressionTree::from(remaining_tokens),
    };

    Ok((s, ode))
}

impl FirstOrderODE {
    pub fn to_cmml(&self) -> String {
        let lhs_expression_tree = MathExpressionTree::Cons(
            Operator::Derivative(Derivative::new(1, 1)),
            vec![MathExpressionTree::Atom(MathExpression::Ci(
                self.lhs_var.clone(),
            ))],
        );
        let combined = MathExpressionTree::Cons(
            Operator::Equals,
            vec![lhs_expression_tree, self.rhs.clone()],
        );
        combined.to_cmml()
    }
}

impl FromStr for FirstOrderODE {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        first_order_ode(s.into())
            .map(|(_, ode)| ode)
            .map_err(|err| err.to_string())
    }
}

//--------------------------------------
// Methods for extraction of PN AMR from ODE's
//--------------------------------------
#[allow(non_snake_case)]
pub fn get_FirstOrderODE_vec_from_file(filepath: &str) -> Vec<FirstOrderODE> {
    let f = File::open(filepath).unwrap();
    let lines = BufReader::new(f).lines();

    let mut ode_vec = Vec::<FirstOrderODE>::new();

    for line in lines.flatten() {
        if let Some('#') = &line.chars().next() {
            // Ignore lines starting with '#'
        } else {
            // Parse MathML into FirstOrderODE
            let mut ode = line
                .parse::<FirstOrderODE>()
                .unwrap_or_else(|_| panic!("Unable to parse line {}!", line));
            ode.rhs = flatten_mults(ode.rhs.clone());
            ode_vec.push(ode);
        }
    }
    ode_vec
}

// this struct is for representing terms in an ODE system on equations
#[derive(Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub struct PnTerm {
    pub dyn_state: String,
    pub exp_states: Vec<String>, // list of state variables in term
    pub polarity: bool,          // polarity of term
    pub expression: String,      // content mathml for the expression
    pub parameters: Vec<String>, // list of parameters in term
}

// refactored
// this function takes in one ode equations and returns a vector of the terms in it
pub fn get_terms(sys_states: Vec<String>, ode: FirstOrderODE) -> Vec<PnTerm> {
    let mut terms = Vec::<PnTerm>::new();
    let _exp_states = Vec::<String>::new();
    let _parameters = Vec::<String>::new();

    let dyn_state = ode.lhs_var.to_string();

    match ode.rhs {
        Cons(ref x, ref y) => match &x {
            Multiply => {
                let mut temp_term = get_term_mult(sys_states, y.clone());
                temp_term.dyn_state = dyn_state;
                terms.push(temp_term);
            }
            Divide => {
                let mut temp_term = get_term_div(sys_states, y.clone());
                temp_term.dyn_state = dyn_state;
                terms.push(temp_term);
            }
            Add => {
                let temp_terms = get_terms_add(sys_states, y.clone());
                for term in temp_terms.iter() {
                    let mut t_term = term.clone();
                    t_term.dyn_state = dyn_state.clone();
                    terms.push(t_term.clone());
                }
            }
            _Sub => {
                let temp_terms = get_terms_sub(sys_states, y.clone());
                for term in temp_terms.iter() {
                    let mut t_term = term.clone();
                    t_term.dyn_state = dyn_state.clone();
                    terms.push(t_term.clone());
                }
            }
        },
        Atom(_x) => {
            println!("Warning unexpected RHS structure")
        }
    }
    terms
}

// this takes in the arguments of a closer to root level add operator and returns the PnTerms for it's subgraphs
// we do expect at most multiplication, subtraction, division, or addition
pub fn get_terms_add(sys_states: Vec<String>, eq: Vec<MathExpressionTree>) -> Vec<PnTerm> {
    let mut terms = Vec::<PnTerm>::new();

    /* found multiple terms */

    for arg in eq.iter() {
        match &arg {
            Cons(x1, ref y1) => match x1 {
                Multiply => {
                    let temp_term = get_term_mult(sys_states.clone(), y1.clone());
                    terms.push(temp_term);
                }
                Divide => {
                    let temp_term = get_term_div(sys_states.clone(), y1.clone());
                    terms.push(temp_term);
                }
                Subtract => {
                    let temp_terms = get_terms_sub(sys_states.clone(), y1.clone());
                    for term in temp_terms.iter() {
                        terms.push(term.clone());
                    }
                }
                Add => {
                    let temp_terms = get_terms_add(sys_states.clone(), y1.clone());
                    for term in temp_terms.iter() {
                        terms.push(term.clone());
                    }
                }
                _ => {
                    println!("Error unsupported operation")
                }
            },
            Atom(_x1) => {
                println!("Not valid term for PN")
            }
        }
    }
    terms
}

// this takes in the arguments of a closer to root level sub operator and returns the PnTerms for it's subgraphs
// we do expect at most multiplication, subtraction, division, or addition
pub fn get_terms_sub(sys_states: Vec<String>, eq: Vec<MathExpressionTree>) -> Vec<PnTerm> {
    let mut terms = Vec::<PnTerm>::new();

    /* found multiple terms */
    /* similar to get_terms_add, but need to swap polarity on term from second arg
    and handle unary subtraction too */

    let arg_len = eq.len();

    // if unary subtraction
    if arg_len == 1 {
        match &eq[0] {
            Cons(x1, ref y1) => match x1 {
                Multiply => {
                    let mut temp_term = get_term_mult(sys_states, y1.clone());
                    if temp_term.polarity {
                        temp_term.polarity = false;
                    } else {
                        temp_term.polarity = true;
                    }
                    terms.push(temp_term);
                }
                Divide => {
                    let mut temp_term = get_term_div(sys_states, y1.clone());
                    if temp_term.polarity {
                        temp_term.polarity = false;
                    } else {
                        temp_term.polarity = true;
                    }
                    terms.push(temp_term);
                }
                _Sub => {
                    let temp_terms = get_terms_sub(sys_states, y1.clone());
                    for term in temp_terms.iter() {
                        // swap polarity of temp term
                        let mut t_term = term.clone();
                        if t_term.polarity {
                            t_term.polarity = false;
                        } else {
                            t_term.polarity = true;
                        }
                        terms.push(t_term.clone());
                    }
                }
                Add => {
                    let temp_terms = get_terms_add(sys_states.clone(), y1.clone());
                    for term in temp_terms.iter() {
                        // swap polarity of temp term
                        let mut t_term = term.clone();
                        if t_term.polarity {
                            t_term.polarity = false;
                        } else {
                            t_term.polarity = true;
                        }
                        terms.push(t_term.clone());
                    }
                }
                _ => {
                    println!("Not valid term for PN")
                }
            },
            Atom(_x1) => {
                println!("Not valid term for PN")
            }
        }
    } else {
        // need to treat second term with polarity swap
        for (i, arg) in eq.iter().enumerate() {
            match &arg {
                Cons(x1, ref y1) => match x1 {
                    Multiply => {
                        let mut temp_term = get_term_mult(sys_states.clone(), y1.clone());
                        if i == 1 {
                            // swap polarity of temp term
                            if temp_term.polarity {
                                temp_term.polarity = false;
                            } else {
                                temp_term.polarity = true;
                            }
                            terms.push(temp_term);
                        } else {
                            terms.push(temp_term);
                        }
                    }
                    Divide => {
                        let mut temp_term = get_term_div(sys_states.clone(), y1.clone());
                        if i == 1 {
                            // swap polarity of temp term
                            if temp_term.polarity {
                                temp_term.polarity = false;
                            } else {
                                temp_term.polarity = true;
                            }
                            terms.push(temp_term);
                        } else {
                            terms.push(temp_term);
                        }
                    }
                    Subtract => {
                        let temp_terms = get_terms_sub(sys_states.clone(), y1.clone());
                        for term in temp_terms.iter() {
                            let mut t_term = term.clone();
                            if i == 1 {
                                // swap polarity of temp term
                                if t_term.polarity {
                                    t_term.polarity = false;
                                } else {
                                    t_term.polarity = true;
                                }
                                terms.push(t_term.clone());
                            } else {
                                terms.push(t_term.clone());
                            }
                        }
                    }
                    Add => {
                        let temp_terms = get_terms_add(sys_states.clone(), y1.clone());
                        for term in temp_terms.iter() {
                            let mut t_term = term.clone();
                            if i == 1 {
                                // swap polarity of temp term
                                if t_term.polarity {
                                    t_term.polarity = false;
                                } else {
                                    t_term.polarity = true;
                                }
                                terms.push(t_term.clone());
                            } else {
                                terms.push(t_term.clone());
                            }
                        }
                    }
                    _ => {
                        println!("Error unsupported operation")
                    }
                },
                Atom(_x1) => {
                    println!("Not valid term for PN")
                }
            }
        }
    }
    terms
}

// this takes in the arguments of a div operator and returns the PnTerm for it
// we do expect at most multiplication, subtraction, or addition
pub fn get_term_div(sys_states: Vec<String>, eq: Vec<MathExpressionTree>) -> PnTerm {
    let mut variables = Vec::<String>::new();
    let mut exp_states = Vec::<String>::new();
    let mut polarity = true;

    // this walks the tree and composes a vector of all variable and polarity changes
    for obj in eq.iter() {
        match obj {
            Cons(x, y) => {
                match &x {
                    Subtract => {
                        if y.len() == 1 {
                            polarity = false;
                            variables.push(y[0].to_string());
                        } else {
                            for var in y.iter() {
                                variables.push(var.to_string().clone());
                            }
                        }
                    }
                    Multiply => {
                        // call mult function to get a partial term
                        let mut temp_term = get_term_mult(sys_states.clone(), y.clone());

                        // parse term polarity
                        polarity = temp_term.polarity;

                        // parse term parameters and expression states
                        // need to do both to populate both later
                        variables.append(&mut temp_term.parameters);
                        variables.append(&mut temp_term.exp_states);
                    }
                    Add => {
                        if y.len() == 1 {
                            // really should need to support unary addition, but oh well
                            variables.push(y[0].to_string());
                        } else {
                            for var in y.iter() {
                                variables.push(var.to_string().clone());
                            }
                        }
                    }
                    _ => {
                        println!("Not expected operation inside Multiply")
                    }
                }
            }
            Atom(x) => variables.push(x.to_string()),
        }
    }

    // this compiles the vector of expression states for the term
    let mut ind = Vec::<usize>::new();
    for (i, var) in variables.iter().enumerate() {
        for sys_var in sys_states.iter() {
            if var == sys_var {
                exp_states.push(var.clone());
                ind.push(i);
            }
        }
    }

    // this removes the expression states from the variable vector
    for i in ind.iter().rev() {
        variables.remove(*i);
    }

    PnTerm {
        dyn_state: "temp".to_string(),
        exp_states,
        polarity,
        expression: MathExpressionTree::Cons(Multiply, eq).to_cmml(),
        parameters: variables,
    }
}

// this takes in the arguments of a multiply operator and returns the PnTerm for it
// we do expect at most division, subtraction, or addition
pub fn get_term_mult(sys_states: Vec<String>, eq: Vec<MathExpressionTree>) -> PnTerm {
    let mut variables = Vec::<String>::new();
    let mut exp_states = Vec::<String>::new();
    let mut polarity = true;

    // this walks the tree and composes a vector of all variable and polarity changes
    for obj in eq.iter() {
        match obj {
            Cons(x, y) => {
                match &x {
                    Subtract => {
                        if y.len() == 1 {
                            polarity = false;
                            variables.push(y[0].to_string());
                        } else {
                            for var in y.iter() {
                                variables.push(var.to_string().clone());
                            }
                        }
                    }
                    Divide => {
                        // call mult function to get a partial term
                        let mut temp_term = get_term_div(sys_states.clone(), y.clone());

                        // parse term polarity
                        polarity = temp_term.polarity;

                        // parse term parameters and expression states
                        // need to do both to populate both later
                        variables.append(&mut temp_term.parameters);
                        variables.append(&mut temp_term.exp_states);
                    }
                    Add => {
                        if y.len() == 1 {
                            // really should need to support unary addition, but oh well
                            variables.push(y[0].to_string());
                        } else {
                            for var in y.iter() {
                                variables.push(var.to_string().clone());
                            }
                        }
                    }
                    _ => {
                        println!("Not expected operation inside Multiply")
                    }
                }
            }
            Atom(x) => variables.push(x.to_string()),
        }
    }

    // this compiles the vector of expression states for the term
    let mut ind = Vec::<usize>::new();
    for (i, var) in variables.iter().enumerate() {
        for sys_var in sys_states.iter() {
            if var == sys_var {
                exp_states.push(var.clone());
                ind.push(i);
            }
        }
    }

    // this removes the expression states from the variable vector
    for i in ind.iter().rev() {
        variables.remove(*i);
    }

    PnTerm {
        dyn_state: "temp".to_string(),
        exp_states,
        polarity,
        expression: MathExpressionTree::Cons(Multiply, eq).to_cmml(),
        parameters: variables,
    }
}

pub fn flatten_mults(mut equation: MathExpressionTree) -> MathExpressionTree {
    match equation {
        Cons(ref x, ref mut y) => match x {
            Multiply => {
                match y[1].clone() {
                    Cons(x1, y1) => match x1 {
                        Multiply => {
                            let temp1 = flatten_mults(y1[0].clone());
                            let temp2 = flatten_mults(y1[1].clone());
                            y.remove(1);
                            y.append(&mut [temp1, temp2].to_vec())
                        }
                        _ => {
                            let temp1 = y[1].clone();
                            y.remove(1);
                            y.append(&mut [temp1].to_vec())
                        }
                    },
                    Atom(_x1) => {}
                }
                match y[0].clone() {
                    Cons(x0, y0) => match x0 {
                        Multiply => {
                            let temp1 = flatten_mults(y0[0].clone());
                            let temp2 = flatten_mults(y0[1].clone());
                            y.remove(0);
                            y.append(&mut [temp1, temp2].to_vec());
                        }
                        _ => {
                            let temp1 = y[0].clone();
                            y.remove(0);
                            y.append(&mut [temp1].to_vec())
                        }
                    },
                    Atom(_x0) => {}
                }
            }
            _ => {
                if y.len() > 1 {
                    let temp1 = flatten_mults(y[1].clone());
                    let temp0 = flatten_mults(y[0].clone());
                    y.remove(1);
                    y.remove(0);
                    y.append(&mut [temp0, temp1].to_vec())
                } else {
                    let temp0 = flatten_mults(y[0].clone());
                    y.remove(0);
                    y.append(&mut [temp0].to_vec())
                }
            }
        },
        Atom(ref _x) => {}
    }
    equation
}

#[test]
fn test_ci_univariate_func() {
    test_parser(
        "<mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>",
        ci_univariate_func,
        Ci::new(
            Some(Type::Function),
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
        (
            Derivative::new(1, 1),
            Ci::new(
                Some(Type::Function),
                Box::new(MathExpression::Mi(Mi("S".to_string()))),
            ),
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
        (
            Derivative::new(1, 1),
            Ci::new(
                Some(Type::Function),
                Box::new(MathExpression::Mi(Mi("S".to_string()))),
            ),
        ),
    );
}

#[test]
fn test_first_order_ode() {
    // ASKEM Hackathon 2, scenario 1, equation 1.
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>-</mo>
        <mi>β</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mfrac><mrow><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mi>N</mi></mfrac>
    </math>
    ";

    let FirstOrderODE { lhs_var, rhs } = input.parse::<FirstOrderODE>().unwrap();

    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(/ (* (* (- β) I) S) N)");

    // ASKEM Hackathon 2, scenario 1, equation 1, but with Newtonian derivative notation.
    let input = "
    <math>
        <mover><mi>S</mi><mo>˙</mo></mover><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>=</mo>
        <mo>-</mo>
        <mi>β</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mfrac><mrow><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mi>N</mi></mfrac>
    </math>
    ";

    let FirstOrderODE { lhs_var, rhs } = input.parse::<FirstOrderODE>().unwrap();

    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(/ (* (* (- β) I) S) N)");
}
