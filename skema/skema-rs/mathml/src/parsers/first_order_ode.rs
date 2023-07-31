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
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::map,
    error::Error,
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
    let (s, _) = stag!("math")(input)?;

    // Recognize LHS derivative
    let (s, (_, ci)) = alt((
        first_order_derivative_leibniz_notation,
        newtonian_derivative,
    ))(s)?;

    // Recognize equals sign
    let (s, _) = delimited(stag!("mo"), equals, etag!("mo"))(s)?;

    // Recognize other tokens
    let (s, remaining_tokens) = many1(alt((
        map(ci_univariate_func, MathExpression::Ci),
        map(ci_unknown, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(Type::Function),
                content,
            })
        }),
        map(operator, MathExpression::Mo),
        math_expression,
    )))(s)?;

    let (s, _) = etag!("math")(s)?;

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
            println!(
                "ode_line rhs string below: {:?}\n",
                ode.rhs.to_string().clone()
            );
            ode.rhs = flatten_mults(ode.rhs.clone());
            println!(
                "ode_line rhs string after: {:?}\n",
                ode.rhs.to_string().clone()
            );
            println!("ode_line rhs object: {:?}\n", ode.rhs.clone());
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

// this function takes in one ode equations and returns a vector of the terms in it
pub fn get_terms(sys_states: Vec<String>, ode: FirstOrderODE) -> Vec<PnTerm> {
    let mut terms = Vec::<PnTerm>::new();
    let _exp_states = Vec::<String>::new();
    let _parameters = Vec::<String>::new();

    let dyn_state = ode.lhs_var.to_string();

    match ode.rhs {
        Cons(ref x, ref y) => {
            match &x {
                Multiply => {
                    let mut temp_term = get_term_mult(sys_states, y.clone());
                    temp_term.dyn_state = dyn_state;
                    terms.push(temp_term.clone());
                    println!("mult temp_term: {:?}\n", temp_term)
                }
                Subtract => {
                    /* found multiple terms */
                    if y.len() == 1 {
                        // unary sub, so much be mult inside
                        match &y[0] {
                            Cons(_x1, ref y1) => {
                                let mut temp_term = get_term_mult(sys_states, y1.clone());
                                temp_term.dyn_state = dyn_state;
                                if temp_term.polarity {
                                    temp_term.polarity = false;
                                } else {
                                    temp_term.polarity = true;
                                }
                                terms.push(temp_term.clone());
                                println!("Unary Sub temp_term: {:?}\n", temp_term)
                            }
                            Atom(_x1) => {
                                println!("Not valid term in PN")
                            }
                        }
                    } else {
                        /* this is the same as Add, but with a polarity swap on the second term */
                        /* this actually need to support an entire binary sub inside it again :( */
                        match &y[0] {
                            Cons(x1, ref y1) => match &x1 {
                                Multiply => {
                                    let mut temp_term =
                                        get_term_mult(sys_states.clone(), y1.clone());
                                    temp_term.dyn_state = dyn_state.clone();
                                    terms.push(temp_term.clone());
                                    println!("binary sub1 mult temp_term: {:?}\n", temp_term)
                                }
                                Subtract => {
                                    match &y1[0] {
                                        Cons(x2, ref y2) => {
                                            if y2.len() == 1 {
                                                let mut temp_term =
                                                    get_term_mult(sys_states.clone(), y2.clone());
                                                temp_term.dyn_state = dyn_state.clone();
                                                if temp_term.polarity {
                                                    temp_term.polarity = false;
                                                } else {
                                                    temp_term.polarity = true;
                                                }
                                                terms.push(temp_term.clone());
                                                println!(
                                                    "binary sub1 unarysub temp_term: {:?}\n",
                                                    temp_term
                                                )
                                            } else {
                                                match &x2 {
                                                    Subtract => {
                                                        println!("to do")
                                                    }
                                                    Add => {
                                                        println!("to do")
                                                    }
                                                    Divide => {
                                                        println!("to do")
                                                    }
                                                    Multiply => {
                                                        let mut temp_term = get_term_mult(
                                                            sys_states.clone(),
                                                            y2.clone(),
                                                        );
                                                        temp_term.dyn_state = dyn_state.clone();
                                                        terms.push(temp_term.clone());
                                                        println!("binary sub1 binary sub1 mult temp_term: {:?}\n", temp_term)
                                                    }
                                                    _ => println!("Not supported equation type"),
                                                }
                                            }
                                        }
                                        _ => {
                                            println!("Not valid term for PN")
                                        }
                                    }
                                    match &y1[1] {
                                        Cons(x2, ref y2) => {
                                            if y2.len() == 1 {
                                                let mut temp_term =
                                                    get_term_mult(sys_states.clone(), y2.clone());
                                                temp_term.dyn_state = dyn_state.clone();
                                                terms.push(temp_term.clone());
                                                println!(
                                                    "binary sub1 unarysub temp_term: {:?}\n",
                                                    temp_term
                                                )
                                            } else {
                                                match &x2 {
                                                    Subtract => {
                                                        println!("to do")
                                                    }
                                                    Add => {
                                                        println!("to do")
                                                    }
                                                    Divide => {
                                                        println!("to do")
                                                    }
                                                    Multiply => {
                                                        let mut temp_term = get_term_mult(
                                                            sys_states.clone(),
                                                            y2.clone(),
                                                        );
                                                        temp_term.dyn_state = dyn_state.clone();
                                                        if temp_term.polarity {
                                                            temp_term.polarity = false;
                                                        } else {
                                                            temp_term.polarity = true;
                                                        }
                                                        terms.push(temp_term.clone());
                                                        println!("binary sub1 binary sub2 mult temp_term: {:?}\n", temp_term)
                                                    }
                                                    _ => println!("Not supported equation type"),
                                                }
                                            }
                                        }
                                        _ => {
                                            println!("Not valid term for PN")
                                        }
                                    }
                                }
                                // new edge case to handle
                                Divide => match &y1[0] {
                                    Cons(_x2, ref y2) => {
                                        let mut temp_term =
                                            get_term_mult(sys_states.clone(), y2.clone());
                                        temp_term.dyn_state = dyn_state.clone();
                                        temp_term.parameters.push(y1[1].to_string());
                                        temp_term.expression =
                                            MathExpressionTree::Cons(Divide, y1.clone()).to_cmml();
                                        terms.push(temp_term.clone());
                                        println!(
                                            "binary sub1 div temp_term: {:?}\n",
                                            temp_term.clone()
                                        )
                                    }
                                    _ => {
                                        println!("dont support this")
                                    }
                                },
                                _ => {
                                    println!("Error unsupported operation")
                                }
                            },
                            Atom(_x1) => {
                                println!("Not valid term for PN")
                            }
                        }
                        match &y[1] {
                            Cons(x1, ref y1) => match x1 {
                                Multiply => {
                                    let mut temp_term = get_term_mult(sys_states, y1.clone());
                                    temp_term.dyn_state = dyn_state;
                                    if temp_term.polarity {
                                        temp_term.polarity = false;
                                    } else {
                                        temp_term.polarity = true;
                                    }
                                    terms.push(temp_term.clone());
                                    println!("binary sub2 mult temp_term: {:?}\n", temp_term)
                                }
                                Subtract => match &y1[0] {
                                    Cons(_x2, ref y2) => {
                                        let mut temp_term = get_term_mult(sys_states, y2.clone());
                                        temp_term.dyn_state = dyn_state;
                                        terms.push(temp_term.clone());
                                        println!(
                                            "binary sub2 unarysub temp_term: {:?}\n",
                                            temp_term
                                        )
                                    }
                                    _ => {
                                        println!("Not valid term for PN")
                                    }
                                },
                                _ => {
                                    println!("Error unsupported operation")
                                }
                            },
                            Atom(_x1) => {
                                println!("Not valid term for PN")
                            }
                        }
                    }
                } // unary or binary
                Add => {
                    /* found multiple terms */
                    match &y[0] {
                        Cons(x1, ref y1) => match x1 {
                            Multiply => {
                                let mut temp_term = get_term_mult(sys_states.clone(), y1.clone());
                                temp_term.dyn_state = dyn_state.clone();
                                terms.push(temp_term);
                            }
                            Subtract => match &y1[0] {
                                Cons(_x2, ref y2) => {
                                    let mut temp_term =
                                        get_term_mult(sys_states.clone(), y2.clone());
                                    temp_term.dyn_state = dyn_state.clone();
                                    if temp_term.polarity {
                                        temp_term.polarity = false;
                                    } else {
                                        temp_term.polarity = true;
                                    }
                                    terms.push(temp_term);
                                }
                                _ => {
                                    println!("Not valid term for PN")
                                }
                            },
                            _ => {
                                println!("Error unsupported operation")
                            }
                        },
                        Atom(_x1) => {
                            println!("Not valid term for PN")
                        }
                    }
                    match &y[1] {
                        Cons(x1, ref y1) => match x1 {
                            Multiply => {
                                let mut temp_term = get_term_mult(sys_states, y1.clone());
                                temp_term.dyn_state = dyn_state;
                                terms.push(temp_term);
                            }
                            Subtract => match &y1[0] {
                                Cons(_x2, ref y2) => {
                                    let mut temp_term = get_term_mult(sys_states, y2.clone());
                                    temp_term.dyn_state = dyn_state;
                                    if temp_term.polarity {
                                        temp_term.polarity = false;
                                    } else {
                                        temp_term.polarity = true;
                                    }
                                    terms.push(temp_term);
                                }
                                _ => {
                                    println!("Not valid term for PN")
                                }
                            },
                            _ => {
                                println!("Error unsupported operation")
                            }
                        },
                        Atom(_x1) => {
                            println!("Not valid term for PN")
                        }
                    }
                }
                Divide => match &y[0] {
                    Cons(x1, ref y1) => match &x1 {
                        Multiply => {
                            let mut temp_term = get_term_mult(sys_states, y1.clone());
                            temp_term.dyn_state = dyn_state;
                            temp_term.parameters.push(y[1].to_string());
                            temp_term.expression =
                                MathExpressionTree::Cons(Divide, y.clone()).to_cmml();
                            terms.push(temp_term.clone())
                        }
                        Subtract => {
                            /* now to support unary subtract as numerator y[0] */
                            match &y1[0] {
                                Cons(_x2, ref y2) => {
                                    /* unary to mult */
                                    /* This has to be a unary sub into a mult (unless really stupid equation) */
                                    let mut temp_term = get_term_mult(sys_states, y2.clone());
                                    // swap polarity of temp term
                                    if temp_term.polarity {
                                        temp_term.polarity = false;
                                    } else {
                                        temp_term.polarity = true;
                                    }
                                    temp_term.parameters.push(y[1].to_string());
                                    temp_term.expression =
                                        MathExpressionTree::Cons(Divide, y.clone()).to_cmml();
                                    terms.push(temp_term.clone())
                                }
                                Atom(_x2) => {
                                    /* This is only the case of a parameter being 1/param and the top being a
                                    negative of the state variable, since no mult */
                                    let temp_term = PnTerm {
                                        dyn_state,
                                        exp_states: [y1[0].to_string()].to_vec(),
                                        polarity: false,
                                        expression: ode.rhs.clone().to_cmml(),
                                        parameters: [y[1].to_string()].to_vec(),
                                    };
                                    terms.push(temp_term)
                                }
                            }
                        }
                        _ => {
                            println!("Error expected only Multiply or Unary subtract for numerator")
                        }
                    },
                    Atom(_x1) => {
                        println!("Error, expected numerator terms to be a Multiplication")
                    }
                }, // divide seem to be in front of mult always, will make that assumption.
                _ => {
                    println!("Warning unsupported operator on expression")
                }
            }
        }
        Atom(_x) => {
            println!("Warning unexpected RHS structure")
        }
    }
    terms
}

// this takes in the arguments of a multiply term and returns the PnTerm for it
// we do expect at most only one unary subtraction
pub fn get_term_mult(sys_states: Vec<String>, eq: Vec<MathExpressionTree>) -> PnTerm {
    let _terms = Vec::<PnTerm>::new();
    let mut variables = Vec::<String>::new();
    let mut exp_states = Vec::<String>::new();
    let mut parameters = Vec::<String>::new();
    let mut polarity = true;

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
                        match &y[0] {
                            Cons(_x1, y1) => {
                                // assumption only unary sub is possible
                                polarity = false;
                                variables.push(y1[0].to_string())
                            }
                            Atom(_x) => variables.push(y[0].to_string()),
                        }
                        match &y[1] {
                            Cons(_x1, y1) => {
                                // assumption only unary sub is possible
                                polarity = false;
                                variables.push(y1[1].to_string())
                            }
                            Atom(_x) => variables.push(y[1].to_string()),
                        }
                    }
                    Add => {
                        if y.len() == 1 {
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

    let mut ind = Vec::<usize>::new();
    for (i, var) in variables.iter().enumerate() {
        for sys_var in sys_states.iter() {
            if var == sys_var {
                exp_states.push(var.clone());
                ind.push(i);
            }
        }
    }

    for i in ind.iter().rev() {
        variables.remove(*i);
    }

    for var in variables.iter() {
        parameters.push(var.clone());
    }

    PnTerm {
        dyn_state: "temp".to_string(),
        exp_states,
        polarity,
        expression: MathExpressionTree::Cons(Multiply, eq).to_cmml(),
        parameters,
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
