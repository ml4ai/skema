//! Pratt parsing module to construct S-expressions from presentation MathML.
//! This is based on the nice tutorial at https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

use crate::ast::{
    operator::{Derivative, Operator},
    Ci, Math, MathExpression, Mi, Mrow,
};
use derive_new::new;
use nom::error::Error;
use std::{fmt, str::FromStr};

#[cfg(test)]
use crate::parsers::first_order_ode::{first_order_ode, FirstOrderODE};

/// An S-expression like structure to represent mathematical expressions.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub enum MathExpressionTree {
    Atom(MathExpression),
    Cons(Operator, Vec<MathExpressionTree>),
}

/// Implementation of Display for MathExpressionTree, in order to have a compact string
/// representation to work with --- this is useful both for human inspection and writing unit
/// tests.
impl fmt::Display for MathExpressionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpressionTree::Atom(MathExpression::Ci(x)) => {
                write!(f, "{}", x.content)
            }
            MathExpressionTree::Atom(i) => write!(f, "{}", i),
            MathExpressionTree::Cons(head, rest) => {
                write!(f, "({}", head)?;
                for s in rest {
                    write!(f, " {}", s)?
                }
                write!(f, ")")
            }
        }
    }
}

impl MathExpressionTree {
    // Translates math expression tree to content mathml
    pub fn to_cmml(&self) -> String {
        let mut content_mathml = String::new();
        match self {
            MathExpressionTree::Atom(MathExpression::Ci(x)) => {
                content_mathml.push_str(&format!("<ci>{}</ci>", x.content.to_string()));
            }
            MathExpressionTree::Atom(i) => {
                match i {
                    MathExpression::Mi(Mi(id)) => {
                        content_mathml.push_str(&format!("<ci>{}</ci>", id.to_string()));
                    }
                    MathExpression::Mn(number) => {
                        content_mathml.push_str(&format!("<cn>{}</cn>", number.to_string()));
                    }
                    MathExpression::Mrow(Mrow(components)) => {
                        let mut before_parenthesis_starts: Vec<MathExpressionTree> = Vec::new();
                        let mut after_parenthesis_starts: Vec<MathExpressionTree> = Vec::new();
                        let mut parenthesis_exists = false;
                        //Handling parenthesis for case when it can be <times/> MathExpression or function of component
                        for items in components.iter() {
                            if let MathExpression::Mo(Lparen) = items {
                                parenthesis_exists = true;
                                continue;
                            }
                            if parenthesis_exists {
                                after_parenthesis_starts
                                    .push(MathExpressionTree::Atom(items.clone()));
                            } else {
                                before_parenthesis_starts
                                    .push(MathExpressionTree::Atom(items.clone()));
                            }
                        }
                        if !before_parenthesis_starts.is_empty()
                            && !after_parenthesis_starts.is_empty()
                        {
                            for b in before_parenthesis_starts {
                                content_mathml.push_str(&b.to_cmml());
                            }
                            //Ignores the function of component (e.g. (t) gets ignored)
                            if after_parenthesis_starts.len() == 1 {
                                println!("after_parenthesis_starts={:?}", after_parenthesis_starts);
                            } else {
                                for a in after_parenthesis_starts {
                                    content_mathml.push_str(&a.to_cmml());
                                }
                            }
                        } else if before_parenthesis_starts.is_empty()
                            && !after_parenthesis_starts.is_empty()
                        {
                            for a in after_parenthesis_starts {
                                content_mathml.push_str(&a.to_cmml());
                            }
                        }
                    }
                    _ => panic!("Unhandled MathExpressionTree in Atom matching"),
                }
            }
            MathExpressionTree::Cons(head, rest) => {
                content_mathml.push_str("<apply>");
                match head {
                    Operator::Add => content_mathml.push_str("<plus/>"),
                    Operator::Subtract => content_mathml.push_str("<minus/>"),
                    Operator::Multiply => content_mathml.push_str("<times/>"),
                    Operator::Equals => content_mathml.push_str("<eq/>"),
                    Operator::Divide => content_mathml.push_str("<divide/>"),
                    Operator::Derivative(Derivative { order, var_index })
                        if (*order, *var_index) == (1 as u8, 1 as u8) =>
                    {
                        content_mathml.push_str("<diff/>")
                    }
                    _ => {}
                }
                for s in rest {
                    content_mathml.push_str(&s.to_cmml());
                }

                content_mathml.push_str("</apply>");
            }
            _ => todo!(),
        }
        content_mathml
    }
}

/// Represents a token for the Pratt parsing algorithm.
#[derive(Debug, Clone, PartialEq, Eq, new)]
pub enum Token {
    Atom(MathExpression),
    Op(Operator),
    Eof,
}

/// Lexer for the Pratt parsing algorithm.
struct Lexer {
    /// Vector of input tokens.
    tokens: Vec<Token>,
}

impl Lexer {
    fn new(input: Vec<MathExpression>) -> Lexer {
        // Recognize derivatives in Newtonian notation.
        let tokens = input.iter().fold(vec![], |mut acc, x| {
            match x {
                MathExpression::Mover(base, overscript) => match **overscript {
                    MathExpression::Mo(Operator::Other(ref os)) => {
                        if os.chars().all(|c| c == '˙') {
                            acc.push(MathExpression::Mo(Operator::new_derivative(
                                Derivative::new(os.chars().count() as u8, 1),
                            )));
                            acc.push(*base.clone());
                        } else {
                            acc.push(x.clone());
                        }
                    }
                    _ => todo!(),
                },
                // Insert implicit division operators.
                MathExpression::Mfrac(numerator, denominator) => {
                    acc.push(*numerator.clone());
                    acc.push(MathExpression::Mo(Operator::Divide));
                    acc.push(*denominator.clone());
                }
                t => {
                    acc.push(t.clone());
                }
            }
            acc
        });

        // Insert implicit multiplication operators.
        let tokens = tokens.iter().fold(vec![], |mut acc, x| {
            if acc.is_empty() {
                acc.push(x);
            } else {
                match x {
                    // Handle left parenthesis operator '('
                    MathExpression::Mo(Operator::Lparen) => {
                        // Check last element of the accumulator.
                        if let Some(MathExpression::Mo(_)) = acc.last() {
                            // If the last element is an Mo, noop.
                        } else {
                            // Otherwise, insert a multiplication operator.
                            acc.push(&MathExpression::Mo(Operator::Multiply));
                        }
                        acc.push(x);
                    }
                    // Handle other types of operators.
                    MathExpression::Mo(_) => {
                        acc.push(x);
                    }
                    // Handle other types of MathExpression objects.
                    t => match acc.last().unwrap() {
                        MathExpression::Mo(_) => {
                            // If the last item in the accumulator is an Mo, add the current token.
                            acc.push(t);
                        }
                        _ => {
                            // Otherwise, insert a multiplication operator followed by the token.
                            acc.push(&MathExpression::Mo(Operator::Multiply));
                            acc.push(t);
                        }
                    },
                }
            }
            acc
        });

        // Convert MathExpression structs into Token structs.
        let mut tokens = tokens
            .into_iter()
            .map(|c| match c {
                MathExpression::Mo(op) => Token::Op(op.clone()),
                _ => Token::Atom(c.clone()),
            })
            .collect::<Vec<_>>();

        // Reverse the tokens for the Pratt parsing algorithm.
        tokens.reverse();
        Lexer { tokens }
    }

    /// Get the next Token and advance the iterator.
    fn next(&mut self) -> Token {
        self.tokens.pop().unwrap_or(Token::Eof)
    }

    /// Get the next token without advancing the iterator.
    fn peek(&self) -> Token {
        self.tokens.last().cloned().unwrap_or(Token::Eof)
    }
}

/// Construct a MathExpressionTree from a vector of MathExpression structs.
fn expr(input: Vec<MathExpression>) -> MathExpressionTree {
    let mut lexer = Lexer::new(input);
    expr_bp(&mut lexer, 0)
}

impl From<Vec<MathExpression>> for MathExpressionTree {
    fn from(input: Vec<MathExpression>) -> Self {
        expr(input)
    }
}

impl From<Math> for MathExpressionTree {
    fn from(input: Math) -> Self {
        expr(input.content)
    }
}

impl FromStr for MathExpressionTree {
    type Err = Error<String>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let math = s.parse::<Math>()?;
        Ok(MathExpressionTree::from(math))
    }
}

/// The Pratt parsing algorithm for constructing an S-expression representing an equation.
fn expr_bp(lexer: &mut Lexer, min_bp: u8) -> MathExpressionTree {
    let mut lhs = match lexer.next() {
        Token::Atom(it) => MathExpressionTree::Atom(it),
        Token::Op(Operator::Lparen) => {
            let lhs = expr_bp(lexer, 0);
            assert_eq!(lexer.next(), Token::Op(Operator::Rparen));
            lhs
        }
        Token::Op(op) => {
            let ((), r_bp) = prefix_binding_power(&op);
            let rhs = expr_bp(lexer, r_bp);
            MathExpressionTree::Cons(op, vec![rhs])
        }
        t => panic!("bad token: {:?}", t),
    };
    loop {
        let op = match lexer.peek() {
            Token::Eof => break,
            Token::Op(op) => op,
            t => panic!("bad token: {:?}", t),
        };
        if let Some((l_bp, ())) = postfix_binding_power(&op) {
            if l_bp < min_bp {
                break;
            }
            lexer.next();
            lhs = MathExpressionTree::Cons(op, vec![lhs]);
            continue;
        }
        if let Some((l_bp, r_bp)) = infix_binding_power(&op) {
            if l_bp < min_bp {
                break;
            }
            lexer.next();
            lhs = {
                let rhs = expr_bp(lexer, r_bp);
                MathExpressionTree::Cons(op, vec![lhs, rhs])
            };
            continue;
        }
        break;
    }
    lhs
}

/// Table of binding powers for prefix operators.
fn prefix_binding_power(op: &Operator) -> ((), u8) {
    match op {
        Operator::Add | Operator::Subtract => ((), 9),
        Operator::Derivative(Derivative { .. }) => ((), 15),
        _ => panic!("Bad operator: {:?}", op),
    }
}

/// Table of binding powers for postfix operators.
fn postfix_binding_power(op: &Operator) -> Option<(u8, ())> {
    let res = match op {
        Operator::Factorial => (11, ()),
        _ => return None,
    };
    Some(res)
}

/// Table of binding powers for infix operators.
fn infix_binding_power(op: &Operator) -> Option<(u8, u8)> {
    let res = match op {
        Operator::Equals => (2, 1),
        Operator::Add | Operator::Subtract => (5, 6),
        Operator::Multiply | Operator::Divide => (7, 8),
        Operator::Compose => (14, 13),
        Operator::Other(op) => panic!("Unhandled operator: {}!", op),
        _ => return None,
    };
    Some(res)
}

#[test]
fn test_conversion() {
    let input = "<math><mi>x</mi><mo>+</mo><mi>y</mi></math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(+ x y)");
    println!("Output: {s}\n");

    let input = "<math><mi>a</mi><mo>=</mo><mi>x</mi><mo>+</mo><mi>y</mi><mi>z</mi></math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(= a (+ x (* y z)))");
    println!("Output: {s}\n");

    let input = "<math>
        <mover><mi>S</mi><mo>˙</mo></mover><mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(= (D(1, 1) S) (* (* (- β) S) I))");
    println!("Output: {s}\n");

    let input = "<math>
        <mover><mi>S</mi><mo>˙˙</mo></mover><mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(= (D(2, 1) S) (* (* (- β) S) I))");
    println!("Output: {s}\n");

    let input = "<math><mi>a</mi><mo>+</mo><mo>(</mo><mo>-</mo><mi>b</mi><mo>)</mo></math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(+ a (- b))");
    println!("Output: {s}\n");

    let input =
        "<math><mn>2</mn><mi>a</mi><mo>(</mo><mi>c</mi><mo>+</mo><mi>d</mi><mo>)</mo></math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(* (* 2 a) (+ c d))");
    println!("Output: {s}\n");

    let input = "
    <math>
        <mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>
        ";
    println!("Input: {input}");
    let FirstOrderODE { lhs_var, rhs } = first_order_ode(input.into()).unwrap().1;
    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(* (* (- β) S) I)");
    println!("Output: {s}\n");

    let input = "
    <math>
        <mfrac><mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>
        ";
    println!("Input: {input}");
    let FirstOrderODE { lhs_var, rhs } = first_order_ode(input.into()).unwrap().1;
    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(rhs.to_string(), "(* (* (- β) S) I)");
    println!("Output: {s}\n");
}

#[test]
fn test_to_content_mathml_example1() {
    let input = "<math><mi>x</mi><mo>+</mo><mi>y</mi></math>";
    let s = input.parse::<MathExpressionTree>().unwrap();
    let content = s.to_cmml();
    assert_eq!(content, "<apply><plus/><ci>x</ci><ci>y</ci></apply>");
}
#[test]
fn test_to_content_mathml_example2() {
    let input = "<math>
        <mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>
        ";
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>S</ci></apply><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>S</ci></apply><ci>I</ci></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq1() {
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>S</ci></apply><apply><divide/><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply></apply>");
}
