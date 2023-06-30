//! Pratt parsing module to construct S-expressions from presentation MathML.
//! This is based on the nice tutorial at https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

use crate::{
    ast::{Math, MathExpression, Operator},
    parsing::math_expression,
};
use nom::multi::many0;
use std::fmt;

/// An S-expression like structure.
enum MathExpressionTree {
    Atom(MathExpression),
    Cons(Operator, Vec<MathExpressionTree>),
}

impl fmt::Display for MathExpressionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Atom(MathExpression),
    Op(Operator),
    Eof,
}

struct Lexer {
    tokens: Vec<Token>,
}

/// Check if fraction is a derivative of a single-variable function expressed in Leibniz notation,
/// and if so, return a derivative operator and the identifier of the function.
pub fn recognize_leibniz_diff_op<'a>(
    numerator: &Box<MathExpression>,
    denominator: &Box<MathExpression>,
) -> Result<(Operator, MathExpression), &'a str> {
    let mut numerator_contains_d = false;
    let mut denominator_contains_d = false;

    let mut numerator_contains_partial = false;
    let mut denominator_contains_partial = false;
    let mut function_candidate: Option<MathExpression> = None;

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

            // Gather the second identifier as a potential candidate function.
            function_candidate = Some(num_expressions[1].clone());
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

    if (numerator_contains_d && denominator_contains_d)
        || (numerator_contains_partial && denominator_contains_partial)
    {
        Ok((Operator::new_derivative(1, 1), function_candidate.unwrap()))
    } else {
        Err("This Mfrac does not correspond to a derivative in Leibniz notation")
    }
}

impl Lexer {
    fn new(input: Vec<MathExpression>) -> Lexer {
        // Recognize derivatives whenever possible.
        let tokens = input.clone().iter().fold(vec![], |mut acc, x| match x {
            MathExpression::Mover(base, overscript) => match **overscript {
                MathExpression::Mo(Operator::Other(ref os)) => {
                    if os.chars().all(|c| c == '˙') {
                        acc.push(MathExpression::Mo(Operator::new_derivative(
                            os.chars().count() as u8,
                            1,
                        )));
                        acc.push(*base.clone());
                        acc
                    } else {
                        acc.push(x.clone());
                        acc
                    }
                }
                _ => todo!(),
            },
            // TODO Implement detecting derivatives in Leibniz notation.
            MathExpression::Mfrac(numerator, denominator) => {
                if let Ok((derivative, function)) =
                    recognize_leibniz_diff_op(numerator, denominator)
                {
                    acc.push(MathExpression::Mo(derivative));
                    acc.push(function);
                } else {
                    acc.push(*numerator.clone());
                    acc.push(MathExpression::Mo(Operator::Divide));
                    acc.push(*denominator.clone());
                }
                acc
            }
            t => {
                acc.push(t.clone());
                acc
            }
        });
        //print!("tokens 1: [ ");
        //for token in &tokens {
        //print!("{token} ");
        //}
        //println!("]");

        // Insert implicit multiplication operators.
        let tokens = tokens.iter().fold(vec![], |mut acc, x| {
            if acc.len() == 0 {
                acc.push(x);
                acc
            } else {
                match x {
                    MathExpression::Mo(op) => {
                        if let Operator::Lparen = op {
                            // Check last element of acc
                            if let Some(MathExpression::Mo(_)) = acc.last() {
                            } else {
                                acc.push(&MathExpression::Mo(Operator::Multiply));
                            }
                        }
                        acc.push(x);
                        acc
                    }
                    t => match acc.last().unwrap() {
                        MathExpression::Mo(_) => {
                            acc.push(t);
                            acc
                        }
                        _ => {
                            acc.push(&MathExpression::Mo(Operator::Multiply));
                            acc.push(t);
                            acc
                        }
                    },
                }
            }
        });

        //print!("tokens 2: [ ");
        //for token in &tokens {
        //print!("{token} ");
        //}
        //println!("]");

        let mut tokens = tokens
            .into_iter()
            .map(|c| match c {
                MathExpression::Mo(op) => Token::Op(op.clone()),
                _ => Token::Atom(c.clone()),
            })
            .collect::<Vec<_>>();
        tokens.reverse();
        Lexer { tokens }
    }
    fn next(&mut self) -> Token {
        self.tokens.pop().unwrap_or(Token::Eof)
    }
    fn peek(&mut self) -> Token {
        self.tokens.last().cloned().unwrap_or(Token::Eof)
    }
}

fn expr(input: Vec<MathExpression>) -> MathExpressionTree {
    let mut lexer = Lexer::new(input);
    expr_bp(&mut lexer, 0)
}

impl From<Math> for MathExpressionTree {
    fn from(input: Math) -> Self {
        expr(input.content)
    }
}

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

fn prefix_binding_power(op: &Operator) -> ((), u8) {
    match op {
        Operator::Add | Operator::Subtract => ((), 9),
        Operator::Derivative { .. } => ((), 15),
        _ => panic!("Bad operator: {:?}", op),
    }
}

fn postfix_binding_power(op: &Operator) -> Option<(u8, ())> {
    let res = match op {
        Operator::Factorial => (11, ()),
        _ => return None,
    };
    Some(res)
}

fn infix_binding_power(op: &Operator) -> Option<(u8, u8)> {
    let res = match op {
        Operator::Equals => (2, 1),
        Operator::Add | Operator::Subtract => (5, 6),
        Operator::Multiply | Operator::Divide => (7, 8),
        Operator::Compose => (14, 13),
        Operator::Other(op) => panic!(format!("Unhandled operator: {op}!")),
        _ => return None,
    };
    Some(res)
}
#[test]
fn test_conversion() {
    let input = "<mi>x</mi><mo>+</mo><mi>y</mi>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(+ x y)");
    println!("Output: {s}\n");

    let input = "<mi>a</mi><mo>=</mo><mi>x</mi><mo>+</mo><mi>y</mi><mi>z</mi>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(= a (+ x (* y z)))");
    println!("Output: {s}\n");

    let input =
        "<mover><mi>S</mi><mo>˙</mo></mover><mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(= (D(1, 1) S) (* (* (- β) S) I))");
    println!("Output: {s}\n");

    let input =
        "<mover><mi>S</mi><mo>˙˙</mo></mover><mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(= (D(2, 1) S) (* (* (- β) S) I))");
    println!("Output: {s}\n");

    let input = "<mi>a</mi><mo>+</mo><mo>(</mo><mo>-</mo><mi>b</mi><mo>)</mo>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(+ a (- b))");
    println!("Output: {s}\n");

    let input = "<mn>2</mn><mi>a</mi><mo>(</mo><mi>c</mi><mo>+</mo><mi>d</mi><mo>)</mo>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(* (* 2 a) (+ c d))");
    println!("Output: {s}\n");

    let input = "<mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>";
    println!("Input: {input}");
    let (_, elements) = many0(math_expression)(input.into()).unwrap();
    let s = expr(elements);
    assert_eq!(s.to_string(), "(= (D(1, 1) S) (* (* (- β) S) I))");
    println!("Output: {s}\n");
}
