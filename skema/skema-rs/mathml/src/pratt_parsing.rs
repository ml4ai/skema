use crate::{
    ast::{MathExpression, Operator},
    parsing::math_expression,
};
use nom::multi::many0;
use std::{fmt, io::BufRead};
enum S {
    Atom(MathExpression),
    Cons(Operator, Vec<S>),
}
impl fmt::Display for S {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            S::Atom(i) => write!(f, "{}", i),
            S::Cons(head, rest) => {
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

impl Lexer {
    fn new(input: Vec<MathExpression>) -> Lexer {
        let mut tokens = input
            .iter()
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

fn expr(input: Vec<MathExpression>) -> S {
    let mut lexer = Lexer::new(input);
    expr_bp(&mut lexer, 0)
}

fn expr_bp(lexer: &mut Lexer, min_bp: u8) -> S {
    let mut lhs = match lexer.next() {
        Token::Atom(it) => S::Atom(it),
        Token::Op(Operator::Lparen) => {
            let lhs = expr_bp(lexer, 0);
            assert_eq!(lexer.next(), Token::Op(Operator::Rparen));
            lhs
        }
        Token::Op(op) => {
            let ((), r_bp) = prefix_binding_power(&op);
            let rhs = expr_bp(lexer, r_bp);
            S::Cons(op, vec![rhs])
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
            lhs = S::Cons(op, vec![lhs]);
            continue;
        }
        if let Some((l_bp, r_bp)) = infix_binding_power(&op) {
            if l_bp < min_bp {
                break;
            }
            lexer.next();
            lhs = {
                let rhs = expr_bp(lexer, r_bp);
                S::Cons(op, vec![lhs, rhs])
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
        _ => panic!("bad op: {:?}", op),
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
        _ => return None,
    };
    Some(res)
}
#[test]
fn test_conversion() {
    let elements = many0(math_expression)("<mi>x</mi><mo>+</mo><mi>y</mi>".into())
        .unwrap()
        .1;
    let s = expr(elements);
    println!("{s}");
}
