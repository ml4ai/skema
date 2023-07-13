//! Pratt parsing module to construct S-expressions from presentation MathML.
//! This is based on the nice tutorial at https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, Cn, Math, MathExpression, Mi, Mrow,
    },
    parsers::interpreted_mathml::math_expression_tree_document,
};

use derive_new::new;
use nom::error::Error;

use std::{fmt, str::FromStr};

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub enum Atom {
    Ci(Ci),
    Cn(Cn),
}

impl From<MathExpression> for Atom {
    fn from(expression: MathExpression) -> Self {
        match expression {
            MathExpression::Ci(x) => Atom::Ci(x),
            MathExpression::Mn(x) => Atom::Cn(Cn::new(None, x)),
            MathExpression::Mi(x) => Atom::Ci(Ci::new(None, Box::new(MathExpression::Mi(x)))),
            t => todo!("{t:?}"),
        }
    }
}

/// An S-expression like structure to represent mathematical expressions.
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub enum MathExpressionTree {
    Atom(Atom),
    Cons(Operator, Vec<MathExpressionTree>),
}

/// Implementation of Display for MathExpressionTree, in order to have a compact string
/// representation to work with --- this is useful both for human inspection and writing unit
/// tests.
impl fmt::Display for MathExpressionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpressionTree::Atom(x) => match x {
                Atom::Ci(Ci { content, .. }) => {
                    write!(f, "{}", content)
                }
                Atom::Cn(Cn { content, .. }) => {
                    write!(f, "{}", content)
                }
            },
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
    /// Translates a MathExpressionTree struct to a content MathML string.
    pub fn to_cmml(&self) -> String {
        let mut content_mathml = String::new();
        match self {
            MathExpressionTree::Atom(atom) => match atom {
                Atom::Ci(Ci { r#type, content }) => {
                    content_mathml.push_str(&format!("<ci>{}</ci>", content));
                }
                Atom::Cn(Cn { r#type, content }) => {
                    content_mathml.push_str(&format!("<ci>{}</ci>", content));
                }
            },
            MathExpressionTree::Cons(head, rest) => {
                content_mathml.push_str("<apply>");
                match head {
                    Operator::Add => content_mathml.push_str("<plus/>"),
                    Operator::Subtract => content_mathml.push_str("<minus/>"),
                    Operator::Multiply => content_mathml.push_str("<times/>"),
                    Operator::Equals => content_mathml.push_str("<eq/>"),
                    Operator::Divide => content_mathml.push_str("<divide/>"),
                    Operator::Derivative(Derivative { order, var_index })
                        if (order, var_index) == (&1_u8, &1_u8) =>
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

impl MathExpression {
    /// Flatten Mfrac and Mrow elements.
    fn flatten(&self, tokens: &mut Vec<MathExpression>) {
        match self {
            // Flatten/unwrap Mrows
            MathExpression::Mrow(Mrow(elements)) => {
                tokens.push(MathExpression::Mo(Operator::Lparen));
                for element in elements {
                    element.flatten(tokens);
                }
                tokens.push(MathExpression::Mo(Operator::Rparen));
            }
            // Insert implicit division operators, and wrap numerators and denominators in
            // parentheses for the Pratt parsing algorithm.
            MathExpression::Mfrac(numerator, denominator) => {
                tokens.push(MathExpression::Mo(Operator::Lparen));
                numerator.flatten(tokens);
                tokens.push(MathExpression::Mo(Operator::Rparen));
                tokens.push(MathExpression::Mo(Operator::Divide));
                tokens.push(MathExpression::Mo(Operator::Lparen));
                denominator.flatten(tokens);
                tokens.push(MathExpression::Mo(Operator::Rparen));
            }
            t => tokens.push(t.clone()),
        }
    }
}

impl Lexer {
    fn new(input: Vec<MathExpression>) -> Lexer {
        // Flatten mrows and mfracs
        let tokens = input.iter().fold(vec![], |mut acc, x| {
            x.flatten(&mut acc);
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
                    t => {
                        let last_token = acc.last().unwrap();
                        match last_token {
                            MathExpression::Mo(Operator::Rparen) => {
                                // If the last item in the accumulator is a right parenthesis ')',
                                // insert a multiplication operator
                                acc.push(&MathExpression::Mo(Operator::Multiply));
                            }
                            MathExpression::Mo(_) => {
                                // If the last item in the accumulator is an Mo (but not a right
                                // parenthesis), noop
                            }
                            _ => {
                                // Otherwise, insert a multiplication operator
                                acc.push(&MathExpression::Mo(Operator::Multiply));
                            }
                        }
                        // Insert the token
                        acc.push(t);
                    }
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

impl From<Ci> for MathExpressionTree {
    fn from(input: Ci) -> Self {
        MathExpressionTree::Atom(Atom::Ci(input))
    }
}

impl From<Math> for MathExpressionTree {
    fn from(input: Math) -> Self {
        expr(input.content)
    }
}

impl FromStr for MathExpressionTree {
    type Err = Error<String>;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (_, math) = math_expression_tree_document(input.into()).unwrap();
        Ok(math)
    }
}

/// The Pratt parsing algorithm for constructing an S-expression representing an equation.
fn expr_bp(lexer: &mut Lexer, min_bp: u8) -> MathExpressionTree {
    let mut lhs = match lexer.next() {
        Token::Atom(atom) => MathExpressionTree::Atom(atom.into()),
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
    println!("Output: {s}\n");
    assert_eq!(s.to_string(), "(+ x y)");

    let input = "<math><mi>a</mi><mo>=</mo><mi>x</mi><mo>+</mo><mi>y</mi><mi>z</mi></math>";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(= a (+ x (* y z)))");
    println!("Output: {s}\n");

    let input = "<math><mi>a</mi><mo>+</mo><mo>(</mo><mo>-</mo><mi>b</mi><mo>)</mo></math>";
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(+ a (- b))");

    let input =
        "<math><mn>2</mn><mi>a</mi><mo>(</mo><mi>c</mi><mo>+</mo><mi>d</mi><mo>)</mo></math>";
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(* (* 2 a) (+ c d))");

    let input = "
    <math>
        <mfrac><mrow><mi>d</mi><mi>S</mi></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>
        ";
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(= (D(1, 1) S) (* (* (- β) S) I))");

    let input = "
    <math>
        <mfrac><mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>
        ";
    println!("Input: {input}");
    let s = input.parse::<MathExpressionTree>().unwrap();
    assert_eq!(s.to_string(), "(= (D(1, 1) S) (* (* (- β) S) I))");
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
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
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
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>S</ci></apply><apply><divide/><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq2() {
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>E</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mi>β</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mfrac><mrow><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mi>N</mi></mfrac>
        <mo>−</mo>
        <mi>δ</mi><mi>E</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml,"<apply><eq/><apply><diff/><ci>E</ci></apply><apply><minus/><apply><divide/><apply><times/><apply><times/><ci>β</ci><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply><apply><times/><ci>δ</ci><ci>E</ci></apply></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq3() {
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mi>δ</mi><mi>E</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>−</mo>
        <mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><mi>γ</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>−</mo>
        <mi>α</mi><mi>ρ</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>I</ci></apply><apply><minus/><apply><minus/><apply><times/><ci>δ</ci><ci>E</ci></apply><apply><times/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci></apply><ci>I</ci></apply></apply><apply><times/><apply><times/><ci>α</ci><ci>ρ</ci></apply><ci>I</ci></apply></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq4() {
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><mi>γ</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>R</ci></apply><apply><times/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci></apply><ci>I</ci></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq5() {
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>D</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mi>α</mi>
        <mi>ρ</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>D</ci></apply><apply><times/><apply><times/><ci>α</ci><ci>ρ</ci></apply><ci>I</ci></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq6() {
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
        <mo>+</mo>
        <mi>ϵ</mi>
        <mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>S</ci></apply><apply><plus/><apply><divide/><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply><apply><times/><ci>ϵ</ci><ci>R</ci></apply></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq7() {
    let input = "
    <math>
        <mfrac>
        <mrow><mi>d</mi><mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><mi>γ</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>-</mo>
        <mi>ϵ</mi>
        <mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let ode = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", ode.to_string());
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><ci>R</ci></apply><apply><minus/><apply><times/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci></apply><ci>I</ci></apply><apply><times/><ci>ϵ</ci><ci>R</ci></apply></apply></apply>");
}

#[test]
fn test_content_hackathon2_scenario1_eq8() {
    let input = "
    <math>
        <mi>β</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>=</mo>
        <mi>κ</mi>
        <mi>m</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    println!("{}", exp.to_string());
    let cmml = exp.to_cmml();
    assert_eq!(
        cmml,
        "<apply><eq/><ci>β</ci><apply><times/><ci>κ</ci><ci>m</ci></apply></apply>"
    );
}
