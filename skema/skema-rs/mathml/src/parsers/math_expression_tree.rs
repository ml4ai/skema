//! Pratt parsing module to construct S-expressions from presentation MathML.
//! This is based on the nice tutorial at https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

use crate::{
    ast::{
        operator::{Derivative, Operator},
        Math, MathExpression, Mi, Mrow,
    },
    parsers::interpreted_mathml::interpreted_math,
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
                write!(f, "{}", x.content)?;
                for func in x.func_of.iter() {
                    write!(f, "(")?;
                    for (i, v) in func.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", v.content)?
                    }
                    write!(f, ")")?;
                }
                Ok(())
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
    /// Translates a MathExpressionTree struct to a content MathML string.
    pub fn to_cmml(&self) -> String {
        let mut content_mathml = String::new();
        match self {
            MathExpressionTree::Atom(i) => match i {
                MathExpression::Ci(x) => {
                    if x.func_of == None {
                        content_mathml.push_str(&format!("<ci>{}</ci>", x.content));
                    } else {
                        for func in x.func_of.iter() {
                            for v in func {
                                if v.content == Box::new(MathExpression::Mi(Mi("".to_string()))) {
                                    content_mathml.push_str(&format!("<ci>{}</ci>", x.content));
                                } else {
                                    content_mathml.push_str(&format!(
                                        "<apply><ci>{}</ci></apply>",
                                        x.content
                                    ));
                                }
                            }
                        }
                    }
                    for func in x.func_of.iter() {
                        for v in func {
                            if v.content == Box::new(MathExpression::Mi(Mi("".to_string()))) {
                            } else {
                                content_mathml.push_str(&format!("<ci>{}</ci>", v.content));
                            }
                        }
                    }
                }
                MathExpression::Mi(Mi(id)) => {
                    content_mathml.push_str(&format!("<ci>{}</ci>", id));
                }
                MathExpression::Mn(number) => {
                    content_mathml.push_str(&format!("<cn>{}</cn>", number));
                }
                MathExpression::Mrow(_) => {
                    panic!("All Mrows should have been removed by now!");
                }
                t => panic!("Unhandled MathExpression: {:?}", t),
            },
            MathExpressionTree::Cons(head, rest) => {
                content_mathml.push_str("<apply>");
                match head {
                    Operator::Add => content_mathml.push_str("<plus/>"),
                    Operator::Subtract => content_mathml.push_str("<minus/>"),
                    Operator::Multiply => content_mathml.push_str("<times/>"),
                    Operator::Equals => content_mathml.push_str("<eq/>"),
                    Operator::Divide => content_mathml.push_str("<divide/>"),
                    Operator::Power => content_mathml.push_str("<power/>"),
                    Operator::Exp => content_mathml.push_str("<exp/>"),
                    Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }) if (*order, *var_index) == (1_u8, 1_u8) => {
                        content_mathml.push_str("<diff/>");
                        content_mathml.push_str(&format!("<bvar>{}</bar>", bound_var));
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

    /// Translates to infix math expression to provide "string expressions" (e.g. ((α*ρ)*I)  )
    /// TA-4 uses "string expressions" to display over the transitions in their visual front end.
    pub fn to_infix_expression(&self) -> String {
        let mut expression = String::new();
        match self {
            MathExpressionTree::Atom(i) => match i {
                MathExpression::Ci(x) => {
                    expression.push_str(&format!("{}", x.content));
                }
                MathExpression::Mi(Mi(id)) => {
                    expression.push_str(&id.to_string());
                }
                MathExpression::Mn(number) => {
                    expression.push_str(&number.to_string());
                }
                MathExpression::Mrow(_) => {
                    panic!("All Mrows should have been removed by now!");
                }
                t => panic!("Unhandled MathExpression: {:?}", t),
            },

            MathExpressionTree::Cons(head, rest) => {
                let mut operation = String::new();
                match head {
                    Operator::Add => operation.push('+'),
                    Operator::Subtract => operation.push('-'),
                    Operator::Multiply => operation.push('*'),
                    Operator::Equals => operation.push('='),
                    Operator::Divide => operation.push('/'),
                    _ => {}
                }
                let mut component = Vec::new();
                for s in rest {
                    component.push(s.to_infix_expression());
                }
                let math_exp = format!("({})", component.join(&operation.to_string()));
                expression.push_str(&math_exp);
            }
        }
        expression
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
                    if let MathExpression::Ci(x) = element {
                        // Handles cos and sin as operators
                        if x.content == Box::new(MathExpression::Mi(Mi("cos".to_string()))) {
                            tokens.push(MathExpression::Mo(Operator::Cos));
                            if let Some(vec) = x.func_of.clone() {
                                for v in vec {
                                    tokens.push(MathExpression::Ci(v));
                                }
                            }
                        } else if x.content == Box::new(MathExpression::Mi(Mi("sin".to_string()))) {
                            tokens.push(MathExpression::Mo(Operator::Sin));
                        } else {
                            element.flatten(tokens);
                        }
                    } else {
                        element.flatten(tokens);
                    }
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
            MathExpression::Msup(base, superscript) => {
                if let MathExpression::Mi(b) = &**base {
                    if b == &Mi("e".to_string()) {
                        tokens.push(MathExpression::Mo(Operator::Exp));
                        tokens.push(MathExpression::Mo(Operator::Lparen));
                        superscript.flatten(tokens);
                        tokens.push(MathExpression::Mo(Operator::Rparen));
                    } else {
                        tokens.push(MathExpression::Mo(Operator::Lparen));
                        base.flatten(tokens);
                        tokens.push(MathExpression::Mo(Operator::Rparen));
                        tokens.push(MathExpression::Mo(Operator::Power));
                        tokens.push(MathExpression::Mo(Operator::Lparen));
                        superscript.flatten(tokens);
                        tokens.push(MathExpression::Mo(Operator::Rparen));
                    }
                } else {
                    tokens.push(MathExpression::Mo(Operator::Lparen));
                    base.flatten(tokens);
                    tokens.push(MathExpression::Mo(Operator::Rparen));
                    tokens.push(MathExpression::Mo(Operator::Power));
                    tokens.push(MathExpression::Mo(Operator::Lparen));
                    superscript.flatten(tokens);
                    tokens.push(MathExpression::Mo(Operator::Rparen));
                }
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
        let x = expr(input);
        x
    }
}

impl From<Math> for MathExpressionTree {
    fn from(input: Math) -> Self {
        let x = expr(input.content);
        x
    }
}

impl FromStr for MathExpressionTree {
    type Err = Error<String>;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (_, math) = interpreted_math(input.into()).unwrap();
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
        Operator::Exp => ((), 17),
        Operator::Cos => ((), 18),
        Operator::Sin => ((), 19),
        Operator::Tan => ((), 20),
        Operator::Derivative(Derivative { .. }) => ((), 21),
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
        Operator::Power => (16, 15),
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
    let FirstOrderODE {
        lhs_var,
        func_of,
        with_respect_to,
        rhs,
    } = first_order_ode(input.into()).unwrap().1;
    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(func_of[0].to_string(), "");
    assert_eq!(with_respect_to.to_string(), "t");
    assert_eq!(rhs.to_string(), "(* (* (- β) S) I)");
    println!("Output: {s}\n");

    let input = "
    <math>
        <mfrac><mrow><mi>d</mi><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mrow><mi>d</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo><mo>−</mo><mi>β</mi><mi>S</mi><mi>I</mi>
        </math>
        ";
    println!("Input: {input}");
    let FirstOrderODE {
        lhs_var,
        func_of,
        with_respect_to,
        rhs,
    } = first_order_ode(input.into()).unwrap().1;
    assert_eq!(lhs_var.to_string(), "S");
    assert_eq!(func_of[0].to_string(), "t");
    assert_eq!(with_respect_to.to_string(), "t");
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
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>S</ci></apply><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>S</ci></apply><ci>I</ci></apply></apply>");
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
    println!("ode={:?}", ode);
    let cmml = ode.to_cmml();
    println!("cmml={:?}", cmml);
    let FirstOrderODE {
        lhs_var,
        func_of,
        with_respect_to,
        rhs,
    } = first_order_ode(input.into()).unwrap().1;
    println!("rhs = {:?}", rhs.to_string());
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>S</ci></apply><apply><divide/><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply></apply>");
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml,"<apply><eq/><apply><diff/><bvar>t</bar><ci>E</ci></apply><apply><minus/><apply><divide/><apply><times/><apply><times/><ci>β</ci><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply><apply><times/><ci>δ</ci><ci>E</ci></apply></apply></apply>");
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>I</ci></apply><apply><minus/><apply><minus/><apply><times/><ci>δ</ci><ci>E</ci></apply><apply><times/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci></apply><ci>I</ci></apply></apply><apply><times/><apply><times/><ci>α</ci><ci>ρ</ci></apply><ci>I</ci></apply></apply></apply>");
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>R</ci></apply><apply><times/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci></apply><ci>I</ci></apply></apply>");
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>D</ci></apply><apply><times/><apply><times/><ci>α</ci><ci>ρ</ci></apply><ci>I</ci></apply></apply>");
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>S</ci></apply><apply><plus/><apply><divide/><apply><times/><apply><times/><apply><minus/><ci>β</ci></apply><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply><apply><times/><ci>ϵ</ci><ci>R</ci></apply></apply></apply>");
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
    let ode = input.parse::<FirstOrderODE>().unwrap();
    let cmml = ode.to_cmml();
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>R</ci></apply><apply><minus/><apply><times/><apply><times/><apply><minus/><cn>1</cn><ci>α</ci></apply><ci>γ</ci></apply><ci>I</ci></apply><apply><times/><ci>ϵ</ci><ci>R</ci></apply></apply></apply>");
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
    let cmml = exp.to_cmml();
    assert_eq!(
        cmml,
        "<apply><eq/><apply><ci>β</ci></apply><ci>t</ci><apply><times/><ci>κ</ci><apply><ci>m</ci></apply><ci>t</ci></apply></apply>"
    );
}

#[test]
fn test_expression1() {
    let input = "<math><mi>γ</mi><mi>I</mi></math>";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let math = exp.to_infix_expression();
    assert_eq!(math, "(γ*I)");
}

#[test]
fn test_expression2() {
    let input = "
    <math>
        <mi>α</mi>
        <mi>ρ</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let math = exp.to_infix_expression();
    println!("math = {:?}", math);
    assert_eq!(math, "((α*ρ)*I)");
}

#[test]
fn test_expression3() {
    let input = "
    <math>
        <mi>β</mi>
        <mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mfrac><mrow><mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo></mrow><mi>N</mi></mfrac>
        <mo>−</mo>
        <mi>δ</mi><mi>E</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let math = exp.to_infix_expression();
    assert_eq!(math, "((((β*I)*S)/N)-(δ*E))")
}

#[test]
fn test_expression4() {
    let input = "
    <math>
        <mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><mi>γ</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>-</mo>
        <mi>ϵ</mi>
        <mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let math = exp.to_infix_expression();
    assert_eq!(math, "((((1-α)*γ)*I)-(ϵ*R))")
}

#[test]
fn test_mfrac() {
    let input = "
    <math>
        <mfrac><mi>S</mi><mi>N</mi></mfrac>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", exp);
}

#[test]
fn test_superscript() {
    let input = "
    <math>
        <msup>
        <mi>x</mi>
        <mn>3</mn>
        </msup>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", exp);
}

#[test]
fn test_msup_exp() {
    let input = "
    <math>
        <msup>
        <mi>e</mi>
        <mrow><mo>-</mo><mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><mi>γ</mi><mi>I</mi></mrow>
        </msup>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", exp);
}

#[test]
fn test_trig_cos() {
    let input = "
    <math>
        <mrow>
        <mi>cos</mi>
        <mi>x</mi>
        </mrow>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", exp);
}

#[test]
fn test_trig_another_cos() {
    let input = "
    <math>
        <mrow>
        <mi>cos</mi>
        <mo>(</mo>
        <mi>x</mi>
        <mo>)</mo>
        </mrow>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    println!("exp={:?}", exp);
}
