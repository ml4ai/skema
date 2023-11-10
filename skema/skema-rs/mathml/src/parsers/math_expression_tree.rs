//! Pratt parsing module to construct S-expressions from presentation MathML.
//! This is based on the nice tutorial at https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

//use crate::parsers::math_expression_tree::MathExpression::Differential;
use crate::{
    ast::{
        operator::{Derivative, Operator},
        Math, MathExpression, Mi, Mrow,
    },
    parsers::interpreted_mathml::interpreted_math,
};
use derive_new::new;
use nom::error::Error;
use regex::Regex;
use std::{fmt, str::FromStr};
// use crate::expression::Atom::Operator;

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
            MathExpressionTree::Atom(MathExpression::Mo(Operator::Derivative(x))) => {
                write!(f, "{:?}", x)
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

/// Converts Unicode, Greek letters, their symbols, and English representations of Greek letters in an input string to their respective LaTeX expressions.
fn unicode_to_latex(input: &str) -> String {
    // Step 1: Handle English representations of Greek letters
    let re_english_greek = Regex::new(r"Lambda|lambda|Alpha|alpha|Beta|beta|Gamma|gamma|Delta|delta|Epsilon|epsilon|Zeta|zeta|Eta|eta|Theta|theta|Iota|iota|Kappa|kappa|Lambda|lambda|Mu|mu|Nu|nu|Xi|xi|Omicron|omicron|Pi|pi|Rho|rho|Sigma|sigma|Tau|tau|Upsilon|upsilon|Phi|phi|Chi|chi|Psi|psi|Omega|omega").unwrap();
    let replaced_english_greek =
        re_english_greek.replace_all(input, |caps: &regex::Captures| match &caps[0] {
            "Lambda" => "\\Lambda".to_string(),
            "lambda" => "\\lambda".to_string(),
            "Alpha" => "\\Alpha".to_string(),
            "alpha" => "\\alpha".to_string(),
            "Beta" => "\\Beta".to_string(),
            "beta" => "\\beta".to_string(),
            "Gamma" => "\\Gamma".to_string(),
            "gamma" => "\\gamma".to_string(),
            "Delta" => "\\Delta".to_string(),
            "delta" => "\\delta".to_string(),
            "Epsilon" => "\\Epsilon".to_string(),
            "epsilon" => "\\epsilon".to_string(),
            "Zeta" => "\\Zeta".to_string(),
            "zeta" => "\\zeta".to_string(),
            "Eta" => "\\Eta".to_string(),
            "eta" => "\\eta".to_string(),
            "Theta" => "\\Theta".to_string(),
            "theta" => "\\theta".to_string(),
            "Iota" => "\\Iota".to_string(),
            "iota" => "\\iota".to_string(),
            "Kappa" => "\\Kappa".to_string(),
            "kappa" => "\\kappa".to_string(),
            "Lambda" => "\\Lambda".to_string(),
            "lambda" => "\\lambda".to_string(),
            "Mu" => "\\Mu".to_string(),
            "mu" => "\\mu".to_string(),
            "Nu" => "\\Nu".to_string(),
            "nu" => "\\nu".to_string(),
            "Xi" => "\\Xi".to_string(),
            "xi" => "\\xi".to_string(),
            "Omicron" => "\\Omicron".to_string(),
            "omicron" => "\\omicron".to_string(),
            "Pi" => "\\Pi".to_string(),
            "pi" => "\\pi".to_string(),
            "Rho" => "\\Rho".to_string(),
            "rho" => "\\rho".to_string(),
            "Sigma" => "\\Sigma".to_string(),
            "sigma" => "\\sigma".to_string(),
            "Tau" => "\\Tau".to_string(),
            "tau" => "\\tau".to_string(),
            "Upsilon" => "\\Upsilon".to_string(),
            "upsilon" => "\\upsilon".to_string(),
            "Phi" => "\\Phi".to_string(),
            "phi" => "\\phi".to_string(),
            "Chi" => "\\Chi".to_string(),
            "chi" => "\\chi".to_string(),
            "Psi" => "\\Psi".to_string(),
            "psi" => "\\psi".to_string(),
            "Omega" => "\\Omega".to_string(),
            "omega" => "\\omega".to_string(),
            _ => caps[0].to_string(),
        });

    // Step 2: Handle Greek letters represented in Unicode
    let re_unicode = Regex::new(r"&#x([0-9A-Fa-f]+);").unwrap();
    let replaced_unicode =
        re_unicode.replace_all(&replaced_english_greek, |caps: &regex::Captures| {
            let unicode = u32::from_str_radix(&caps[1], 16).unwrap();
            match unicode {
                0x0391 => "\\Alpha".to_string(),
                0x03B1 => "\\alpha".to_string(),
                0x0392 => "\\Beta".to_string(),
                0x03B2 => "\\beta".to_string(),
                0x0393 => "\\Gamma".to_string(),
                0x03B3 => "\\gamma".to_string(),
                0x0394 => "\\Delta".to_string(),
                0x03B4 => "\\delta".to_string(),
                0x0395 => "\\Epsilon".to_string(),
                0x03B5 => "\\epsilon".to_string(),
                0x0396 => "\\Zeta".to_string(),
                0x03B6 => "\\zeta".to_string(),
                0x0397 => "\\Eta".to_string(),
                0x03B7 => "\\eta".to_string(),
                0x0398 => "\\Theta".to_string(),
                0x03B8 => "\\theta".to_string(),
                0x0399 => "\\Iota".to_string(),
                0x03B9 => "\\iota".to_string(),
                0x039A => "\\Kappa".to_string(),
                0x03BA => "\\kappa".to_string(),
                0x039B => "\\Lambda".to_string(),
                0x03BB => "\\lambda".to_string(),
                0x039C => "\\Mu".to_string(),
                0x03BC => "\\mu".to_string(),
                0x039D => "\\Nu".to_string(),
                0x03BD => "\\nu".to_string(),
                0x039E => "\\Xi".to_string(),
                0x03BE => "\\xi".to_string(),
                0x039F => "\\Omicron".to_string(),
                0x03BF => "\\omicron".to_string(),
                0x03A0 => "\\Pi".to_string(),
                0x03C0 => "\\pi".to_string(),
                0x03A1 => "\\Rho".to_string(),
                0x03C1 => "\\rho".to_string(),
                0x03A3 => "\\Sigma".to_string(),
                0x03C3 => "\\sigma".to_string(),
                0x03A4 => "\\Tau".to_string(),
                0x03C4 => "\\tau".to_string(),
                0x03A5 => "\\Upsilon".to_string(),
                0x03C5 => "\\upsilon".to_string(),
                0x03A6 => "\\Phi".to_string(),
                0x03C6 => "\\phi".to_string(),
                0x03A7 => "\\Chi".to_string(),
                0x03C7 => "\\chi".to_string(),
                0x03A8 => "\\Psi".to_string(),
                0x03C8 => "\\psi".to_string(),
                0x03A9 => "\\Omega".to_string(),
                0x03C9 => "\\omega".to_string(),
                _ => caps[0].to_string(),
            }
        });

    // Step 3: Handle other Unicode representations
    let re_other_unicode = Regex::new(r"&#x([0-9A-Fa-f]+);").unwrap();
    let replaced_other_unicode =
        re_other_unicode.replace_all(&replaced_unicode, |caps: &regex::Captures| {
            format!(
                "\\unicode{{U+{:X}}}",
                u32::from_str_radix(&caps[1], 16).unwrap()
            )
        });

    // Step 4: Handle Greek letter symbols
    let re_greek_symbols = Regex::new(r"Λ|λ|Α|α|Β|β|Γ|γ|Δ|δ|Ε|ε|Ζ|ζ|Η|η|Θ|θ|Ι|ι|Κ|κ|Λ|λ|Μ|μ|Ν|ν|Ξ|ξ|Ο|ο|Π|π|Ρ|ρ|Σ|σ|ς|Τ|τ|Υ|υ|Φ|φ|Χ|χ|Ψ|ψ|Ω|ω").unwrap();
    let replaced_greek_symbols =
        re_greek_symbols.replace_all(&replaced_other_unicode, |caps: &regex::Captures| {
            match &caps[0] {
                "Λ" => "\\Lambda".to_string(),
                "λ" => "\\lambda".to_string(),
                "Α" => "\\Alpha".to_string(),
                "α" => "\\alpha".to_string(),
                "Β" => "\\Beta".to_string(),
                "β" => "\\beta".to_string(),
                "Γ" => "\\Gamma".to_string(),
                "γ" => "\\gamma".to_string(),
                "Δ" => "\\Delta".to_string(),
                "δ" => "\\delta".to_string(),
                "Ε" => "\\Epsilon".to_string(),
                "ε" => "\\epsilon".to_string(),
                "Ζ" => "\\Zeta".to_string(),
                "ζ" => "\\zeta".to_string(),
                "Η" => "\\Eta".to_string(),
                "η" => "\\eta".to_string(),
                "Θ" => "\\Theta".to_string(),
                "θ" => "\\theta".to_string(),
                "Ι" => "\\Iota".to_string(),
                "ι" => "\\iota".to_string(),
                "Κ" => "\\Kappa".to_string(),
                "κ" => "\\kappa".to_string(),
                "Λ" => "\\Lambda".to_string(),
                "λ" => "\\lambda".to_string(),
                "Μ" => "\\Mu".to_string(),
                "μ" => "\\mu".to_string(),
                "Ν" => "\\Nu".to_string(),
                "ν" => "\\nu".to_string(),
                "Ξ" => "\\Xi".to_string(),
                "ξ" => "\\xi".to_string(),
                "Ο" => "\\Omicron".to_string(),
                "ο" => "\\omicron".to_string(),
                "Π" => "\\Pi".to_string(),
                "π" => "\\pi".to_string(),
                "Ρ" => "\\Rho".to_string(),
                "ρ" => "\\rho".to_string(),
                "Σ" => "\\Sigma".to_string(),
                "σ" => "\\sigma".to_string(),
                "ς" => "\\varsigma".to_string(), // 处理 sigma 的 final form
                "Τ" => "\\Tau".to_string(),
                "τ" => "\\tau".to_string(),
                "Υ" => "\\Upsilon".to_string(),
                "υ" => "\\upsilon".to_string(),
                "Φ" => "\\Phi".to_string(),
                "φ" => "\\phi".to_string(),
                "Χ" => "\\Chi".to_string(),
                "χ" => "\\chi".to_string(),
                "Ψ" => "\\Psi".to_string(),
                "ψ" => "\\psi".to_string(),
                "Ω" => "\\Omega".to_string(),
                "ω" => "\\omega".to_string(),
                _ => caps[0].to_string(),
            }
        });

    replaced_greek_symbols.to_string()
}

fn is_unary_operator(op: &Operator) -> bool {
    match op {
        Operator::Sqrt
        | Operator::Factorial
        | Operator::Exp
        | Operator::Power
        | Operator::Grad
        | Operator::Div
        | Operator::Abs
        | Operator::Derivative(_)
        | Operator::Sin
        | Operator::Cos
        | Operator::Tan
        | Operator::Sec
        | Operator::Csc
        | Operator::Cot
        | Operator::Arcsin
        | Operator::Arccos
        | Operator::Arctan
        | Operator::Arcsec
        | Operator::Arccsc
        | Operator::Arccot
        | Operator::Mean => true,
        _ => false,
    }
}

// Process parentheses in an expression and update the LaTeX string.
// If the expression is a unary operator, it is added to the LaTeX string as is.
// If the expression is not a unary operator, it is wrapped in parentheses before being added to the LaTeX string.
// If the expression is an atom, it is added to the LaTeX string directly.
fn process_expression_parentheses(expression: &mut String, met: &MathExpressionTree) {
    // Check if the rest vector is not empty and contains a MathExpressionTree::Cons variant.
    if let MathExpressionTree::Cons(op, _) = met {
        // Check if the operator is a unary operator.
        if is_unary_operator(op) {
            // If it is a unary operator, add it to the LaTeX string as is.
            expression.push_str(&format!("{}", met.to_latex()));
        } else {
            // If it is not a unary operator, wrap it in parentheses before adding it to the LaTeX string.
            expression.push_str(&format!("({})", met.to_latex()));
        }
    } else {
        // If the expression is an atom, add it to the LaTeX string directly.
        expression.push_str(&format!("{}", met.to_latex()));
    }
}

/// Processes a MathExpression under the type of MathExpressionTree::Atom and appends
/// the corresponding LaTeX representation to the provided String.
fn process_math_expression(expr: &MathExpression, expression: &mut String) {
    match expr {
        // If it's a Ci variant, recursively process its content
        MathExpression::Ci(x) => {
            process_math_expression(&*x.content, expression);
        }
        MathExpression::Mi(Mi(id)) => {
            expression.push_str(unicode_to_latex(&id.to_string()).as_str());
        }
        MathExpression::Mn(number) => {
            expression.push_str(&number.to_string());
        }
        MathExpression::Mrow(_) => {
            panic!("All Mrows should have been removed by now!");
        }
        t => panic!("Unhandled MathExpression: {:?}", t),
    }
}

impl MathExpressionTree {
    /// Translates a MathExpressionTree struct to a content MathML string.
    pub fn to_cmml(&self) -> String {
        let mut content_mathml = String::new();
        match self {
            MathExpressionTree::Atom(i) => match i {
                MathExpression::Ci(x) => {
                    content_mathml.push_str(&format!("<ci>{}</ci>", x.content));
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
                    Operator::Exp => operation.push_str("exp"),
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

    /// Translates a MathExpressionTree struct to a LaTeX expression.
    pub fn to_latex(&self) -> String {
        let mut expression = String::new();
        match self {
            MathExpressionTree::Atom(i) => {
                process_math_expression(i, &mut expression);
            }
            MathExpressionTree::Cons(head, rest) => {
                match head {
                    Operator::Add => {
                        for (index, r) in rest.iter().enumerate() {
                            if let MathExpressionTree::Cons(op, _) = r {
                                if is_unary_operator(op) {
                                    expression.push_str(&format!("{}", r.to_latex()));
                                } else if let Operator::Add = op {
                                    expression.push_str(&format!("{}", r.to_latex()));
                                } else {
                                    expression.push_str(&format!("({})", r.to_latex()));
                                }
                            } else {
                                expression.push_str(&format!("{}", r.to_latex()));
                            }

                            // Add "+" if it's not the last element
                            if index < rest.len() - 1 {
                                expression.push_str("+");
                            }
                        }
                    }
                    Operator::Subtract => {
                        for (index, r) in rest.iter().enumerate() {
                            process_expression_parentheses(&mut expression, r);
                            // Add "-" if it's not the last element
                            if index < rest.len() - 1 {
                                expression.push_str("-");
                            }
                        }
                    }
                    Operator::Multiply => {
                        for (index, r) in rest.iter().enumerate() {
                            if let MathExpressionTree::Cons(op, _) = r {
                                if is_unary_operator(op) {
                                    expression.push_str(&format!("{}", r.to_latex()));
                                } else if let Operator::Multiply = op {
                                    expression.push_str(&format!("{}", r.to_latex()));
                                } else if let Operator::Divide = op {
                                    expression.push_str(&format!("{}", r.to_latex()));
                                } else if let Operator::Dot = op {
                                    expression.push_str(&format!("{}", r.to_latex()));
                                } else {
                                    expression.push_str(&format!("({})", r.to_latex()));
                                }
                            } else {
                                expression.push_str(&format!("{}", r.to_latex()));
                            }
                            // Add "*" if it's not the last element
                            if index < rest.len() - 1 {
                                expression.push_str("*");
                            }
                        }
                    }
                    Operator::Equals => {
                        expression.push_str(&format!("{}", rest[0].to_latex()));
                        expression.push_str("=");
                        expression.push_str(&format!("{}", rest[1].to_latex()));
                    }
                    Operator::Divide => {
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str("/");
                        process_expression_parentheses(&mut expression, &rest[1]);
                    }
                    Operator::Exp => {
                        expression.push_str("\\mathrm{exp}");
                        expression.push_str(&format!("{{{}}}", rest[0].to_latex()));
                    }
                    Operator::Sqrt => {
                        expression.push_str("\\sqrt");
                        expression.push_str(&format!("{{{}}}", rest[0].to_latex()));
                    }
                    Operator::Lparen => {
                        expression.push_str("(");
                        expression.push_str(&format!("{}", rest[0].to_latex()));
                    }
                    Operator::Rparen => {
                        expression.push_str(")");
                        expression.push_str(&format!("{}", rest[0].to_latex()));
                    }
                    Operator::Compose => {
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str("_");
                        process_expression_parentheses(&mut expression, &rest[1]);
                    }
                    Operator::Factorial => {
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str("!");
                    }
                    Operator::Power => {
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str("^");
                        expression.push_str(&format!("{{{}}}", rest[1].to_latex()));
                    }
                    Operator::Comma => {
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str(",");
                        process_expression_parentheses(&mut expression, &rest[1]);
                    }
                    Operator::Grad => {
                        expression.push_str("\\nabla");
                        expression.push_str(&format!("{{{}}}", rest[0].to_latex()));
                    }
                    Operator::Dot => {
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str(" \\cdot ");
                        process_expression_parentheses(&mut expression, &rest[1]);
                    }
                    Operator::Abs => {
                        expression.push_str(&format!("\\left|{}\\right|", rest[0].to_latex()));
                    }
                    Operator::Derivative(d) => {
                        expression.push_str("\\frac{d ");
                        process_expression_parentheses(&mut expression, &rest[0]);
                        expression.push_str("}{d");
                        process_math_expression(&*d.bound_var.content, &mut expression);
                        expression.push_str("}");
                    }
                    Operator::Sin => {
                        expression.push_str(&format!("\\sin({})", rest[0].to_latex()));
                    }
                    Operator::Cos => {
                        expression.push_str(&format!("\\cos({})", rest[0].to_latex()));
                    }
                    Operator::Tan => {
                        expression.push_str(&format!("\\tan({})", rest[0].to_latex()));
                    }
                    Operator::Sec => {
                        expression.push_str(&format!("\\sec({})", rest[0].to_latex()));
                    }
                    Operator::Csc => {
                        expression.push_str(&format!("\\csc({})", rest[0].to_latex()));
                    }
                    Operator::Cot => {
                        expression.push_str(&format!("\\cot({})", rest[0].to_latex()));
                    }
                    Operator::Arcsin => {
                        expression.push_str(&format!("\\arcsin({})", rest[0].to_latex()));
                    }
                    Operator::Arccos => {
                        expression.push_str(&format!("\\arccos({})", rest[0].to_latex()));
                    }
                    Operator::Arctan => {
                        expression.push_str(&format!("\\arctan({})", rest[0].to_latex()));
                    }
                    Operator::Arcsec => {
                        expression.push_str(&format!("\\arcsec({})", rest[0].to_latex()));
                    }
                    Operator::Arccsc => {
                        expression.push_str(&format!("\\arccsc({})", rest[0].to_latex()));
                    }
                    Operator::Arccot => {
                        expression.push_str(&format!("\\arccot({})", rest[0].to_latex()));
                    }
                    Operator::Mean => {
                        expression.push_str(&format!("\\langle {} \\rangle", rest[0].to_latex()));
                    }
                    _ => {
                        return "Contain unsupported operators.".to_string();
                    }
                }
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
#[derive(Debug)]
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
            MathExpression::Differential(x) => {
                tokens.push(MathExpression::Mo(Operator::Lparen));
                if x.diff == Box::new(MathExpression::Mo(Operator::Grad)) {
                    tokens.push(MathExpression::Mo(Operator::Grad));
                } else {
                    x.diff.flatten(tokens);
                }
                x.func.flatten(tokens);
                tokens.push(MathExpression::Mo(Operator::Rparen));
            }
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
                if let MathExpression::Ci(x) = &**base {
                    if x.content == Box::new(MathExpression::Mi(Mi("e".to_string()))) {
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
                    //tokens.push(MathExpression::Mo(Operator::Lparen));
                    base.flatten(tokens);
                    tokens.push(MathExpression::Mo(Operator::Power));
                    tokens.push(MathExpression::Mo(Operator::Lparen));
                    superscript.flatten(tokens);
                    tokens.push(MathExpression::Mo(Operator::Rparen));
                }
            }
            MathExpression::AbsoluteSup(base, superscript) => {
                tokens.push(MathExpression::Mo(Operator::Lparen));
                tokens.push(MathExpression::Mo(Operator::Abs));
                tokens.push(MathExpression::Mo(Operator::Lparen));
                base.flatten(tokens);
                tokens.push(MathExpression::Mo(Operator::Rparen));
                tokens.push(MathExpression::Mo(Operator::Rparen));
                tokens.push(MathExpression::Mo(Operator::Power));
                tokens.push(MathExpression::Mo(Operator::Lparen));
                superscript.flatten(tokens);
                tokens.push(MathExpression::Mo(Operator::Rparen));
            }
            MathExpression::Mover(base, over) => {
                if MathExpression::Mo(Operator::Mean) == **over {
                    tokens.push(MathExpression::Mo(Operator::Mean));
                    tokens.push(MathExpression::Mo(Operator::Lparen));
                    base.flatten(tokens);
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
    insert_multiple_between_paren(&mut lexer);
    let mut result: MathExpressionTree = expr_bp(&mut lexer, 0);
    let mut math_vec: Vec<MathExpressionTree> = vec![];
    while lexer.next() != Token::Eof {
        let mut math_result = expr_bp(&mut lexer, 0);
        math_vec.push(math_result.clone());
    }

    if !math_vec.is_empty() {
        result = MathExpressionTree::Cons(Operator::Multiply, math_vec);
    }

    result
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

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (_, math) = interpreted_math(input.into()).unwrap();
        Ok(MathExpressionTree::from(math))
    }
}

/// Inserts an `Operator::Multiply` token between adjacent `Operator::Lparen` and `Operator::Rparen` tokens in the given Lexer.
fn insert_multiple_between_paren(lexer: &mut Lexer) {
    let mut new_tokens = Vec::new();
    let mut iter = lexer.tokens.iter().peekable();

    while let Some(token) = iter.next() {
        new_tokens.push(token.clone());

        if let Some(next_token) = iter.peek() {
            if let Token::Op(Operator::Lparen) = token {
                if let Token::Op(Operator::Rparen) = **next_token {
                    new_tokens.push(Token::Op(Operator::Multiply).clone());
                } else {
                }
            }
        }
    }
    lexer.tokens = new_tokens;
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
        Operator::Exp => ((), 21),
        Operator::Cos => ((), 21),
        Operator::Sin => ((), 21),
        Operator::Tan => ((), 21),
        Operator::Mean => ((), 25),
        Operator::Dot => ((), 25),
        Operator::Grad => ((), 25),
        Operator::Derivative(Derivative { .. }) => ((), 25),
        Operator::Div => ((), 25),
        Operator::Abs => ((), 25),
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
    let cmml = ode.to_cmml();
    let FirstOrderODE {
        lhs_var: _,
        func_of: _,
        with_respect_to: _,
        rhs,
    } = first_order_ode(input.into()).unwrap().1;
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
    assert_eq!(cmml, "<apply><eq/><apply><diff/><bvar>t</bar><ci>E</ci></apply><apply><minus/><apply><divide/><apply><times/><apply><times/><ci>β</ci><ci>I</ci></apply><ci>S</ci></apply><ci>N</ci></apply><apply><times/><ci>δ</ci><ci>E</ci></apply></apply></apply>");
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
        "<apply><eq/><ci>β</ci><apply><times/><ci>κ</ci><ci>m</ci></apply></apply>"
    );
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(= β (* κ m))");
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
    println!("exp={:?}", exp);
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
    let math = exp.to_infix_expression();
    let s_exp = exp.to_string();
    assert_eq!(math, "(S/N)");
    assert_eq!(s_exp, "(/ S N)");
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
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(^ x 3)");
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
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(exp (* (* (- (- 1 α)) γ) I))");
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
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(Cos x)");
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
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(Cos x)");
}

#[test]
fn test_mover_mean() {
    let input = "
    <math>
        <mover>
        <mi>T</mi>
        <mo>¯</mo>
        </mover>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(Mean T)");
}

#[test]
fn test_one_dimensional_ebm() {
    let input = "
    <math>
        <mi>C</mi>
        <mfrac>
        <mrow><mi>∂</mi><mi>T</mi><mo>(</mo><mi>ϕ</mi><mo>,</mo><mi>t</mi><mo>)</mo></mrow>
        <mrow><mi>∂</mi><mi>t</mi></mrow>
        </mfrac>
        <mo>=</mo>
        <mo>(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo>)</mo><mi>S</mi><mo>(</mo><mi>ϕ</mi><mo>,</mo><mi>t</mi><mo>)</mo>
        <mo>-</mo>
        <mo>(</mo><mi>A</mi><mo>+</mo><mi>B</mi><mi>T</mi><mo>(</mo><mi>ϕ</mi><mo>,</mo><mi>t</mi><mo>)</mo><mo>)</mo>
        <mo>+</mo>
        <mfrac>
        <mn>1</mn>
        <mrow><mi>cos</mi><mi>ϕ</mi></mrow>
        </mfrac>
        <mfrac>
        <mi>∂</mi>
        <mrow><mi>∂</mi><mi>ϕ</mi></mrow>
        </mfrac>
        <mo>(</mo>
        <mrow><mi>cos</mi><mi>ϕ</mi></mrow>
        <mfrac>
        <mrow><mi>∂</mi><mi>T</mi></mrow>
        <mrow><mi>∂</mi><mi>ϕ</mi></mrow>
        </mfrac>
        <mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    println!("S-exp={:?}", s_exp);
}

#[test]
fn test_absolute_value() {
    let input = "
    <math>
        <mo>|</mo><mo>&#x2207;</mo><mi>H</mi><mo>|</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(Grad H)");
}

#[test]
fn test_grad() {
    let input = "
    <math>
        <mo>&#x2207;</mo><mi>H</mi>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(Grad H)");
}

#[test]
fn test_absolute_as_msup() {
    let input = "
    <math>
        <mo>|</mo><mrow><mo>&#x2207;</mo><mi>H</mi></mrow>
        <msup><mo>|</mo>
        <mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msup>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(^ (Abs (Grad H)) (- n 1))");
}

#[test]
fn test_equation_halfar_dome() {
    let input = "
    <math>
        <mfrac><mrow><mi>∂</mi><mi>H</mi></mrow><mrow><mi>∂</mi><mi>t</mi></mrow></mfrac>
        <mo>=</mo>
        <mo>&#x2207;</mo>
        <mo>&#x22c5;</mo>
        <mo>(</mo>
        <mi>Γ</mi>
        <msup><mi>H</mi><mrow><mi>n</mi><mo>+</mo><mn>2</mn></mrow></msup>
        <mo>|</mo><mrow><mo>&#x2207;</mo><mi>H</mi></mrow>
        <msup><mo>|</mo>
        <mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msup>
        <mo>&#x2207;</mo><mi>H</mi>
        <mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(
        s_exp,
        "(= (D(1, t) H) (Div (* (* (* Γ (^ H (+ n 2))) (^ (Abs (Grad H)) (- n 1))) (Grad H))))"
    );
}

#[test]
fn test_halfar_dome_rhs() {
    let input = "
    <math>
        <mo>&#x2207;</mo>
        <mo>&#x22c5;</mo>
        <mo>(</mo>
        <mi>Γ</mi>
        <msup><mi>H</mi><mrow><mi>n</mi><mo>+</mo><mn>2</mn></mrow></msup>
        <mo>|</mo><mrow><mo>&#x2207;</mo><mi>H</mi></mrow>
        <msup><mo>|</mo>
        <mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msup>
        <mo>&#x2207;</mo><mi>H</mi>
        <mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(
        s_exp,
        "(Div (* (* (* Γ (^ H (+ n 2))) (^ (Abs (Grad H)) (- n 1))) (Grad H)))"
    );
}

#[test]
fn test_func_paren() {
    let input = "
    <math>
        <mo>(</mo><mi>a</mi><mo>+</mo><mo>(</mo><mi>b</mi><mo>+</mo><mi>c</mi><mo>)</mo>
        <mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(+ a (+ b c))");
}

#[test]
fn test_func_paren_and_times() {
    let input = "
    <math>
        <mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>(</mo><mi>a</mi><mo>+</mo><mi>b</mi><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(* S (+ a b))");
}

#[test]
fn test_func_a_plus_b() {
    let input = "
    <math>
        <mo>(</mo><mi>a</mi><mo>+</mo><mi>b</mi><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(+ a b)");
}

#[test]
fn test_func_paren_and_times_another() {
    let input = "
    <math>
        <mi>S</mi><mo>(</mo><mi>t</mi><mo>)</mo>
        <mo>(</mo><mi>a</mi><mi>I</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>+</mo><mi>b</mi><mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>)</mo>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(* S (+ (* a I) (* b R)))");
}

#[test]
fn test_divergence() {
    let input = "
    <math>
        <mo>&#x2207;</mo><mo>&#x22c5;</mo>
        <mi>H</mi>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(Div H)");
}

#[test]
fn test_combination() {
    let input = "
    <math>
        <mi>S</mi>
        <mrow><mi>n</mi><mo>+</mo><mn>4</mn></mrow>
        <mrow><mi>i</mi><mo>-</mo><mn>3</mn></mrow>
        <msup><mi>H</mi><mrow><mi>m</mi><mo>-</mo><mn>2</mn></mrow></msup>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    assert_eq!(s_exp, "(* (* (* S (+ n 4)) (- i 3)) (^ H (- m 2)))");
}

#[test]
fn test_mi_multiply() {
    let input = "
    <math>
        <mi>A</mi>
        <msup><mi>ρ</mi><mi>n</mi></msup>
        <msup><mi>g</mi><mi>n</mi></msup>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let s_exp = exp.to_string();
    println!("s_exp={:?}", s_exp);
}

#[test]
fn test_unicode_conversion() {
    let input1 = "&#x039B; is a Greek letter.";
    let input2 = "&#x03bb; is another Greek letter.";
    let input3 = "Λ and λ are Greek letters.";
    let input4 = "Lambda and lambda are English representations of Greek letters.";

    assert_eq!(unicode_to_latex(input1), "\\Lambda is a Greek letter.");
    assert_eq!(
        unicode_to_latex(input2),
        "\\lambda is another Greek letter."
    );
    assert_eq!(
        unicode_to_latex(input3),
        "\\Lambda and \\lambda are Greek letters."
    );
    assert_eq!(
        unicode_to_latex(input4),
        "\\Lambda and \\lambda are English representations of Greek letters."
    );
}
#[test]
fn test_sexp2latex() {
    let input = "
    <math>
        <mi>&#x03bb;</mi>
        <mrow><mi>n</mi><mo>+</mo><mn>4</mn></mrow>
        <mrow><mi>i</mi><mo>-</mo><mn>3</mn></mrow>
        <msup><mi>H</mi><mrow><mi>m</mi><mo>-</mo><mn>2</mn></mrow></msup>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let latex_exp = exp.to_latex();
    assert_eq!(latex_exp, "\\lambda*(n+4)*(i-3)*H^{m-2}");
}

#[test]
fn test_sexp2latex_derivative() {
    let input = "
    <math>
    <mfrac>
        <mrow><mi>d</mi><mi>S</mi></mrow>
        <mrow><mi>d</mi><mi>t</mi></mrow>
        </mfrac>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let latex_exp = exp.to_latex();
    assert_eq!(latex_exp, "\\frac{d S}{dt}");
}
