/// Functionality for normalizing MathExpression enums.

use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mi, Mn, Mo, Mrow, Msub},
};

impl MathExpression {
    /// Collapse subscripts
    fn collapse_subscripts(&mut self) {
        match self {
            Msub(base, subscript) => {
                let mut combined = String::from(&base.get_string_repr());
                combined.push_str("_{");
                combined.push_str(&subscript.get_string_repr());
                combined.push('}');
                *self = Mi(combined);
            }

            Mrow(xs) => {
                for x in xs {
                    x.collapse_subscripts();
                }
            }
            _ => (),
        }
    }

    /// Get the string representation of a MathExpression
    fn get_string_repr(&self) -> String {
        match self {
            Mi(x) => x.to_string(),
            Mo(x) => x.to_string(),
            Mn(x) => x.to_string(),
            Mrow(xs) => xs
                .iter()
                .map(|x| x.get_string_repr())
                .collect::<Vec<String>>()
                .join(""),
            _ => {
                panic!("The method 'get_string_repr' for the MathExpression enum only handles the following MathML element types: [mi, mo, mn, mrow]!");
            }
        }
    }
}

impl Math {
    /// Normalize the math expression, performing the following steps:
    /// 1. Collapse subscripts assuming we don't need their substructure.
    pub fn normalize(&mut self) {
        for expr in &mut self.content {
            expr.collapse_subscripts();
        }
    }
}
#[test]
fn test_get_string_repr() {
    assert_eq!(Mi("t".to_string()).get_string_repr(), "t".to_string());
    assert_eq!(Mo("+".to_string()).get_string_repr(), "+".to_string());
    assert_eq!(
        Mrow(vec![Mi("t".into()), Mo("+".into()), Mi("1".to_string())]).get_string_repr(),
        "t+1".to_string()
    );
}

#[test]
fn test_subscript_collapsing() {
    let mut expr = Msub(
        Box::new(Mi("S".to_string())),
        Box::new(Mrow(vec![
            Mi("t".to_string()),
            Mo("+".to_string()),
            Mi("1".to_string()),
        ])),
    );
    expr.collapse_subscripts();
    assert_eq!(expr, Mi("S_{t+1}".to_string()));
}

#[test]
fn test_normalize() {
    use crate::parsing::parse;
    let contents = std::fs::read_to_string("tests/sir.xml").unwrap();
    let (_, mut math) = parse(&contents).unwrap();
    math.normalize();
    assert_eq!(
        &math.content[0],
        &Mrow(vec![
            Mi("S_{t+1}".to_string()),
            Mo("=".to_string()),
            Mi("S_{t}".to_string()),
            Mo("-".to_string()),
            Mi("β".to_string()),
            Mi("S_{t}".to_string()),
            Mi("I_{t}".to_string()),
        ])
    );
}
