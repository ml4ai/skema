/// Functionality for normalizing MathExpression enums.
use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mn, Mo, Msub},
    Mi, Mrow,
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
                *self = MathExpression::Mi(Mi(combined));
            }

            MathExpression::Mrow(Mrow(xs)) => {
                for x in xs {
                    x.collapse_subscripts();
                }
            }
            _ => (),
        }
    }

    /// Get the string representation of a MathExpression
    pub fn get_string_repr(&self) -> String {
        match self {
            MathExpression::Mi(Mi(x)) => x.to_string(),
            Mo(x) => x.to_string(),
            Mn(x) => x.to_string(),
            MathExpression::Mrow(Mrow(xs)) => xs
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
    use crate::ast::Operator;
    assert_eq!(
        MathExpression::Mi(Mi("t".to_string())).get_string_repr(),
        "t".to_string()
    );
    assert_eq!(Mo(Operator::Add).get_string_repr(), "+");
    assert_eq!(
        MathExpression::Mrow(Mrow(vec![
            MathExpression::Mi(Mi("t".into())),
            Mo(Operator::Add),
            MathExpression::Mi(Mi("1".to_string()))
        ]))
        .get_string_repr(),
        "t+1".to_string()
    );
}

#[test]
fn test_subscript_collapsing() {
    use crate::ast::Operator;
    let mut expr = Msub(
        Box::new(MathExpression::Mi(Mi("S".to_string()))),
        Box::new(MathExpression::Mrow(Mrow(vec![
            MathExpression::Mi(Mi("t".to_string())),
            Mo(Operator::Add),
            MathExpression::Mi(Mi("1".to_string())),
        ]))),
    );
    expr.collapse_subscripts();
    assert_eq!(expr, MathExpression::Mi(Mi("S_{t+1}".to_string())));
}

#[test]
fn test_normalize() {
    use crate::ast::Operator;
    let contents = std::fs::read_to_string("tests/sir.xml").unwrap();
    let mut math = contents.parse::<Math>().unwrap();
    math.normalize();
    assert_eq!(
        &math.content[0],
        &MathExpression::Mrow(Mrow(vec![
            MathExpression::Mi(Mi("S_{t+1}".to_string())),
            Mo(Operator::Equals),
            MathExpression::Mi(Mi("S_{t}".to_string())),
            Mo(Operator::Subtract),
            MathExpression::Mi(Mi("β".to_string())),
            MathExpression::Mi(Mi("S_{t}".to_string())),
            MathExpression::Mi(Mi("I_{t}".to_string())),
        ]))
    );
}
