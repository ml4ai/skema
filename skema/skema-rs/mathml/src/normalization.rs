use crate::parsing::parse;
use crate::ast::{
    Math,
    MathExpression,
    MathExpression::{
        Mfrac, Mi, Mn, Mo, MoLine, Mover, Mrow, Mspace, Msqrt, Mstyle, Msub, Msubsup, Msup, Mtext,
        Munder,
    },
};
use std::rc::Rc;


impl MathExpression {
    fn collapse_subscripts(&self) -> Option<MathExpression> {
        match self {
            Msub(base, subscript) => {
                let mut combined = String::from(&base.get_string_repr());
                combined.push_str(&subscript.get_string_repr());
                Some(Mi(combined))
            }
            _ => None
        }
    }

    fn get_string_repr(&self) -> String {
        match self {
            Mi(x) => x.to_string(),
            Mo(x) => x.to_string(),
            Mrow(xs) => {
                xs.iter().map(|x| x.get_string_repr()).collect::<Vec<String>>().join("")
            }
            _ => {
                panic!("Unhandled type!");
            }
        }
    }

}

fn normalize(math: Math) {
    for mut expr in math.content {
        let mut storage = Vec::<String>::new();
        if let Some(e) = &mut expr.collapse_subscripts() {
            expr = *e;
        }
    }
}

#[test]
fn test_get_string_repr() {
    assert_eq!(Mi("t".to_string()).get_string_repr(), "t".to_string());
    assert_eq!(Mo("+".to_string()).get_string_repr(), "+".to_string());
    assert_eq!(Mrow(vec![Mi("t".into()), Mo("+".into()), Mi("1".to_string())]).get_string_repr(), "t+1".to_string());
}

#[test]
fn test_subscript_collapsing() {
    let expr = Msub(Box::new(Mi("S".to_string())), Box::new(Mrow(vec![Mi("t".to_string()), Mo("+".to_string()), Mi("1".to_string())])));
    let mut storage = Vec::<String>::new();
    assert_eq!(expr.collapse_subscripts().unwrap(), Mi("St+1".to_string()));
}
