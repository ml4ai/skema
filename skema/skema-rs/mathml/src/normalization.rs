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


impl<'a> MathExpression<'a> {
    fn collapse_subscripts(&self, storage: &mut Vec<Rc<String>>) -> Option<MathExpression> {
        match self {
            Msub(base, subscript) => {
                let mut combined = Rc::new(String::from(&base.get_string_repr()));
                combined.push_str(&subscript.get_string_repr());
                storage.push(Rc::clone(&combined));
                Some(Mi(&combined))
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
        let mut storage = Vec::<Rc<String>>::new();
        if let Some(e) = &mut expr.collapse_subscripts(&mut storage) {
            expr = *e;
        }
    }
}

#[test]
fn test_get_string_repr() {
    assert_eq!(Mi("t").get_string_repr(), "t");
    assert_eq!(Mo("+").get_string_repr(), "+");
    assert_eq!(Mrow(vec![Mi("t".into()), Mo("+".into()), Mi("1")]).get_string_repr(), "t+1");
}

#[test]
fn test_subscript_collapsing() {
    let expr = Msub(Box::new(Mi("S")), Box::new(Mrow(vec![Mi("t"), Mo("+"), Mi("1")])));
    let mut storage = Vec::<Rc<String>>::new();
    assert_eq!(expr.collapse_subscripts(&mut storage).unwrap(), Mi("St+1"));
}
