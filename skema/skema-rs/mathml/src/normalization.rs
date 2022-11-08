use crate::parsing::parse;
use crate::ast::{
    Math,
    MathExpression,
    MathExpression::{
        Mfrac, Mi, Mn, Mo, MoLine, Mover, Mrow, Mspace, Msqrt, Mstyle, Msub, Msubsup, Msup, Mtext,
        Munder,
    },
};


impl<'a> MathExpression<'a> {
    fn collapse_subscripts(&self, storage: &'a str) -> Option<MathExpression> {
        match self {
            Msub(base, subscript) => {
                storage.to_owned().push_str(&base.get_string_repr());
                storage.to_owned().push_str(&subscript.get_string_repr());
                Some(Mi(storage))
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

//fn normalize(math: Math) {
    //for expr in math.content {
        //let storage = String::new();
        //if let Some(e) = expr.collapse_subscripts() {
            //*expr = e;
        //}
    //}
//}

#[test]
fn test_get_string_repr() {
    assert_eq!(Mi("t").get_string_repr(), "t");
    assert_eq!(Mo("+").get_string_repr(), "+");
    assert_eq!(Mrow(vec![Mi("t".into()), Mo("+".into()), Mi(&"1")]).get_string_repr(), "t+1");
}

#[test]
fn test_subscript_collapsing() {
    let expr = Msub(Box::new(Mi("S")), Box::new(Mrow(vec![Mi("t"), Mo("+"), Mi("1")])));
    let mut storage = String::new();
    assert_eq!(expr.collapse_subscripts(&storage).unwrap(), Mi("St+1"));
}
