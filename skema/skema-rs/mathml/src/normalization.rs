use crate::ast::{
    Math, MathExpression,
    MathExpression::{
        Mfrac, Mi, Mn, Mo, MoLine, Mover, Mrow, Mspace, Msqrt, Mstyle, Msub, Msubsup, Msup, Mtext,
        Munder,
    },
};
use crate::parsing::parse;

impl MathExpression {
    fn collapse_subscripts(&mut self) {
        match self {
            Msub(base, subscript) => {
                let mut combined = String::from(&base.get_string_repr());
                combined.push_str(&subscript.get_string_repr());
                *self = Mi(combined);
                //Some(Mi(combined))
            }
            //Mn(x) => add_to_graph_0(graph, parent_index, x),
            //Msqrt(x) => add_to_graph_n!(graph, parent_index, "msqrt", x),
            //Msup(x1, x2) => add_to_graph_n!(graph, parent_index, "msup", x1, x2),
            //Msub(x1, x2) => add_to_graph_n!(graph, parent_index, "msub", x1, x2),
            //Mfrac(x1, x2) => add_to_graph_n!(graph, parent_index, "mfrac", x1, x2),
            //Mrow(xs) => add_to_graph_many0(graph, parent_index, "mrow", xs),
            //Munder(xs) => add_to_graph_many0(graph, parent_index, "munder", xs),
            //Mover(xs) => add_to_graph_many0(graph, parent_index, "mover", xs),
            //Msubsup(xs) => add_to_graph_many0(graph, parent_index, "msubsup", xs),
            //Mtext(x) => add_to_graph_0(graph, parent_index, x),
            //Mstyle(xs) => add_to_graph_many0(graph, parent_index, "mstyle", xs),
            //Mspace(x) => add_to_graph_0(graph, parent_index, x),
            //MoLine(x) => add_to_graph_0(graph, parent_index, x),
            _ => (),
        }
    }

    fn get_string_repr(&self) -> String {
        match self {
            Mi(x) => x.to_string(),
            Mo(x) => x.to_string(),
            Mrow(xs) => xs
                .iter()
                .map(|x| x.get_string_repr())
                .collect::<Vec<String>>()
                .join(""),
            _ => {
                panic!("Unhandled type!");
            }
        }
    }
}

fn normalize(math: &mut Math) {
    for expr in &mut math.content {
        expr.collapse_subscripts();
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
    assert_eq!(expr, Mi("St+1".to_string()));
}

#[test]
fn test_normalize() {
    let mut math = Math {
        content: vec![Msub(
            Box::new(Mi("S".to_string())),
            Box::new(Mrow(vec![
                Mi("t".to_string()),
                Mo("+".to_string()),
                Mi("1".to_string()),
            ])),
        )],
    };
    normalize(&mut math);
    assert_eq!(&math.content[0], &Mi("St+1".to_string()));
}
