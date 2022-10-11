use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mo, Mrow, Msub},
};

use petgraph::Graph;

type MathMLGraph<'a> = Graph<&'a str, &'a str>;

impl<'a> MathExpression<'a> {
    pub fn traverse(&self, mut G: MathMLGraph<'a>) {
        match self {
            MathExpression::Mi(x) => {
                let g = G.add_node(x);
                G.add_node(x);
                dbg!(G);
            }
            MathExpression::Mrow(xs) => {
                for elem in xs {
                    dbg!(elem);
                }
            }
            _ => (),
        }
    }
}

#[test]
fn test_graph_building() {
    let mut G = MathMLGraph::new();
    let m = Mrow(vec![Mo("-"), Mi("b")]);
    m.traverse(G);
}
