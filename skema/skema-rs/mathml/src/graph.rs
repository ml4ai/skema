use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mo, Mrow, Msub},
};

use petgraph::Graph;

pub type MathMLGraph<'a> = Graph<&'a str, &'a str>;

//impl Math {
    //for element in math.content {
        //element.to_graph(&mut G);
    //}
//}

impl<'a> MathExpression<'a> {
    pub fn to_graph(&self, G: &mut MathMLGraph<'a>) {
        match self {
            MathExpression::Mi(x) => {
                G.add_node(x);
            }
            MathExpression::Mo(x) => {
                G.add_node(x);
            }
            MathExpression::Mrow(xs) => {
                for x in xs {
                    x.to_graph(G);
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
    m.to_graph(G);
}
