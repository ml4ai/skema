use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mo, Mrow, Msub},
};

use petgraph::Graph;

pub type MathMLGraph<'a> = Graph<&'a str, &'a str>;

impl<'a> MathExpression<'a> {
    pub fn add_to_graph(&self, graph: &mut MathMLGraph<'a>) {
        match self {
            MathExpression::Mi(x) => {
                graph.add_node(x);
            }
            MathExpression::Mo(x) => {
                graph.add_node(x);
            }
            MathExpression::Mn(x) => {
                graph.add_node(x);
            }

            MathExpression::Mfrac(numerator, denominator) => {
                numerator.add_to_graph(graph);
                denominator.add_to_graph(graph);
            }
            MathExpression::Mrow(xs) => {
                for x in xs {
                    x.add_to_graph(graph);
                }
            }
            _ => (panic!("Unhandled type!")),
        }
    }
}

impl<'a> Math<'a> {
    pub fn to_graph(&self) -> MathMLGraph {
        let mut g = MathMLGraph::new();
        for element in &self.content {
            element.add_to_graph(&mut g);
        }
        g
    }
}

#[test]
fn test_graph_building() {
    let mut G = MathMLGraph::new();
    let m = Mrow(vec![Mo("-"), Mi("b")]);
    m.add_to_graph(&mut G);
}
