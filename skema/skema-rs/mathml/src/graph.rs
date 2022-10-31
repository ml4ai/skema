use crate::ast::{
    Math, MathExpression,
    MathExpression::{
        Mfrac, Mi, Mn, Mo, Mover, Mrow, Msqrt, Msub, Msubsup, Msup, Mtext, Mstyle, Munder, Mspace, MoLine,
    },
};

use petgraph::{graph::NodeIndex, Graph};

pub type MathMLGraph<'a> = Graph<&'a str, u32>;

impl<'a> MathExpression<'a> {
    pub fn add_to_graph(&self, graph: &mut MathMLGraph<'a>, mut parent_index: Option<NodeIndex>) {
        match self {
            Mi(x) => {
                let node_index = graph.add_node(x);
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
            }
            Mo(x) => {
                let node_index = graph.add_node(x);
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
            }
            Mn(x) => {
                let node_index = graph.add_node(x);
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
            }

            Msqrt(contents) => {
                let node_index = graph.add_node("msqrt");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                contents.add_to_graph(graph, parent_index);
            }

            Msup(base, superscript) => {
                let node_index = graph.add_node("msup");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                base.add_to_graph(graph, parent_index);
                superscript.add_to_graph(graph, parent_index);
            }

            Msub(base, subscript) => {
                let node_index = graph.add_node("msub");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                base.add_to_graph(graph, parent_index);
                subscript.add_to_graph(graph, parent_index);
            }

            Mfrac(numerator, denominator) => {
                let node_index = graph.add_node("mfrac");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                numerator.add_to_graph(graph, parent_index);
                denominator.add_to_graph(graph, parent_index);
            }

            Mrow(elements) => {
                let node_index = graph.add_node("mrow");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                for element in elements {
                    element.add_to_graph(graph, parent_index);
                }
            }

            Munder(elements) => {
                let node_index = graph.add_node("munder");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                for element in elements {
                    element.add_to_graph(graph, parent_index);
                }
            }

            Mover(elements) => {
                let node_index = graph.add_node("mover");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                for element in elements {
                    element.add_to_graph(graph, parent_index);
                }
            }

            Msubsup(elements) => {
                let node_index = graph.add_node("msubsup");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                for element in elements {
                    element.add_to_graph(graph, parent_index);
                }
            }

            Mtext(x) => {
                let node_index = graph.add_node(x);
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
            }

	    
	    Mstyle(elements) => {
                let node_index = graph.add_node("mrow");
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
                parent_index = Some(node_index);
                for element in elements {
                    element.add_to_graph(graph, parent_index);
                }
            }
	    
	    Mspace(x) => {
                let node_index = graph.add_node(x);
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
            }

	    MoLine(x) => {
                let node_index = graph.add_node(x);
                if let Some(p) = parent_index {
                    graph.add_edge(p, node_index, 1);
                }
            }


        }
    }
}

impl<'a> Math<'a> {
    pub fn to_graph(&self) -> MathMLGraph {
        let mut g = MathMLGraph::new();
        let root_index = g.add_node("root");
        for element in &self.content {
            element.add_to_graph(&mut g, Some(root_index));
        }
        g
    }
}
