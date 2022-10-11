use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mo, Mrow, Msub},
};

use petgraph::Graph;

impl<'a> MathExpression<'a> {
    pub fn traverse(&self) {
        match self {
            MathExpression::Mi(_) => {
                dbg!(self);
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

//#[test]
//fn test_graph_building() {}
