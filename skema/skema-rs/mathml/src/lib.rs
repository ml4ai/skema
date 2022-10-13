//! Library for MathML parsing

pub mod ast;
pub mod graph;
pub mod parsing;

#[test]
fn test_parser() {
    use crate::{
        ast::{Math, MathExpression::*},
        parsing::parse,
    };
    assert_eq!(
        parse(
            "<math>
                <mrow>
                    <mo>-</mo>
                    <mi>b</mi>
                </mrow>
            </math>"
        )
        .unwrap()
        .1,
        Math {
            content: vec![Mrow(vec![Mo("-"), Mi("b")])]
        }
    )
}
