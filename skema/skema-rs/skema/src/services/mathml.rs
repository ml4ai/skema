use actix_web::put;
use std::string::String;
use mathml::parsing::parse;
use utoipa;
use petgraph::dot::{Config, Dot};

/// Parse MathML and return a DOT representation of the abstract syntax tree (AST)
#[utoipa::path(
    request_body = String,
    responses(
        (
            status = 200,
            body = String
        )
    )
)]
#[put("/mathml/ast-graph")]
pub async fn get_ast_graph(payload: String) -> String {
    let contents = &payload;
    let (_, math) = parse(&contents).expect(format!("Unable to parse payload!").as_str());

    let g = math.to_graph();
    let dot_representation = Dot::with_config(&g, &[Config::EdgeNoLabel]);
    dot_representation.to_string()
}

/// Parse a MathML representation of an equation and return a DOT representation of the math
/// expression graph (MEG), which can be used to perform structural alignment with the scientific
/// model code that corresponds to the equation.
#[utoipa::path(
    request_body = String,
    responses(
        (
            status = 200,
            body = String
        )
    )
)]
#[put("/mathml/math-exp-graph")]
pub async fn get_math_exp_graph(payload: String) -> String {
    let contents = &payload;
    let (_, mut math) = parse(&contents).expect(format!("Unable to parse payload!").as_str());
    math.normalize();
    let g = &mut math.content[0].clone().to_graph();
    let dot_representation = Dot::new(&*g);
    dot_representation.to_string()
}
