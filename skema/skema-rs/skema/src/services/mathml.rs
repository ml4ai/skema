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
    let (_, mut math) = parse(&contents).expect(format!("Unable to parse payload!").as_str());
    math.normalize();
    let g = &mut math.content[0].clone().to_graph();
    let dot_representation = Dot::new(&*g);
    dot_representation.to_string()
}
