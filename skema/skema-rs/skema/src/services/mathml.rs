use actix_web::{get, put, web, HttpResponse};
use mathml::{
    acset::ACSet,
    ast::Math,
    expression::{preprocess_content, wrap_math},
    parsing::parse,
};
use petgraph::dot::{Config, Dot};
use serde::{Deserialize, Serialize};
use std::string::String;
use utoipa;
use utoipa::ToSchema;

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
    let mut contents = payload.clone();
    contents = preprocess_content(contents);
    let (_, mut math) = parse(&contents).expect(format!("Unable to parse payload!").as_str());
    math.normalize();
    let mut new_math = wrap_math(math);
    let g = new_math.clone().to_graph();
    let dot_representation = Dot::new(&g);
    dot_representation.to_string()
}

/// Return a JSON representation of an ACSet constructed from an array of MathML strings.
#[utoipa::path(
    request_body = Vec<String>,
    responses(
        (
            status = 200,
            body = ACSet
        )
    )
)]
#[put("/mathml/acset")]
pub async fn get_acset(payload: web::Json<Vec<String>>) -> HttpResponse {
    let asts: Vec<Math> = payload.iter().map(|x| parse(&x).unwrap().1).collect();
    HttpResponse::Ok().json(web::Json(ACSet::from(asts)))
}
