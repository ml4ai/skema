use actix_web::put;
use std::string::String;
use mathml::parsing::parse;
use utoipa;
use petgraph::dot::{Config, Dot};

#[utoipa::path(
    request_body = String,
    responses(
        (status = 200, description = "Parse MathML and return a DOT representation", body = String)
    )
)]
#[put("/parse-mathml")]
pub async fn parse_mathml(payload: String) -> String {
    let contents = &payload;
    let (_, math) = parse(&contents).expect(format!("Unable to parse payload!").as_str());

    let g = math.to_graph(); 
    let dot_representation = Dot::with_config(&g, &[Config::EdgeNoLabel]);
    dot_representation.to_string()
}
