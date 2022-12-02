use actix_web::{put, web, HttpResponse};
use std::string::String;
use mathml::parsing::parse;
use utoipa;
use utoipa::ToSchema;
use petgraph::dot::{Config, Dot};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MathMLParseRequest {
    pub input: String,
}


#[utoipa::path(
    request_body = MathMLParseRequest,
    responses(
        (status = 200, description = "Parse MathML and return a DOT representation", body = String)
    )
)]
#[put("/parse-mathml")]
pub async fn parse_mathml(payload: web::Json<MathMLParseRequest>) -> HttpResponse {
    let contents = &payload.input;
    let (_, math) = parse(&contents).expect(format!("Unable to parse payload!").as_str());

    let g = math.to_graph(); 
    let dot_representation = Dot::with_config(&g, &[Config::EdgeNoLabel]);
    
    return HttpResponse::Ok().body(dot_representation.to_string());
}
