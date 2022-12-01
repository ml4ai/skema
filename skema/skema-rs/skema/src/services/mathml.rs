use actix_web::{get, web, HttpResponse};
use std::string::String;
use mathml::parsing::parse;
use serde::{Deserialize, Serialize};
use utoipa;
use utoipa::ToSchema;
use petgraph::dot::{Config, Dot};

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MathmlParseRequest {
    pub input: String,
}

impl MathmlParseRequest {
    pub fn new(input: String) -> Self {
        Self { input }
    }
}


#[utoipa::path(
    request_body = MatmlVisualizeRequest,
    responses(
        (status = 200, description = "Visualize XML", body = String)
    )
)]
#[get("/mathml_parse")]
pub async fn mathml_parse(payload: web::Json<MathmlParseRequest>) -> HttpResponse {
    let contents = &payload.input;
    let (_, mut math) = parse(&contents).expect(format!("Unable to parse file {contents}!").as_str());
    math.normalize();

    let g = math.to_graph(); 
    let dot_representation = Dot::with_config(&g, &[Config::EdgeNoLabel]);
    
    return HttpResponse::Ok().body(dot_representation.to_string());
}