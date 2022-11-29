use actix_web::{get, web, HttpResponse};
use std::string::String;
use mathml::parsing::parse;
use serde::{Deserialize, Serialize};
use utoipa;
use utoipa::ToSchema;
use petgraph::dot::{Config, Dot, Graph};

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
        (status = 200, description = "Visualize XML", body = Dot<&Graph<String, u32>>)
    )
)]
#[get("/mathml_parse")]
pub async fn visualize_xml(payload: web::Json<MathmlParseRequest>) -> HttpResponse {
    let contents = &payload.input;
    let (_, mut math) = parse(&contents).expect(format!("Unable to parse file {contents}!").as_str());
    math.normalize();

    let g = math.to_graph(); 
    let dotRepresentation = Dot::with_config(&g, &[Config::EdgeNoLabel]);
    
    return HttpResponse::Ok().json(web::Json(dotRepresentation));
}