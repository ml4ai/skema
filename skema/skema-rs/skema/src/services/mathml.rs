use actix_web::{get, web, HttpResponse};
use std::string::String;
use mathml::parsing;
use serde::{Deserialize, Serialize};
use utoipa;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MatmlVisualizeRequest {
    pub xmlString: String,
}

impl MatmlVisualizeRequest {
    pub fn new(xmlString: String) -> Self {
        Self { xmlString }
    }
}

#[utoipa::path(
    request_body = MatmlVisualizeRequest,
    responses(
        (status = 200, description = "Visualize XML", body = String)
    )
)]
#[get("/visualize_xml")]
pub async fn visualize_xml(payload: web::Json<MatmlVisualizeRequest>) -> HttpResponse {
    HttpResponse::Ok().body("Hello")
}