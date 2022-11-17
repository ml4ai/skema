use serde::{Deserialize, Serialize};
use actix_web::{get, web, Responder, Result, HttpResponse};
use comment_extraction::languages::python::get_comments_from_string as get_python_comments;

#[derive(Debug, Serialize, Deserialize)]
pub struct CommentExtractionRequest {
    pub language: String,
    pub source_code: String,
}

impl CommentExtractionRequest {
    pub fn new(language: &str, source_code: String) -> Self {
        Self {
            language: language.to_string(),
            source_code: source_code
        }
    }
}
pub async fn get_comments(payload: web::Json<CommentExtractionRequest>) -> HttpResponse {
    HttpResponse::Ok().json(web::Json(get_python_comments(&payload.source_code)))
}

