use serde::{Deserialize, Serialize};
use actix_web::{web, HttpResponse};
use comment_extraction::languages::python::get_comments_from_string as get_python_comments;

#[derive(Debug, Serialize, Deserialize)]
pub struct CommentExtractionRequest {
    pub language: String,
    pub code: String,
}

impl CommentExtractionRequest {
    pub fn new(language: &str, code: String) -> Self {
        Self {
            language: language.to_string(),
            code
        }
    }
}
pub async fn get_comments(payload: web::Json<CommentExtractionRequest>) -> HttpResponse {
    HttpResponse::Ok().json(web::Json(get_python_comments(&payload.code)))
}

