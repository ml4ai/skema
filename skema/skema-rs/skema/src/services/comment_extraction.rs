use serde::{Deserialize, Serialize};
use actix_web::{get, web, HttpResponse};
use comment_extraction::languages::python::get_comments_from_string as get_python_comments;
use comment_extraction::languages::python::Comments;
use utoipa::{ToSchema, IntoParams};
use utoipa;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CommentExtractionRequest {
    #[schema(example = "Python")]
    pub language: String,
    pub code: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SingleLineComment {
    line: u32,
    contents: String
}


#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct  Docstring {
        object_name: String,
        contents: Vec<String>,
}


#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CommentExtractionResponse {
    single_line_comments: Vec<SingleLineComment>,
    docstrings: Vec<Docstring>
}

impl CommentExtractionResponse {
    fn new(comments: Comments) -> Self {
        let mut single_line_comments = Vec::new();
        for (line, comment) in comments.comments.iter() {
            single_line_comments.push(SingleLineComment {line: *line, contents: comment.to_string()});
        }
        let mut docstrings = Vec::new();
        for (object_name, contents) in comments.docstrings {
            docstrings.push(Docstring {object_name: object_name, contents: contents});
        }
        Self {
            single_line_comments: single_line_comments,
            docstrings: docstrings
        }
    }
}


impl CommentExtractionRequest {
    pub fn new(language: &str, code: String) -> Self {
        Self {
            language: language.to_string(),
            code
        }
    }
}

#[utoipa::path(
    request_body = CommentExtractionRequest,
    responses(
        (status = 200, description = "Get comments", body = CommentExtractionResponse)
    )
)]
#[get("/get_comments")]
pub async fn get_comments(payload: web::Json<CommentExtractionRequest>) -> HttpResponse {
    HttpResponse::Ok().json(web::Json(get_python_comments(&payload.code)))
}

