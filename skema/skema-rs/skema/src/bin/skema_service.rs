use actix_web::{get, post, web, App, HttpResponse, HttpRequest, HttpServer, Responder, Result};
use serde::{Deserialize, Serialize};
use comment_extraction::languages::python::get_comments_from_string as get_python_comments;
use comment_extraction::languages::python::Comments;

#[derive(Debug, Serialize, Deserialize)]
struct CommentExtractionRequest {
    language: String,
    source_code: String
}

#[get("/")]
async fn get_comments(request: web::Json<CommentExtractionRequest>) -> Result<impl Responder> {
    Ok(web::Json(get_python_comments(&request.source_code)))
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(get_comments)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
