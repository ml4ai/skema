use actix_web::{test, get, post, web, App, HttpResponse, HttpRequest, HttpServer, Responder, Result};
use serde::{Deserialize, Serialize};
use comment_extraction::languages::python::get_comments_from_string as get_python_comments;
use comment_extraction::languages::python::Comments;

#[derive(Debug, Serialize, Deserialize)]
struct CommentExtractionRequest {
    language: String,
    source_code: String
}

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


#[actix_web::test]
async fn test_get_comments() {
    let app = test::init_service(App::new().route("/get_comments", web::get().to(get_comments))).await;
    let payload = CommentExtractionRequest {
        language: "python".to_string(),
        source_code: std::fs::read_to_string("../comment_extraction/tests/data/CHIME_SIR.py").unwrap()
    };
    let request = test::TestRequest::default().set_json(payload).to_http_request();
    let response = test::call_service(&app, request).await;
}
