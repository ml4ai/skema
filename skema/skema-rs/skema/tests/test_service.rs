use skema::services::comment_extraction::{CommentExtractionRequest, get_comments};
use actix_web::{test, App, web};

#[actix_web::test]
async fn test_get_comments() {
    let app = test::init_service(App::new().service(web::resource("/get_comments").route(web::get().to(get_comments)))).await;
    let payload = CommentExtractionRequest::new("python", std::fs::read_to_string("../comment_extraction/tests/data/CHIME_SIR.py").unwrap());
    let request = test::TestRequest::get().uri("/get_comments").set_json(&payload).to_request();
    let response = test::call_service(&app, request).await;
    assert!(response.status().is_success());
}
