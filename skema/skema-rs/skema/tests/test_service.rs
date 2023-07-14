use actix_web::{http::header::ContentType, test, App};
use skema::services::{
    comment_extraction::{get_comments, CommentExtractionRequest, Language},
    mathml::get_amr,
};
use std::fs;

#[actix_web::test]
async fn test_get_comments() {
    let app = test::init_service(App::new().service(get_comments)).await;
    let payload = CommentExtractionRequest::new(
        Language::Python,
        fs::read_to_string("../comment_extraction/tests/data/CHIME_SIR.py").unwrap(),
    );
    let request = test::TestRequest::post()
        .uri("/extract-comments")
        .set_json(&payload)
        .to_request();
    let response = test::call_service(&app, request).await;
    assert!(response.status().is_success());
}

#[actix_web::test]
async fn test_get_amr() {
    let app = test::init_service(App::new().service(get_amr)).await;
    let payload = fs::read_to_string("tests/data/get_amr_payload.json").unwrap();
    let request = test::TestRequest::put()
        .uri("/mathml/amr")
        .insert_header(ContentType::json())
        .set_payload(payload)
        .to_request();
    let response = test::call_service(&app, request).await;
    assert!(response.status().is_success());
}
