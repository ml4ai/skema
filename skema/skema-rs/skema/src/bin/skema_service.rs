use actix_web::{web, App, HttpServer};
use skema::services::comment_extraction::get_comments;


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/get_comments", web::get().to(get_comments))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
