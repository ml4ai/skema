use actix_web::{App, HttpServer};
use skema::services::comment_extraction::{
    get_comments, CommentExtractionRequest, CommentExtractionResponse, Docstring, Language,
    SingleLineComment,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    #[derive(OpenApi)]
    #[openapi(
        paths(skema::services::comment_extraction::get_comments),
        components(
            schemas(
                Language,
                CommentExtractionRequest,
                SingleLineComment,
                Docstring,
                CommentExtractionResponse,
            )
        ),
        tags(
            (name = "SKEMA", description = "SKEMA web services.")
        ),
    )]
    struct ApiDoc;

    let openapi = ApiDoc::openapi();
    HttpServer::new(move || {
        App::new().service(get_comments).service(
            SwaggerUi::new("/api-docs/{_:.*}").url("/api-doc/openapi.json", openapi.clone()),
        )
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
