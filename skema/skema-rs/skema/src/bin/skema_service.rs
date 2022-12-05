use actix_web::{App, HttpServer, HttpResponse, get};
use skema::services::comment_extraction::{
    get_comments, CommentExtractionRequest, CommentExtractionResponse, Docstring, Language,
    SingleLineComment,
};
use skema::queries::{module_request, push_model_request};
use skema::services::mathml::get_ast_graph;

use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use clap::Parser;

#[derive(Parser, Debug)]
struct Cli {
    /// Host
    #[arg(long, default_value_t = String::from("localhost"))]
    host: String,

    /// Port
    #[arg(short, long, default_value_t = 8080)]
    port: u16
}

/// This endpoint can be used to check the health of the service.
#[utoipa::path(
    responses(
        (status = 200, description = "Ping")
    )
)]
#[get("/ping")]
pub async fn ping() -> HttpResponse {
    HttpResponse::Ok().body("The SKEMA Rust web services are running.")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    #[derive(OpenApi)]
    #[openapi(
        paths(
            skema::services::comment_extraction::get_comments,
            skema::services::mathml::get_ast_graph,
            ping
        ),
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

    let args = Cli::parse();
    HttpServer::new(move || {
        App::new()
            .service(get_comments)
            .service(
                SwaggerUi::new("/api-docs/{_:.*}").url("/api-doc/openapi.json", openapi.clone()),
            )
            .service(ping)
            .service(module_request)
            .service(push_model_request)
            .service(get_comments)
            .service(get_ast_graph)
            .service(
            SwaggerUi::new("/api-docs/{_:.*}").url("/api-doc/openapi.json", openapi.clone()),
            )
            .service(ping)
    })
    .bind((args.host, args.port))?
    .run()
    .await
}
