use actix_web::{App, HttpServer, HttpResponse, get, web::Data};
use skema::services::comment_extraction::{
    get_comments, CommentExtractionRequest, CommentExtractionResponse, Docstring, Language,
    SingleLineComment,
};
use skema::services::{
    gromet::{get_model_ids, post_model, delete_model, get_named_opos, get_named_opis, get_subgraph},
    mathml::{get_ast_graph, get_math_exp_graph}
};
use skema::config::Config;

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
    port: u16,

    /// Database host
    #[arg(long, default_value_t = String::from("localhost"))]
    db_host: String
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
            skema::services::mathml::get_math_exp_graph,
            skema::services::gromet::get_model_ids,
            skema::services::gromet::post_model,
            skema::services::gromet::delete_model,
            skema::services::gromet::get_named_opos,
            skema::services::gromet::get_named_opis,
            skema::services::gromet::get_subgraph,
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
            .app_data(Data::new(Config {
                db_host: args.db_host.clone()
            }))
            .service(get_comments)
            .service(ping)
            .service(get_model_ids)
            .service(post_model)
            .service(delete_model)
            .service(get_named_opos)
            .service(get_named_opis)
            .service(get_comments)
            .service(get_ast_graph)
            .service(get_math_exp_graph)
            .service(get_subgraph)
            .service(
            SwaggerUi::new("/docs/{_:.*}")
                .url("/api-doc/openapi.json", openapi.clone()),
            )
    })
    .bind((args.host, args.port))?
    .run()
    .await
}
