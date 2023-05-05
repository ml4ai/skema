use actix_web::{get, web::Data, App, HttpResponse, HttpServer};
use skema::config::Config;
use skema::services::{comment_extraction, gromet};

use clap::Parser;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

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
    db_host: String,
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
            comment_extraction::get_comments,
            skema::services::mathml::get_ast_graph,
            skema::services::mathml::get_math_exp_graph,
            skema::services::mathml::get_acset,
            gromet::get_model_ids,
            gromet::post_model,
            gromet::delete_model,
            gromet::get_named_opos,
            gromet::get_named_opis,
            gromet::get_named_ports,
            gromet::get_subgraph,
            ping
        ),
        components(
            schemas(
                comment_extraction::Language,
                comment_extraction::CommentExtractionRequest,
                comment_extraction::SingleLineComment,
                comment_extraction::Docstring,
                comment_extraction::CommentExtractionResponse,
                mathml::acset::ModelRepPn,
                mathml::acset::Model,
                mathml::acset::State,
                mathml::acset::Transition,
                mathml::acset::Grounding,
                mathml::acset::Initial,
                mathml::acset::Rate,
                mathml::acset::Properties,
                mathml::acset::Parameter,
                mathml::acset::Distribution,
                skema::Attribute,
                skema::FunctionNet,
                skema::FunctionType,
                skema::FnType,
                skema::Gromet,
                skema::GrometBox,
                skema::GrometBoxConditional,
                skema::GrometBoxLoop,
                skema::GrometPort,
                skema::GrometWire,
                skema::Grounding,
                skema::Metadata,
                skema::ModuleCollection,
                skema::TextExtraction,
                skema::Files,
                skema::Provenance,
                skema::ValueL,
                skema::ValueMeta,
            )
        ),
        tags(
            (name = "SKEMA", description = "SKEMA web services."),
        ),
    )]
    struct ApiDoc;

    let openapi = ApiDoc::openapi();

    let args = Cli::parse();

    HttpServer::new(move || {
        App::new()
            .app_data(Data::new(Config {
                db_host: args.db_host.clone(),
            }))
            .configure(gromet::configure())
            .service(comment_extraction::get_comments)
            .service(skema::services::mathml::get_ast_graph)
            .service(skema::services::mathml::get_math_exp_graph)
            .service(skema::services::mathml::get_acset)
            .service(ping)
            .service(SwaggerUi::new("/docs/{_:.*}").url("/api-doc/openapi.json", openapi.clone()))
    })
    .bind((args.host, args.port))?
    .run()
    .await
}
