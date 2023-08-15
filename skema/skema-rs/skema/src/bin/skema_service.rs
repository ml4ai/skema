use actix_web::{get, http::header::ContentType, web::Data, App, HttpResponse, HttpServer};
use clap::Parser;
use skema::config::Config;
use skema::services::{comment_extraction, gromet};
use std::env;
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

/// This endpoint can be used to check the version of the service.
#[utoipa::path(
    responses(
        (status = 200, description = "Version")
    )
)]
#[get("/version")]
pub async fn version() -> HttpResponse {
    let end_version = env::var("APP_VERSION").unwrap_or("?????".to_string());
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .body(end_version)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    pretty_env_logger::init();

    #[derive(OpenApi)]
    #[openapi(
        info(
            title = "SKEMA RUST SERVICES",
            version = "version",
        ),
        paths(
            comment_extraction::get_comments,
            comment_extraction::get_comments_from_zipfile,
            skema::services::mathml::get_ast_graph,
            skema::services::mathml::get_math_exp_graph,
            skema::services::mathml::get_acset,
            skema::services::mathml::get_content_mathml,
            skema::services::mathml::get_regnet,
            skema::services::mathml::get_amr,
            gromet::get_model_ids,
            gromet::post_model,
            gromet::delete_model,
            gromet::get_named_opos,
            gromet::get_named_opis,
            gromet::get_named_ports,
            gromet::get_subgraph,
            gromet::get_model_PN,
            gromet::get_model_RN,
            gromet::model2PN,
            gromet::model2RN,
            ping,
            version
        ),
        components(
            schemas(
                comment_extraction::Language,
                comment_extraction::CommentExtractionRequest,
                comment_extraction::SingleLineComment,
                comment_extraction::Docstring,
                comment_extraction::CommentExtractionResponse,
                mathml::acset::AMRmathml,
                mathml::acset::RegNet,
                mathml::acset::ModelRegNet,
                mathml::acset::ModelPetriNet,
                mathml::acset::PetriNet,
                mathml::acset::State,
                mathml::acset::Transition,
                mathml::acset::Grounding,
                mathml::acset::Initial,
                mathml::acset::Rate,
                mathml::acset::Properties,
                mathml::acset::Parameter,
                mathml::acset::Distribution,
                mathml::acset::RegState,
                mathml::acset::RegTransition,
                mathml::acset::Units,
                mathml::acset::Metadata,
                mathml::acset::Semantics,
                mathml::acset::Ode,
                mathml::acset::Observable,
                mathml::acset::Time,
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

    let version_hash = env::var("APP_VERSION").unwrap_or("?????".to_string());

    let mut openapi = ApiDoc::openapi();
    openapi.info.version = version_hash.to_string();
    let args = Cli::parse();

    HttpServer::new(move || {
        App::new()
            .app_data(Data::new(Config {
                db_host: args.db_host.clone(),
            }))
            .configure(gromet::configure())
            .service(comment_extraction::get_comments)
            .service(comment_extraction::get_comments_from_zipfile)
            .service(skema::services::mathml::get_ast_graph)
            .service(skema::services::mathml::get_math_exp_graph)
            .service(skema::services::mathml::get_content_mathml)
            .service(skema::services::mathml::get_acset)
            .service(skema::services::mathml::get_regnet)
            .service(skema::services::mathml::get_amr)
            .service(gromet::get_model_PN)
            .service(gromet::get_model_RN)
            .service(gromet::model2PN)
            .service(gromet::model2RN)
            .service(ping)
            .service(version)
            .service(SwaggerUi::new("/docs/{_:.*}").url("/api-doc/openapi.json", openapi.clone()))
    })
    .bind((args.host, args.port))?
    .run()
    .await
}
