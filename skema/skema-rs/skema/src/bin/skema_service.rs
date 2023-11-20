use actix_web::{get, http::header::ContentType, web::Data, App, HttpResponse, HttpServer};
use clap::Parser;
use skema::config::Config;
use skema::services::gromet;
use std::env;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

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
            skema::services::mathml::get_ast_graph,
            skema::services::mathml::get_math_exp_graph,
            skema::services::mathml::get_latex,
            skema::services::mathml::get_acset,
            skema::services::mathml::get_content_mathml,
            skema::services::mathml::get_regnet,
            skema::services::mathml::get_amr,
            skema::services::mathml::get_decapodes,
            gromet::get_model_ids,
            gromet::post_model,
            gromet::delete_model,
            gromet::get_named_opos,
            gromet::get_named_opis,
            gromet::get_named_ports,
            gromet::get_subgraph,
            gromet::get_model_RN,
            gromet::model2PN,
            gromet::model2RN,
            ping,
            version
        ),
        components(
            schemas(
                mathml::parsers::decapodes_serialization::DecapodesCollection,
                mathml::parsers::decapodes_serialization::WiringDiagram,
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
    let host_env = env::var("SKEMA_RS_HOST").unwrap_or("0.0.0.0".to_string());
    let port_env = env::var("SKEMA_RS_PORT").unwrap_or("8080".to_string());
    let db_host = env::var("SKEMA_GRAPH_DB_HOST").unwrap_or("127.0.0.1".to_string());
    let db_port = env::var("SKEMA_GRAPH_DB_PORT").unwrap_or("7687".to_string());


    let mut openapi = ApiDoc::openapi();
    openapi.info.version = version_hash.to_string();

    HttpServer::new(move || {
        App::new()
            .app_data(Data::new(Config {
                db_host: db_host.clone(),
                db_port: db_port.parse::<u16>().unwrap()
            }))
            .configure(gromet::configure())
            .service(skema::services::mathml::get_ast_graph)
            .service(skema::services::mathml::get_math_exp_graph)
            .service(skema::services::mathml::get_latex)
            .service(skema::services::mathml::get_content_mathml)
            .service(skema::services::mathml::get_acset)
            .service(skema::services::mathml::get_regnet)
            .service(skema::services::mathml::get_amr)
            .service(skema::services::mathml::get_decapodes)
            .service(gromet::get_model_RN)
            .service(gromet::model2PN)
            .service(gromet::model2RN)
            .service(ping)
            .service(version)
            .service(SwaggerUi::new("/docs/{_:.*}").url("/api-doc/openapi.json", openapi.clone()))
    })
    .bind((host_env, port_env.parse::<u16>().unwrap()))?
    .run()
    .await
}
