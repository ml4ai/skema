//! REST API endpoints related to CRUD operations and other queries on GroMEt objects.
use crate::config::Config;
use crate::database::{execute_query, parse_gromet_queries};
use crate::model_extraction::module_id2mathml_MET_ast;
use crate::model_extraction::module_id2mathml_ast;
use crate::ModuleCollection;
use actix_web::web::ServiceConfig;
use actix_web::{delete, get, post, put, web, HttpResponse};
use mathml::acset::{PetriNet, RegNet};
use mathml::mml2pn::ACSet;
use rsmgclient::{ConnectParams, Connection, MgError, Value};
use std::collections::HashMap;
use utoipa;

pub fn configure() -> impl FnOnce(&mut ServiceConfig) {
    |config: &mut ServiceConfig| {
        config
            .service(get_model_ids)
            .service(post_model)
            .service(delete_model)
            .service(get_named_opos)
            .service(get_named_opis)
            .service(get_named_ports)
            .service(get_subgraph);
    }
}
#[allow(non_snake_case)]
pub fn model_to_RN(gromet: ModuleCollection, config: Config) -> Result<RegNet, MgError> {
    let module_id = push_model_to_db(gromet, config.clone()); // pushes model to db and gets id
    let ref_module_id1 = module_id.as_ref();
    let ref_module_id2 = module_id.as_ref();
    let mathml_ast = module_id2mathml_ast(*ref_module_id1.unwrap(), config.clone()); // turns model into mathml ast equations
    let _del_response = delete_module(*ref_module_id2.unwrap(), config.clone()); // deletes model from db
    Ok(RegNet::from(mathml_ast))
}

// this is updated to mathexpressiontrees
#[allow(non_snake_case)]
pub fn model_to_PN(gromet: ModuleCollection, config: Config) -> Result<PetriNet, MgError> {
    let module_id = push_model_to_db(gromet, config.clone()); // pushes model to db and gets id
    let ref_module_id1 = module_id.as_ref();
    let ref_module_id2 = module_id.as_ref();
    let mathml_ast = module_id2mathml_MET_ast(*ref_module_id1.unwrap(), config.clone()); // turns model into mathml ast equations
    let _del_response = delete_module(*ref_module_id2.unwrap(), config.clone()); // deletes model from db
    Ok(PetriNet::from(mathml_ast))
}

pub fn push_model_to_db(gromet: ModuleCollection, config: Config) -> Result<i64, MgError> {
    // parse gromet into vec of queries
    let queries = parse_gromet_queries(gromet);

    // need to make the whole query list one line, individual executions are treated as different graphs for each execution.
    let mut full_query = queries[0].clone();
    for i in 1..(queries.len()) {
        full_query.push('\n');
        let temp_str = &queries[i].clone();
        full_query.push_str(temp_str);
    }
    execute_query(&full_query, config.clone())?;
    let model_ids = module_query(config.clone())?;
    let last_model_id = model_ids[model_ids.len() - 1];
    Ok(last_model_id)
}

pub fn delete_module(module_id: i64, config: Config) -> Result<(), MgError> {
    // construct the query that will delete the module with a given unique identifier

    let query = format!(
        "MATCH (n)-[r:Contains|Port_Of|Wire*1..5]->(m) WHERE id(n) = {}\nDETACH DELETE n,m",
        module_id
    );
    execute_query(&query, config.clone())?;
    Ok(())
}

pub fn named_opi_query(module_id: i64, config: Config) -> Result<Vec<String>, MgError> {
    // construct the query that will delete the module with a given unique identifier

    // Connect to Memgraph.
    let connect_params = config.db_connection();
    let mut connection = Connection::connect(&connect_params)?;

    // create query
    let query = format!(
        "MATCH (n)-[r:Contains|Port_Of|Wire*1..5]->(m) WHERE id(n) = {}
        \nwith DISTINCT m\nmatch (m:Opi) where not m.name = 'un-named'\nreturn collect(m.name)",
        module_id
    );

    // Run Query.
    connection.execute(&query, None)?;

    // Check that the first value of the first record is a list
    let mut port_names = Vec::<String>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        port_names = xs
            .iter()
            .filter_map(|x| match x {
                Value::String(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    Ok(port_names)
}

pub fn named_opo_query(module_id: i64, config: Config) -> Result<Vec<String>, MgError> {
    // construct the query that will delete the module with a given unique identifier

    // Connect to Memgraph.
    let connect_params = config.db_connection();
    let mut connection = Connection::connect(&connect_params)?;

    // create query
    let query = format!(
        "MATCH (n)-[r:Contains|Port_Of|Wire*1..5]->(m) WHERE id(n) = {}
        \nwith DISTINCT m\nmatch (m:Opo) where not m.name = 'un-named'\nreturn collect(m.name)",
        module_id
    );

    // Run Query.
    connection.execute(&query, None)?;

    // Check that the first value of the first record is a list
    let mut port_names = Vec::<String>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        port_names = xs
            .iter()
            .filter_map(|x| match x {
                Value::String(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    Ok(port_names)
}

pub fn named_port_query(module_id: i64, config: Config) -> Result<HashMap<&'static str, Vec<String>>, MgError> {
    let mut result = HashMap::<&str, Vec<String>>::new();
    let opis = named_opi_query(module_id, config.clone());
    let opos = named_opo_query(module_id, config.clone());
    result.insert("opis", opis.unwrap());
    result.insert("opos", opos.unwrap());
    Ok(result)
}

pub fn get_subgraph_query(module_id: i64, config: Config) -> Result<Vec<String>, MgError> {
    // Connect to Memgraph.
    let connect_params = config.db_connection();
    let mut connection = Connection::connect(&connect_params)?;

    // create query1
    let query1 = format!(
        "MATCH p = (n)-[r*]->(m) WHERE id(n) = {}
    \nWITH reduce(output = [], n IN nodes(p) | output + n ) AS nodes1
    \nUNWIND nodes1 AS nodes2
    \nWITH DISTINCT nodes2
    \nRETURN collect(nodes2);",
        module_id
    );

    // Run Query1.
    connection.execute(&query1, None)?;

    // Check that the first value of the first record is a list
    let mut node_list = Vec::<String>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        node_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::String(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    Ok(node_list)
}

pub fn module_query(config: Config) -> Result<Vec<i64>, MgError> {
    // Connect to Memgraph.
    let connect_params = config.db_connection();
    let mut connection = Connection::connect(&connect_params)?;

    // Run Query.
    connection.execute("MATCH (n:Module) RETURN collect(id(n));", None)?;

    // Check that the first value of the first record is a list
    let mut ids = Vec::<i64>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        ids = xs
            .iter()
            .filter_map(|x| match x {
                Value::Int(x) => Some(*x),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    Ok(ids)
}

/// This retrieves the model IDs.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved model IDs")
    )
)]
#[get("/models")]
pub async fn get_model_ids(config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let response = module_query(config1).unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// Pushes a gromet JSON to the Memgraph database
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (status = 200, description = "Model successfully pushed")
    )
)]
#[post("/models")]
pub async fn post_model(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let model_id = push_model_to_db(payload.into_inner(), config1).unwrap();
    HttpResponse::Ok().json(web::Json(model_id))
}

/// Deletes a model from the database based on its id.
#[utoipa::path(
    responses(
        (status = 200, description = "Model deleted")
    )
)]
#[delete("/models/{id}")]
pub async fn delete_model(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let id = path.into_inner();
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    delete_module(id, config1).unwrap();
    HttpResponse::Ok().body("Model deleted")
}

/// This retrieves named Opos based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved named outports")
    )
)]
#[get("/models/{id}/named_opos")]
pub async fn get_named_opos(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let response = named_opo_query(path.into_inner(), config1).unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves named ports based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved named ports")
    )
)]
#[get("/models/{id}/named_ports")]
pub async fn get_named_ports(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let response = named_port_query(path.into_inner(), config1).unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves named Opis based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved named input ports")
    )
)]
#[get("/models/{id}/named_opis")]
pub async fn get_named_opis(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let response = named_opi_query(path.into_inner(), config1).unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves a subgraph based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved subgraph")
    )
)]
#[get("/models/{id}/subgraph")]
pub async fn get_subgraph(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let response = get_subgraph_query(path.into_inner(), config1).unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves a RegNet AMR based on model id.
#[allow(non_snake_case)]
#[utoipa::path(
    responses(
        (
            status = 200, description = "Successfully retrieved RN AMR",
            body = RegNet
        )
    )
)]
#[get("/models/{id}/RN")]
pub async fn get_model_RN(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    let mathml_ast = module_id2mathml_ast(path.into_inner(), config1);
    HttpResponse::Ok().json(web::Json(RegNet::from(mathml_ast)))
}

/// This returns a PetriNet AMR from a gromet.
#[allow(non_snake_case)]
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (
            status = 200, description = "Successfully retrieved PN AMR",
            body = ModelPetriNet
        )
    )
)]
#[put("/models/PN")]
pub async fn model2PN(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    HttpResponse::Ok().json(web::Json(
        model_to_PN(payload.into_inner(), config1).unwrap(),
    ))
}

/// This returns a RegNet AMR from a gromet.
#[allow(non_snake_case)]
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (
            status = 200, description = "Successfully retrieved RN AMR",
            body = ModelRegNet
        )
    )
)]
#[put("/models/RN")]
pub async fn model2RN(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port.clone(),
    };
    HttpResponse::Ok().json(web::Json(
        model_to_RN(payload.into_inner(), config1).unwrap(),
    ))
}
