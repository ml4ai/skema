//! REST API endpoints related to CRUD operations and other queries on GroMEt objects.

use crate::database::{execute_query, parse_gromet_queries};
use crate::Gromet;
use rsmgclient::{ConnectParams, Connection, MgError, Value};
use serde::{Deserialize, Serialize};
use std::process::Termination;

use actix_web::{HttpResponse, get, post, web};
use utoipa;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModuleId {
    pub module_id: u32,
}

pub fn push_model_to_db(gromet: Gromet) -> Result<(), MgError> {

    // parse gromet into vec of queries
    let queries = parse_gromet_queries(gromet);

    // need to make the whole query list one line, individual executions are treated as different graphs for each execution.
    let mut full_query = queries[0].clone();
    for i in 1..(queries.len()) {
        full_query.push_str("\n");
        let temp_str = &queries[i].clone();
        full_query.push_str(&temp_str);
    }
    execute_query(&full_query);
    Ok(())
}

pub fn delete_module(module_id: u32) -> Result<(),MgError> {
    // construct the query that will delete the module with a given unique identifier

    let query = format!("MATCH (n)-[r:Contains|Port_Of|Wire*1..5]->(m) WHERE id(n) = {}\nDETACH DELETE n,m", module_id);
    execute_query(&query);
    Ok(())
}

pub fn module_query() -> Result<String, MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(String::from("localhost")),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // Run Query.
    let columns = connection.execute("MATCH (n:Module) RETURN n;", None)?;
    let mut result = String::from("Modules:");
    for record in connection.fetchall()? {
        for value in record.values {
            match value {
                Value::Node(node) => result = format!("{} \n filename: {}, id: {}", result.clone(), node.properties.get("filename").unwrap(), node.id),
                Value::Relationship(edge) => print!("edge"),
                value => print!("{}", value),
            }
        }
    }
    connection.commit()?;

    Ok(result)
}
/// This retrieves the module ids and filename
#[utoipa::path(
    responses(
        (status = 200, description = "Modules successfully pinged")
    )
)]
#[get("/module_ping")]
pub async fn module_ping() -> HttpResponse {
    let response = module_query().unwrap();
    HttpResponse::Ok().body(response)
}
/// Pushes a gromet JSON to the Memgraph database
#[utoipa::path(
    request_body = Gromet,
    responses(
        (status = 200, description = "Model successfully pushed")
    )
)]
#[post("/push_model")]
pub async fn push_model(payload: web::Json<Gromet>) -> HttpResponse {
    push_model_to_db(payload.into_inner());
    HttpResponse::Ok().body("Pushed model to Database")
}
/// This deletes a model from the Memgraph Database instance based on it's id
#[utoipa::path(
    request_body = ModuleId,
    responses(
        (status = 200, description = "Model Deleted", body = ModuleId)
    )
)]
#[get("/module_delete")]
pub async fn module_delete(payload: web::Json<ModuleId>) -> HttpResponse {
    delete_module(payload.module_id);
    HttpResponse::Ok().body("Module Deleted")
}
