//! REST API endpoints related to CRUD operations and other queries on GroMEt objects.

use crate::database::{execute_query, parse_gromet_queries};
use crate::Gromet;
use rsmgclient::{ConnectParams, Connection, MgError, Value};
use serde::{Deserialize, Serialize};

use actix_web::{HttpResponse, get, post, web, delete};
use utoipa;

pub fn push_model_to_db(gromet: Gromet) -> Result<i64, MgError> {

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
    let model_ids = module_query()?;
    let last_model_id = model_ids[model_ids.len() - 1];
    Ok(last_model_id)
}

pub fn delete_module(module_id: i64) -> Result<(),MgError> {
    // construct the query that will delete the module with a given unique identifier

    let query = format!("MATCH (n)-[r:Contains|Port_Of|Wire*1..5]->(m) WHERE id(n) = {}\nDETACH DELETE n,m", module_id);
    execute_query(&query);
    Ok(())
}

pub fn module_query() -> Result<Vec<i64>, MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(String::from("localhost")),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // Run Query.
    connection.execute("MATCH (n:Module) RETURN collect(id(n));", None)?;

    // Check that the first value of the first record is a list
    let mut ids = Vec::<i64>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        ids = xs.iter().filter_map(|x| match x {
            Value::Int(x) => Some(x.clone()),
            _ => None
        }).collect();
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
pub async fn get_model_ids() -> HttpResponse {
    let response = module_query().unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// Pushes a gromet JSON to the Memgraph database
#[utoipa::path(
    request_body = Gromet,
    responses(
        (status = 200, description = "Model successfully pushed")
    )
)]
#[post("/models")]
pub async fn post_model(payload: web::Json<Gromet>) -> HttpResponse {
    let model_id = push_model_to_db(payload.into_inner()).unwrap();
    HttpResponse::Ok().json(web::Json(model_id))
}

/// Deletes a model from the database based on its id.
#[utoipa::path(
    responses(
        (status = 200, description = "Model deleted")
    )
)]
#[delete("/models/{id}")]
pub async fn delete_model(path: web::Path<i64>) -> HttpResponse {
    let id = path.into_inner();
    delete_module(id);
    HttpResponse::Ok().body("Model deleted")
}
