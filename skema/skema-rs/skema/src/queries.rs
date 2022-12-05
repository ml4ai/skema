use crate::gromet_memgraph::{execute_query, parse_gromet_queries};
use crate::Gromet;
use rsmgclient::{ConnectParams, Connection, MgError, Value};

use std::process::Termination;

use actix_web::{HttpResponse, get, post, web};
use utoipa;

pub fn push_model(gromet: Gromet) -> Result<(), MgError> {

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
                Value::Node(node) => result = format!("{} \n {}", result.clone(), node.properties.get("filename").unwrap()),
                Value::Relationship(edge) => print!("edge"),
                value => print!("{}", value),
            }
        }
    }
    connection.commit()?;

    Ok(result)
}

#[utoipa::path(
    responses(
        (status = 200, description = "Module Query")
    )
)]
#[get("/module_request")]
pub async fn module_request() -> HttpResponse {
    let response = module_query().unwrap();
    HttpResponse::Ok().body(response)
}

#[utoipa::path(
    request_body = Gromet,
    responses(
        (status = 200, description = "Pushes Gromet model to DB", body = Gromet)
    )
)]
#[post("/push_model_request")]
pub async fn push_model_request(payload: web::Json<Gromet>) -> HttpResponse {
    push_model(payload.into_inner());
    HttpResponse::Ok().body("Pushed model to Database")
}
