//! Program for inserting GroMEt models into a database from the command line.

use clap::Parser;
use skema::config::Config;
use skema::{
    database::{parse_gromet_queries, run_queries},
    ModuleCollection,
};
use std::env;
use std::fs;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to GroMEt JSON file to ingest into database
    path: String,

    /// Database host
    #[arg(long, default_value_t = String::from("localhost"))]
    db_host: String,
}

#[tokio::main]
async fn main() {
    // debug outputs
    let debug = true;
    // take in gromet location and deserialize

    let args = Cli::parse();
    let data = fs::read_to_string(&args.path).expect("Unable to read file");
    let gromet: ModuleCollection = serde_json::from_str(&data).expect("Unable to parse");

    // parse gromet into vec of queries
    let queries = parse_gromet_queries(gromet);

    // need to make the whole query list one line, individual executions are treated as different graphs for each execution.
    let mut full_query = queries[0].clone();
    for que in queries.iter().skip(1) {
        full_query.push('\n');
        let temp_str = que;
        full_query.push_str(temp_str);
    }

    if debug {
        fs::write("debug.txt", full_query.clone()).expect("Unable to write file");
    }

    let db_protocol = env::var("SKEMA_GRAPH_DB_PROTO").unwrap_or("bolt+s://".to_string());
    let db_host =
        env::var("SKEMA_GRAPH_DB_HOST").unwrap_or("graphdb-bolt.askem.lum.ai".to_string());
    let db_port = env::var("SKEMA_GRAPH_DB_PORT").unwrap_or("443".to_string());

    let config = Config {
        db_protocol: db_protocol.clone(),
        db_host: db_host.clone(),
        db_port: db_port.parse::<u16>().unwrap(),
    };

    run_queries(queries, config.clone()).await.unwrap(); // The properties need to have quotes!!
                                                         // writing output to file, since too long for std out now.
}
