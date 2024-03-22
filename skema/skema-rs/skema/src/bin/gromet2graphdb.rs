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
    let push_graph = true;
    let clear_graph = true;
    //let json_output = true;
    let csv_output = true;
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
        fs::write("./output_queries/debug.txt", full_query.clone()).expect("Unable to write file");
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

    if push_graph {
        run_queries(queries, config.clone()).await.unwrap(); // The properties need to have quotes!!
        if csv_output {
            let csv_query = ["WITH \"MATCH path = (c)-[r]->(m) RETURN c,r,m\" AS query\n
            CALL export_util.csv_query(query, \"/usr/lib/memgraph/query_modules/export.csv\", True)\n
            YIELD file_path, data\n
            RETURN file_path, data;".to_string()].to_vec();
            run_queries(csv_query, config.clone()).await.unwrap();
        }
        if clear_graph {
            let clear_query = ["MATCH (n) DETACH DELETE n".to_string()].to_vec();
            run_queries(clear_query, config.clone()).await.unwrap();
        }                                            
    }
}
