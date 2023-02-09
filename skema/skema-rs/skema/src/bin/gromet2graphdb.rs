//! Program for inserting GroMEt models into a database from the command line.

use clap::Parser;
use serde_json;
use skema::{
    database::{execute_query, parse_gromet_queries},
    ModuleCollection,
};
use std::fs;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to GroMEt JSON file to ingest into database
    path: String,

    /// Database host
    #[arg(long, default_value_t = String::from("localhost"))]
    db_host: String,
}

fn main() {
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
    for i in 1..(queries.len()) {
        full_query.push_str("\n");
        let temp_str = &queries[i].clone();
        full_query.push_str(&temp_str);
    }

    if debug {
        fs::write("debug.txt", full_query.clone()).expect("Unable to write file");
    }

    execute_query(&full_query, &args.db_host).unwrap(); // The properties need to have quotes!!
                                                        // writing output to file, since too long for std out now.
}
