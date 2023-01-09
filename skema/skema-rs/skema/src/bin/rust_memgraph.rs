//! Program for inserting GroMEt models into a database from the command line.

use serde_json;
use skema::{
    database::{execute_query, parse_gromet_queries},
    Gromet,
};
use std::env;
use std::fs;

fn main() {
    // take in gromet location and deserialize
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let data = fs::read_to_string(path).expect("Unable to read file");
    let gromet: Gromet = serde_json::from_str(&data).expect("Unable to parse");

    // parse gromet into vec of queries
    let queries = parse_gromet_queries(gromet);

    /* for query in queries.iter() {
        println!("{}", query);
    } */

    // need to make the whole query list one line, individual executions are treated as different graphs for each execution.
    let mut full_query = queries[0].clone();
    for i in 1..(queries.len()) {
        full_query.push_str("\n");
        let temp_str = &queries[i].clone();
        full_query.push_str(&temp_str);
    }

    execute_query(&full_query, "localhost"); // The properties need to have quotes!!

    // writing output to file, since too long for std out now.
    fs::write("debug.txt", full_query).expect("Unable to write file");
}
