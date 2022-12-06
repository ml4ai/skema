use serde_json;
use skema::gromet_memgraph::{execute_query, parse_gromet_queries};
use skema::Gromet;
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

    for query in queries.clone() {
        println!("{}", query);
    }

    // need to make the whole query list one line, individual executions are treated as different graphs for each execution.
    let mut full_query = queries[0].clone();
    for i in 1..(queries.len()) {
        full_query.push_str("\n");
        let temp_str = &queries[i].clone();
        full_query.push_str(&temp_str);
    }

    execute_query(&full_query); // The properties need to have quotes!!
}
