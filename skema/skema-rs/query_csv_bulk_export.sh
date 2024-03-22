#!/bin/bash

# set up environment variables for local memgraph db
SKEMA_GRAPH_DB_HOST="localhost"
SKEMA_GRAPH_DB_PORT="7687"
SKEMA_GRAPH_DB_PROTO="bolt://"

# need the memgraph id for grabbing exports
docker_id=$(docker container ls --all --quiet --filter "name=skema-graphdb-1")

# make directories to store exports
mkdir output_queries
mkdir output_csv_graphs

# build the rust exe
cargo build -r --bin gromet2graphdb

# iterate over our function networks to make query and csv of the graph for each one
for filename in ../../../dataset_generation/data/function_nets/*.json; do
    ./target/release/gromet2graphdb -- $filename
    mv ./output_queries/debug.txt ./output_queries/$(basename $filename .json)-queries.txt
    # need to graph the csv from inside the docker container
    cd output_csv_graphs
    docker cp $docker_id:/usr/lib/memgraph/query_modules/export.csv export.csv
    # delete file after copying it
    docker exec $docker_id rm /usr/lib/memgraph/query_modules/export.csv
    mv ./export.csv ./$(basename $filename .json)-graph.csv
    # return to top starting level
    cd ..
done