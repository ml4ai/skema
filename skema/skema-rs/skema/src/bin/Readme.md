# GroMEt -> Memgraph Database Parser

This script takes in a GroMEt json and uploads it to the [Memgraph](https://memgraph.com/) database via Cypher queries. This requires having a local instance of Memgraph running on your machine for the graphs to get sent to. 


# Setting up Memgraph
For the most part simply running the following command should work to set up a Docker container for Memgraph: 

`docker run -it -p 7687:7687 -p 7444:7444 -p 3000:3000 -v mg_lib:/var/lib/memgraph memgraph/memgraph-platform`

(NOTE: Requires you to have [Docker](https://www.docker.com/) installed on your machine.)

## Using Memgraph Lab
If you wise to visualize or run live queries on your local database you can use Memgraph Lab.

Once Memgraph is up and running you can open your web browser and go to [`localhost:3000`](http://localhost:3000/). When the Memgraph Lab loads, click **Connect now**.

## Troubleshooting
All this information for Memgraph and more can be found here: https://memgraph.com/docs/memgraph/tutorials/first-steps-with-memgraph 
# Using the parser
The parser can be run using the standard rust lingo. Namely, `cargo run --bin rust_memgraph ../../../data/gromet/examples/exp2/FN_0.1.4/exp2--Gromet-FN-auto.json` where you have to specify the directory to the GroMEt json you want to parse into the database.  


