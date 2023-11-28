use std::env;
use neo4rs::*;
// NOTE: this is deprecated
use rsmgclient::{ConnectParams};

#[derive(Debug, Clone)]
pub struct Config {
    // NOTE: do not specify procotol (ex. "bolt://") as part of db_host
    pub db_protocol: String,
    pub db_host: String,
    pub db_port: u16
}

impl Default for Config {
  fn default() -> Self {
      // Default initialization using ENV vars and standard values when unset
      Config {
          db_protocol: env::var("SKEMA_GRAPH_DB_PROTO").unwrap_or("bolt://".to_string()),
          db_host: env::var("SKEMA_GRAPH_DB_HOST").unwrap_or("127.0.0.1".to_string()),
          db_port: env::var("SKEMA_GRAPH_DB_PORT").unwrap_or("7687".to_string()).parse::<u16>().unwrap(),
      }
  }
}

impl Config {

    fn new() -> Self {
      Default::default()
    }

    pub fn create_graphdb_uri(&self) -> String {
      format!("{proto}{host}:{port}", proto=self.db_protocol, host=self.db_host, port=self.db_port)
    }
    pub async fn graphdb_connection(&self) -> Graph {
      let uri = self.create_graphdb_uri();
      println!("skema-rs:memgraph uri:\t{addr}", addr=uri);
      let graph_config = ConfigBuilder::new()
       .uri(uri)
       .user("".to_string())
       .password("".to_string())
       .db("memgraph".to_string())
       .fetch_size(200)
       .max_connections(16)
       .build()
       .unwrap();
      let graph = Graph::connect(graph_config).await.unwrap();
      graph
    }

    // this is deprecated
    pub fn db_connection(&self) -> ConnectParams {
      let cp = ConnectParams {
        port: self.db_port,
        host: Some(self.db_host.clone()),
        // consider adding username & password
        ..Default::default()
      };
      println!("skema-rs::memgraph address:\t{host}:{port}", host=self.db_host, port=self.db_port);
      cp
    }
}
