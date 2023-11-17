use std::env;
use rsmgclient::{ConnectParams};

#[derive(Debug, Clone)]
pub struct Config {
    // NOTE: db_host is protocol + host
    pub db_host: String,
    pub db_port: u16
}

impl Default for Config {
  fn default() -> Self {
      // Default initialization using ENV vars and standard values when unset
      Config {
          db_host: env::var("SKEMA_GRAPH_DB_HOST").unwrap_or("bolt://127.0.0.1".to_string()),
          db_port: env::var("SKEMA_GRAPH_DB_PORT").unwrap_or("7687".to_string()).parse::<u16>().unwrap(),
      }
  }
}

impl Config {

    fn new() -> Self {
      Default::default()
    }
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
