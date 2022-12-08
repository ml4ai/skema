use clap::Parser;
use crypto_hash::{hex_digest, Algorithm};
use rand::distributions::{Alphanumeric, DistString};
use rsmgclient::{ConnectParams, Connection, MgError, Value};

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    user_id: String,

    /// Database host
    #[arg(long, default_value_t = String::from("localhost"))]
    host: String,

    /// Database port
    #[arg(short, long, default_value_t = 7687)]
    port: u16,
}

/// Create user if they do not already exist
fn create_user(host: String, port: u16, user_id: &str) -> Result<(), MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(host),
        port,
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;
    let api_key = Alphanumeric.sample_string(&mut rand::rngs::OsRng, 30);
    let hash = hex_digest(Algorithm::SHA256, api_key.as_bytes());

    // Create API key
    connection.execute_without_results(&format!(
        "MERGE (p1:User {{id: \'{user_id}\', hashed_api_key: \'{hash}\'}});"
    ))?;

    println!("The API key for user {user_id} is {api_key}.");
    Ok(())
}

fn main() {
    // Get the command line arguments
    let args = Cli::parse();
    create_user(args.host, args.port, &args.user_id).unwrap();
}
