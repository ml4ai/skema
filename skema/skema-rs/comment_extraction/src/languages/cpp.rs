use std::env;

fn process_file(file_path: &str) {
    println!("Processing file {:?}",file_path);
    let contents = std::fs::read_to_string(file_path);
    match contents {
        Ok(s) => {
            println!("Processing string: {:?}",s);
        },
        Err(e) => println!("Error: {e:?}")
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut iterator = args.iter();
    if let Some(name) = iterator.next() {
        println!("{:?} starting...", name);
    }
    while let Some(file_path) = iterator.next() {
        process_file(file_path);
    }
    println!("Done.");
}
