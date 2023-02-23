use std::env;

pub fn get_comments(file_path: &str) {
    println!("Processing file {:?}",file_path);
    let contents = std::fs::read_to_string(file_path);
    match contents {
        Ok(s) => {
            println!("Processing string: {:?}",s);
        },
        Err(e) => println!("Error: {e:?}")
    }
}

