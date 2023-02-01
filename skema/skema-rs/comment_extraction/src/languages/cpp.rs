use nom::IResult;
use nom_locate::LocatedSpan;
use std::env;

type Span<'a> = LocatedSpan<&'a str>;

struct Comment {
    line: i32,
    text: String,
}

fn test_for_comment(line: i32, text: &str) -> Option<Comment> {
    if line < 6 {
        Some(Comment {
            line: line,
            text: String::from(text),
        })
    } else {
        None
    }
}

fn parse_lines(s: &str) -> Vec<Comment> {
    println!("Processing comments:");
    let v: Vec<&str> = s.split("\n").collect();
    let mut iterator = v.iter();
    let mut line_number = 0;
    let mut output: Vec<Comment> = Vec::new();
    while let Some(line) = iterator.next() {
        line_number += 1;
        output.push(Comment {
            line: line_number,
            text: String::from(line),
        })
    }
    output
}

fn parse_comments(v: &Vec<Comment>) -> Vec<Comment> {
    Vec::new()
}

fn process_file(file_path: &str) {
    println!("Processing file {:?}", file_path);
    let contents = std::fs::read_to_string(file_path);
    match contents {
        Ok(s) => {
            println!("Processing string: {:?}", s);
            let lines = parse_lines(&s);
            parse_comments(&lines);
        }
        Err(e) => println!("Error: {e:?}"),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bar = &args[1..];
    let mut iterator = bar.iter();
    while let Some(file_path) = iterator.next() {
        process_file(file_path);
    }
    println!("Done.");
}
