//! Comment extraction services

use actix_web::{get, post, web, HttpResponse};
use comment_extraction::{comments::Comments, extraction::extract_comments_from_directory};
use comment_extraction::languages::python::get_comments_from_string as get_python_comments;
use serde::{Deserialize, Serialize};
use utoipa;
use utoipa::ToSchema;
use std::{fs, io::Write};
use log::debug;
use tempdir::TempDir;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub enum Language {
    Python,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CommentExtractionRequest {
    pub language: Language,
    pub code: String,
}

/// A single line comment
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SingleLineComment {
    line: usize,
    contents: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Docstring {
    object_name: String,
    contents: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CommentExtractionResponse {
    single_line_comments: Vec<SingleLineComment>,
    docstrings: Vec<Docstring>,
}

impl CommentExtractionResponse {
    fn new(comments: Comments) -> Self {
        let mut single_line_comments = Vec::new();
        for comment in comments.comments.iter() {
            single_line_comments.push(SingleLineComment {
                line: comment.line_number,
                contents: comment.contents.clone(),
            });
        }
        let mut docstrings = Vec::new();
        for (object_name, contents) in comments.docstrings {
            docstrings.push(Docstring {
                object_name,
                contents,
            });
        }
        Self {
            single_line_comments,
            docstrings,
        }
    }
}

impl CommentExtractionRequest {
    pub fn new(language: Language, code: String) -> Self {
        Self { language, code }
    }
}

/// Get comments for a piece of code.
#[utoipa::path(
    request_body = CommentExtractionRequest,
    responses(
        (status = 200, description = "Get comments", body = CommentExtractionResponse)
    )
)]
#[get("/get_comments")]
pub async fn get_comments(payload: web::Json<CommentExtractionRequest>) -> HttpResponse {
    HttpResponse::Ok().json(web::Json(get_python_comments(&payload.code)))
}


/// Get comments for a zipfile containing a directory of code
#[utoipa::path(
    responses(
        (status = 200, description = "Get comments from zipfile")
    )
)]
#[post("/get-comments-from-zipfile")]
pub async fn get_comments_from_zipfile(payload: web::Bytes) -> HttpResponse {
    debug!("Zip file received. Size: {} bytes", payload.len());
    let tmp_dir = TempDir::new("comment_extraction_input").unwrap();
    let file_path = tmp_dir.path().join("directory.zip");
    let mut f = fs::File::create(file_path.clone()).unwrap();
    f.write_all(&payload).unwrap();
    drop(f);

    // Read zipfile
    let file = fs::File::open(file_path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    archive.extract(&tmp_dir).unwrap();
    let comments = extract_comments_from_directory(&tmp_dir, &None);

    tmp_dir.close().unwrap();

    HttpResponse::Ok().json(web::Json(comments))
}
