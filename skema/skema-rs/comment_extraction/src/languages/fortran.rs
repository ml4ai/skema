use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashSet;

/// From FORTRAN Language Reference
/// (https://docs.oracle.com/cd/E19957-01/805-4939/z40007332024/index.html)
///
/// A line with a c, C, '*', d, D, or ! in column one is a comment line, except
/// that if the -xld option is set, then the lines starting with D or d are
/// compiled as debug lines. The d, D, and ! are nonstandard.
///
/// If you put an exclamation mark (!) in any column of the statement field,
/// except within character literals, then everything after the ! on that
/// line is a comment.
///
/// A totally blank line is a comment line as well.
pub fn line_is_comment(line: &str) -> bool {
    lazy_static! {
        static ref FORTRAN_COMMENT_CHAR_SET: HashSet<char> =
            HashSet::from(['c', 'C', 'd', 'D', '*', '!']);
    }

    match &line.chars().next() {
        Some(c) => FORTRAN_COMMENT_CHAR_SET.contains(c),
        None => true,
    }
}

/// Indicates whether a line in the program is the first line of a subprogram
/// definition.
///
/// # Arguments
///
/// * `line` - The line of code to analyze
///Returns:
///    (true, f_name) if line begins a definition for subprogram f_name;
///    (false, None) if line does not begin a subprogram definition.
pub fn line_starts_subpgm(line: &str) -> (bool, Option<String>) {
    lazy_static! {
        static ref RE_SUB_START: Regex = Regex::new(r"\s*subroutine\s+(\w+)\s*\(").unwrap();
        static ref RE_FN_START: Regex =
            Regex::new(r"\s*(\w*\s*){0,2}function\s+(\w+)\s*\(").unwrap();
    }

    if let Some(c) = RE_SUB_START.captures(line) {
        let f_name = &c[1];
        return (true, Some(f_name.to_string()));
    }

    if let Some(c) = RE_FN_START.captures(line) {
        let f_name = &c[2];
        return (true, Some(f_name.to_string()));
    }

    (false, None)
}

pub fn line_ends_subpgm(line: &str) -> bool {
    lazy_static! {
        static ref RE_SUBPGM_END: Regex = Regex::new(r"\s*end\s+").unwrap();
    }
    RE_SUBPGM_END.is_match(line)
}

// TODO: Implement a test for the logic of the function below.
/// From FORTRAN 77 Language Reference
/// (https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vn6l/index.html)   }
///
/// A statement takes one or more lines; the first line is called the initial
/// line; the subsequent lines are called the continuation lines.  You can
/// format a source line in either of two ways: Standard fixed format,
/// or Tab format.
///
/// In Standard Fixed Format, continuation lines are identified by a nonblank,
/// nonzero in column 6.
///
/// Tab-Format source lines are defined as follows: A tab in any of columns
/// 1 through 6, or an ampersand in column 1, establishes the line as a
/// tab-format source line.  If the tab is the first nonblank character, the
/// text following the tab is scanned as if it started in column 7.
/// Continuation lines are identified by an ampersand (&) in column 1, or a
/// nonzero digit after the first tab.
///
/// Returns true iff line is a continuation line, else False.  Currently this
/// is used only for fixed-form input files, i.e., extension is in ('.f', '.for')
pub fn line_is_continuation(line: &str, extension: &str) -> bool {
    if line_is_comment(line) {
        return false;
    }

    // Adarsh: It would be nice if we could make this a global constant, but for some reason it is
    // not obvious how to do this in Rust...
    let fixed_form_ext = HashSet::from(["f", "for", "blk", "inc", "gin"]);

    lazy_static! {
        static ref FIXED_FORM_SET: HashSet<char> =
            HashSet::from(['1', '2', '3', '4', '5', '6', '7', '8', '9']);
    }

    // Adarsh: The code below assumes that we don't need to panic if the line doesn't have
    // characters at the expected positions. I am not 100% sure that is the case without diving
    // into the Fortran language specification, which I am not inclined to do at the moment. In any
    // case, the Python version of this code does not include any checks of this kind. If the
    // expectation was for the script to crash in such cases, the type signature in the original
    // Python version did not reflect it.
    if fixed_form_ext.contains(extension as &str) {
        if line.starts_with('\t') {
            return FIXED_FORM_SET.contains(&line.chars().nth(1).unwrap());
        } else {
            let c_5 = &line.chars().nth(5).unwrap();
            return line.len() > 5 && !(*c_5 == ' ' || *c_5 == '0');
        }
    }

    if line.starts_with('&') {
        return true;
    }

    false
}
