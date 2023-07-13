use crate::parsers::generic_mathml::tag_parser;

/// A macro to recognize specific mi elements.
#[macro_export]
macro_rules! mi {
    ($parser:expr) => {{
        tag_parser!("mi", $parser)
    }};
}

/// A macro to recognize specific mo elements.
#[macro_export]
macro_rules! mo {
    ($parser:expr) => {{
        tag_parser!("mo", $parser)
    }};
}

/// A macro to recognize mrows with specific contents
#[macro_export]
macro_rules! mrow {
    ($parser:expr) => {{
        tag_parser!("mrow", $parser)
    }};
}

/// A macro to recognize mover elements with specific contents
#[macro_export]
macro_rules! mover {
    ($parser:expr) => {{
        tag_parser!("mover", $parser)
    }};
}

/// A macro to recognize munder elements with specific contents
#[macro_export]
macro_rules! munder {
    ($parser:expr) => {{
        tag_parser!("munder", $parser)
    }};
}

/// A macro to recognize mfrac elements with specific contents
#[macro_export]
macro_rules! mfrac {
    ($parser:expr) => {{
        tag_parser!("mfrac", $parser)
    }};
}

/// A macro to recognize mfrac elements with specific contents
#[macro_export]
macro_rules! math {
    ($parser:expr) => {{
        tag_parser!("math", $parser)
    }};
}

pub(crate) use mfrac;
pub(crate) use mi;
pub(crate) use mo;
pub(crate) use mover;
pub(crate) use mrow;
pub(crate) use munder;
pub(crate) use math;
