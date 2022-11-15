#[derive(Debug, PartialEq)]
pub struct Field {
    name: String,
    r#type: String,
    variatic: bool,
}

impl Field {
    /// Constructor that takes only the name and type, and sets variatic to false.
    pub fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
            variatic: false,
        }
    }
}
