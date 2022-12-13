use super::defined_types::Bool;

fn and(x: Bool, y: Bool) -> Bool{
    Bool(x.0 && y.0)
}

fn or(x: Bool, y: Bool) -> Bool{
    Bool(x.0 || y.0)
}
