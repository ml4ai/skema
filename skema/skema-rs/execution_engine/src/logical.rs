use super::defined_types::GrometBool;

fn and(x: GrometBool, y: GrometBool) -> GrometBool{
    result: bool = x.value && y.value;
    GrometBool(value: result)
}

fn or(x: GrometBool, y: GrometBool) -> GrometBool{
    result: bool = x.value || y.value;
    GrometBool(value: result)
}
