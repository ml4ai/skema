use super::defined_types::Number;
use super::defined_types::Int;

// TODO: 
// 1. Add Unit Tests
// 2. Research support for float bitwise operators
// 3. DONE - If only for int, use Int variant as type instead of Number

fn bit_and(x: Int, y: Int) -> Int {
    Int(x.0 & y.0)
}

fn bit_or(x: Int, y: Int) -> Int {
    Int(x.0 | y.0)
}

fn bit_xor(x: Int, y: Int) -> Int {
    Int(x.0 ^ y.0)
}

/*fn lshift(x: Int, y: Int) -> Int {
    Int(x.0 << y.0)
}

fn rshift(x: Int, y: Int) -> Int {
    Int(x.0 >> y.0)
}*/