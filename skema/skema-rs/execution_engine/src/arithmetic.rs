use super::defined_types::Number;
use super::defined_types::Int;

use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::cast::FromPrimitive;

// Issues in Rust:
// 1. Overflow
// 2. Variatic functions 
// 3. Errors that would be seen at runtime in Python
// 4. Calling functions and libraries that come from Python

// Todo:
// 1. Enum variant as type: FloorDiv
// 2. Error handling instead of .unwrap
// 3. Add test cases
// 4. Figure out mut and pub

fn add(x: Number, y: Number) -> Number{ 
    match (x, y) {
        (Number::Int(x_data), Number::Int(y_data)) => Number::from(x_data.0 + y_data.0),
        (Number::Int(x_data), Number::Float(y_data)) => Number::from(x_data.0.to_f64().unwrap() + y_data.0), // #TODO: How to handle this conversion?
        (Number::Float(x_data), Number::Int(y_data)) => Number::from(x_data.0 + y_data.0.to_f64().unwrap()),
        (Number::Float(x_data), Number::Float(y_data)) => Number::from(x_data.0 + y_data.0),
    }
}

fn sub(x: Number, y: Number) -> Number {
    match (x, y) {
        (Number::Int(x_data), Number::Int(y_data)) => Number::from(x_data.0 - y_data.0),
        (Number::Int(x_data), Number::Float(y_data)) => Number::from(x_data.0.to_f64().unwrap() - y_data.0),
        (Number::Float(x_data), Number::Int(y_data)) => Number::from(x_data.0 - y_data.0.to_f64().unwrap()),
        (Number::Float(x_data), Number::Float(y_data)) => Number::from(x_data.0 - y_data.0),
    }
}

fn mult(x: Number, y: Number) -> Number {
    match (x, y) {
        (Number::Int(x_data), Number::Int(y_data)) => Number::from(x_data.0 * y_data.0),
        (Number::Int(x_data), Number::Float(y_data)) => Number::from(x_data.0.to_f64().unwrap() * y_data.0), 
        (Number::Float(x_data), Number::Int(y_data)) => Number::from(x_data.0 * y_data.0.to_f64().unwrap()),
        (Number::Float(x_data), Number::Float(y_data)) => Number::from(x_data.0 * y_data.0),
    }
}

fn div(x: Number, y: Number) -> Number {
    match (x, y) {
        (Number::Int(x_data), Number::Int(y_data)) => Number::from(x_data.0 / y_data.0),
        (Number::Int(x_data), Number::Float(y_data)) => Number::from(x_data.0.to_f64().unwrap() / y_data.0),
        (Number::Float(x_data), Number::Int(y_data)) => Number::from(x_data.0 / y_data.0.to_f64().unwrap()),
        (Number::Float(x_data), Number::Float(y_data)) => Number::from(x_data.0 / y_data.0),
    }
}

// This is called Integer division in Rust. Unlike Python there is no built in // operator.
// Instead we cast both inputs to an Integer and then perform the operation.
fn floor_div(x: Number, y: Number) -> Int {
    let x_integer: BigInt = match x{
        Number::Int(x_data) => x_data.0,
        Number::Float(x_data) => BigInt::from_f64(x_data.0).unwrap()
    };
    let y_integer: BigInt = match y{
        Number::Int(x_data) => x_data.0,
        Number::Float(x_data) => BigInt::from_f64(x_data.0).unwrap()
    };

    Int(x_integer / y_integer)
}

// TODO: How do we handle the case of something that would overflow in Rust, but wouldn't in Python?
// TODO: How to handle float arguments while emulating Python behavior?
fn pow(x: Int, y: Int) -> Int {
    Int(x.0.pow(y.0.to_u32().unwrap()))
}

// TODO: How to handle float arguments while emulating Python behavior?
fn r#mod(x: Int, y: Int) -> Int {
    Int(x.0 % y.0)
}


#[test]
fn test_add_int() {
    // Test integer addition
    assert_eq!(add(Number::from(1), Number::from(2)), Number::from(3));
}
#[test]
fn test_add_float() {
    // Test floating point addition
    assert_eq!(add(Number::from(1.2), Number::from(2.3)), Number::from(3.5));
}
#[test]
fn test_add_mixed() {
    // Test mixed addition
    assert_eq!(add(Number::from(1.1), Number::from(1)), Number::from(2.1))
}

#[test]
fn test_sub_int() {
    // Test integer addition
    assert_eq!(sub(Number::from(1), Number::from(2)), Number::from(-1));
}
#[test]
fn test_sub_float() { //TODO Develop tests to work around float precision errors
    // Test floating point addition
    assert_eq!(sub(Number::from(2.3), Number::from(1.2)), Number::from(1.1));
}
#[test]
fn test_sub_mixed() {
    // Test mixed addition
    assert_eq!(sub(Number::from(1.1), Number::from(1)), Number::from(0.1))
}