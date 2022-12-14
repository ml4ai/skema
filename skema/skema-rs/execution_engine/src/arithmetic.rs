use super::defined_types::GrometNumber;
use super::defined_types::GrometInt;

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

fn add(x: GrometNumber, y: GrometNumber) -> GrometNumber{ 
    match (x, y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value + y_data.value),
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value.to_f64().unwrap() + y_data.value), // #TODO: How to handle this conversion?
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value + y_data.value.to_f64().unwrap()),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value + y_data.value),
    }
}

fn sub(x: GrometNumber, y: GrometNumber) -> GrometNumber {
    match (x, y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value - y_data.value),
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value.to_f64().unwrap() - y_data.value),
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value - y_data.value.to_f64().unwrap()),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value - y_data.value),
    }
}

fn mult(x: GrometNumber, y: GrometNumber) -> GrometNumber {
    match (x, y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value * y_data.value),
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value.to_f64().unwrap() * y_data.value), 
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value * y_data.value.to_f64().unwrap()),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value * y_data.value),
    }
}

fn div(x: GrometNumber, y: GrometNumber) -> GrometNumber {
    match (x, y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value / y_data.value),
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value.to_f64().unwrap() / y_data.value),
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => GrometNumber::from(x_data.value / y_data.value.to_f64().unwrap()),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => GrometNumber::from(x_data.value / y_data.value),
    }
}

// This is called Integer division in Rust. Unlike Python there is no built in // operator.
// Instead we cast both inputs to an Integer and then perform the operation.
fn floor_div(x: GrometNumber, y: GrometNumber) -> GrometInt {
    let x_integer: BigInt = match x{
        GrometNumber::Int(x_data) => x_data.value,
        GrometNumber::Float(x_data) => BigInt::from_f64(x_data.value).unwrap()
    };
    let y_integer: BigInt = match y{
        GrometNumber::Int(x_data) => x_data.value,
        GrometNumber::Float(x_data) => BigInt::from_f64(x_data.value).unwrap()
    };

    let result: BigInt = x_integer / y_integer;
    GrometInt{value: result}
}

// TODO: How do we handle the case of something that would overflow in Rust, but wouldn't in Python?
// TODO: How to handle float arguments while emulating Python behavior?
fn pow(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value.pow(y.value.to_u32().unwrap());
    GrometInt{value: result}
}

// TODO: How to handle float arguments while emulating Python behavior?
fn r#mod(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value % y.value;
    GrometInt{value: result}
}


#[test]
fn test_add_int() {
    // Test integer addition
    assert_eq!(add(GrometNumber::from(1), GrometNumber::from(2)), GrometNumber::from(3));
}
#[test]
fn test_add_float() {
    // Test floating point addition
    assert_eq!(add(GrometNumber::from(1.2), GrometNumber::from(2.3)), GrometNumber::from(3.5));
}
#[test]
fn test_add_mixed() {
    // Test mixed addition
    assert_eq!(add(GrometNumber::from(1.1), GrometNumber::from(1)), GrometNumber::from(2.1))
}

#[test]
fn test_sub_int() {
    // Test integer addition
    assert_eq!(sub(GrometNumber::from(1), GrometNumber::from(2)), GrometNumber::from(-1));
}
#[test]
fn test_sub_float() { //TODO Develop tests to work around float precision errors
    // Test floating point addition
    assert_eq!(sub(GrometNumber::from(2.3), GrometNumber::from(1.2)), GrometNumber::from(1.1));
}
#[test]
fn test_sub_mixed() {
    // Test mixed addition
    assert_eq!(sub(GrometNumber::from(1.1), GrometNumber::from(1)), GrometNumber::from(0.1))
}