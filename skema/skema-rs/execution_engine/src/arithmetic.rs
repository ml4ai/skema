use super::defined_types::{GrometInt, GrometNumber, Int};

use std::ops::{Add, Sub};

use float_eq::assert_float_eq;
use num_bigint::BigInt;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;

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

impl Add for Int {
    type Output = Self;
    fn add(self, augend: Self) -> Self {
        Int(self.0 + augend.0)
    }
}

impl Sub for Int {
    type Output = Self;
    fn sub(self, subend: Self) -> Self {
        Int(self.0 - subend.0)
    }
}

impl Add<f64> for Int {
    type Output = f64;
    fn add(self, augend: f64) -> f64 {
        self.0
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert {} to f64!", self.0))
            + augend
    }
}

impl Add<Int> for f64 {
    type Output = f64;
    fn add(self, augend: Int) -> f64 {
        self + augend
            .0
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert {} to f64!", augend.0))
    }
}

impl Sub<f64> for Int {
    type Output = f64;
    fn sub(self, subend: f64) -> f64 {
        self.0
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert {} to f64!", self.0))
            - subend
    }
}

impl Sub<Int> for f64 {
    type Output = f64;
    fn sub(self, subend: Int) -> f64 {
        self - subend
            .0
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert {} to f64!", subend.0))
    }
}

fn add<T, U, V>(x: T, y: U) -> V
where
    T: Add<U, Output = V>,
{
    x + y
}

fn sub<T, U, V>(x: T, y: U) -> V
where
    T: Sub<U, Output = V>,
{
    x - y
}

fn mult(x: GrometNumber, y: GrometNumber) -> GrometNumber {
    match (x, y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => {
            GrometNumber::from(x_data.value * y_data.value)
        }
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => {
            GrometNumber::from(x_data.value.to_f64().unwrap() * y_data.value)
        }
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => {
            GrometNumber::from(x_data.value * y_data.value.to_f64().unwrap())
        }
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => {
            GrometNumber::from(x_data.value * y_data.value)
        }
    }
}

fn div(x: GrometNumber, y: GrometNumber) -> GrometNumber {
    match (x, y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => {
            GrometNumber::from(x_data.value / y_data.value)
        }
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => {
            GrometNumber::from(x_data.value.to_f64().unwrap() / y_data.value)
        }
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => {
            GrometNumber::from(x_data.value / y_data.value.to_f64().unwrap())
        }
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => {
            GrometNumber::from(x_data.value / y_data.value)
        }
    }
}

// This is called Integer division in Rust. Unlike Python there is no built in // operator.
// Instead we cast both inputs to an Integer and then perform the operation.
fn floor_div(x: GrometNumber, y: GrometNumber) -> GrometInt {
    let x_integer: BigInt = match x {
        GrometNumber::Int(x_data) => x_data.value,
        GrometNumber::Float(x_data) => BigInt::from_f64(x_data.value).unwrap(),
    };
    let y_integer: BigInt = match y {
        GrometNumber::Int(x_data) => x_data.value,
        GrometNumber::Float(x_data) => BigInt::from_f64(x_data.value).unwrap(),
    };

    let result: BigInt = x_integer / y_integer;
    GrometInt { value: result }
}

// TODO: How do we handle the case of something that would overflow in Rust, but wouldn't in Python?
// TODO: How to handle float arguments while emulating Python behavior?
fn pow(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value.pow(y.value.to_u32().unwrap());
    GrometInt { value: result }
}

// TODO: How to handle float arguments while emulating Python behavior?
fn r#mod(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value % y.value;
    GrometInt { value: result }
}

#[test]
fn test_add_int() {
    // Test integer addition
    assert_eq!(add(Int::from(1), Int::from(2)), Int::from(3));
}

#[test]
fn test_add_float() {
    // Test floating point addition
    assert_eq!(add(1.2, 2.3), 3.5);
}

#[test]
fn test_add_mixed() {
    // Test mixed addition
    assert_eq!(add(1.1, Int::from(1)), 2.1);
}

#[test]
fn test_sub_int() {
    // Test integer addition
    assert_eq!(sub(Int::from(1), Int::from(2)), Int::from(-1));
}

#[test]
fn test_sub_float() {
    // Test floating point subtraction
    assert_float_eq!(sub(2.3, 1.2), 1.1, abs <= 0.00001);
}
#[test]
fn test_sub_mixed() {
    // Test mixed subtraction
    assert_float_eq!(sub(1.1, Int::from(1)), 0.1, abs <= 0.00001);
}
