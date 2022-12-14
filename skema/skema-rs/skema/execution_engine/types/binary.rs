use crate::execution_engine::types::defined_types::Number;
use num_bigint::BigInt;
use num::ToPrimitive;
use num::FromPrimitive;

// Todo:
// 1. Enum variant as type: FloorDiv
// 2. Error handling instead of .unwrap

/*fn main() {
    let int_test = Number::Int("100".parse::<BigInt>().unwrap());
    let complex_test = Number::Complex(Complex::new(1.2, 2.3));
    let float_test = Number::Float(0.333);
    println!("{:?}", int_test);
    println!("{:?}", complex_test);
    println!("{:?}", float_test);
}*/

// This is the specific Number type for the Add operator. 
// We will later define a GenAdd operator for the general case.
fn add(x: Number, y: Number) -> Number{ 
    match (x, y) {
        (Number::Int(x_value), Number::Int(y_value)) => Number::Int(x_value + y_value),
        (Number::Int(x_value), Number::Float(y_value)) => Number::Float(x_value.to_f64().unwrap() + y_value), // #TODO: How to handle this conversion?
        (Number::Float(x_value), Number::Int(y_value)) => Number::Float(x_value + y_value.to_f64().unwrap()),
        (Number::Float(x_value), Number::Float(y_value)) => Number::Float(x_value + y_value),
    }
}

fn sub(x: Number, y: Number) -> Number {
    match (x, y) {
        (Number::Int(x_value), Number::Int(y_value)) => Number::Int(x_value - y_value),
        (Number::Int(x_value), Number::Float(y_value)) => Number::Float(x_value.to_f64().unwrap() - y_value),
        (Number::Float(x_value), Number::Int(y_value)) => Number::Float(x_value - y_value.to_f64().unwrap()),
        (Number::Float(x_value), Number::Float(y_value)) => Number::Float(x_value - y_value),
    }
}

fn Mult(x: Number, y) -> Number {
    match (x, y) {
        (Number::Int(x_value), Number::Int(y_value)) => Number::Int(x_value * y_value),
        (Number::Int(x_value), Number::Float(y_value)) => Number::Float(x_value.to_f64().unwrap() * y_value), 
        (Number::Float(x_value), Number::Int(y_value)) => Number::Float(x_value * y_value.to_f64().unwrap()),
        (Number::Float(x_value), Number::Float(y_value)) => Number::Float(x_value * y_value),
    }
}

fn Div(x: Number, y: Number) -> Number {
    match (x, y) {
        (Number::Int(x_value), Number::Int(y_value)) => Number::Int(x_value / y_value),
        (Number::Int(x_value), Number::Float(y_value)) => Number::Float(x_value.to_f64().unwrap() / y_value),
        (Number::Float(x_value), Number::Int(y_value)) => Number::Float(x_value / y_value.to_f64().unwrap()),
        (Number::Float(x_value), Number::Float(y_value)) => Number::Float(x_value / y_value),
    }
}

// This is called Integer division in Rust. Unlike Python there is no built in // operator.
// Instead we cast both inputs to an Integer and then perform the operation.
fn FloorDiv(x: Number, y: Number) -> Number { //TODO: Technically this should return an Number::Int() but thats not possible in rust
    let x_integer: BigInt = match x{
        Number::Int(x_value) => x_value,
        Number::Float(x_value) => BigInt::from_f64(x_value).unwrap()
    }
    let y_integer: BigInt = match y{
        Number::Int(x_value) => x_value,
        Number::Float(x_value) => BigInt::from_f64(x_value).unwrap()
    }
    Number::Int(x_integer / y_integer)
}

// TODO: How do we handle the case of something that would overflow in Rust, but wouldn't in Python?
fn Pow(base: Number, exponent: Number) -> Number {
    return pow(number, exponent);
    // TODO: powf, powi, powc?
}

// TODO: Can you mod a float or complex?
fn Mod(dividend: Number, divisor: Number) -> Number {
    return dividend % divisor;
}


// This is another potential way to have the primitive map represented.
// However, this makes more sense when we are storing more than just the execution code 
/*struct Add {}
impl Add {
    fn exec(augend: Number, addend: Number) -> Number {
        augend + addend
    }
}
*/

#[test]
fn test_add() {
    // Test integer addition
    assert_eq!(Add(1, 2), 3);

    // Test floating point addition
    assert_eq!(Add(1.2, 2.3), 3.5);

    // Test complex addition
    assert_eq!(
        Add(Complex::new(1.2, 2.3), Complex::new(4.5, 6.7)),
        Complex::new(5.7, 9.0)
    );
}
