use super::defined_types::GrometNumber;
use super::defined_types::GrometInt;
use super::defined_types::GrometBool;
use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::cast::FromPrimitive;

pub fn gt(x: GrometNumber, y: GrometNumber) -> GrometBool { // TODO: How to make these comparisons 
    let result: bool = match(x,y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => x_data.value > y_data.value,
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => x_data.value.to_f64().unwrap() > y_data.value, 
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => x_data.value > y_data.value.to_f64().unwrap(),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => x_data.value > y_data.value,
    };

    GrometBool{value: result}
}

pub fn gte(x: GrometNumber, y: GrometNumber) -> GrometBool { // TODO: How to make these comparisons 
    let result: bool = match(x,y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => x_data.value >= y_data.value,
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => x_data.value.to_f64().unwrap() >= y_data.value, 
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => x_data.value >= y_data.value.to_f64().unwrap(),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => x_data.value >= y_data.value,
    };

    GrometBool{value: result}
}

pub fn lt(x: GrometNumber, y: GrometNumber) -> GrometBool { // TODO: How to make these comparisons 
    let result: bool = match(x,y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => x_data.value < y_data.value,
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => x_data.value.to_f64().unwrap() < y_data.value, 
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => x_data.value < y_data.value.to_f64().unwrap(),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => x_data.value < y_data.value,
    };

    GrometBool{value: result}
}

pub fn lte(x: GrometNumber, y: GrometNumber) -> GrometBool { // TODO: How to make these comparisons 
    let result: bool = match(x,y) {
        (GrometNumber::Int(x_data), GrometNumber::Int(y_data)) => x_data.value <= y_data.value,
        (GrometNumber::Int(x_data), GrometNumber::Float(y_data)) => x_data.value.to_f64().unwrap() <= y_data.value, 
        (GrometNumber::Float(x_data), GrometNumber::Int(y_data)) => x_data.value <= y_data.value.to_f64().unwrap(),
        (GrometNumber::Float(x_data), GrometNumber::Float(y_data)) => x_data.value <= y_data.value,
    };

    GrometBool{value: result}
}