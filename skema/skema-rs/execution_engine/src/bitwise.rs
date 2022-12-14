use super::defined_types::GrometNumber;
use super::defined_types::GrometInt;
use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::cast::FromPrimitive;

// TODO: 
// 1. Add Unit Tests
// 2. Research support for float bitwise operators
// 3. DONE - If only for int, use Int variant as type instead of Number

fn bit_and(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value & y.value;
    GrometInt{value: result}
}

fn bit_or(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value | y.value;
    GrometInt{value: result}
}

fn bit_xor(x: GrometInt, y: GrometInt) -> GrometInt {
    let result: BigInt = x.value ^ y.value;
    GrometInt{value: result}
}

fn lshift(x: GrometInt, y: GrometInt) -> GrometInt {
    // BigInt doesn't support shifting by BigInt, so the shift should be cast to Int. 
    // Using largest int type posible here - i128
    let result: BigInt = x.value << y.value.to_i128().unwrap();
    GrometInt{value: result}
}

fn rshift(x: GrometInt, y: GrometInt) -> GrometInt {
    // BigInt doesn't support shifting by BigInt, so the shift should be cast to Int. 
    // Using largest int type posible here - i128
    let result: BigInt = x.value >> y.value.to_i128().unwrap();
    GrometInt{value: result}
}