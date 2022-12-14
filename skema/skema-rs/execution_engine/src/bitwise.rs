use std::ops::{BitAnd, BitOr, BitXor, Shl, Shr};

fn bit_and<T: BitAnd<U, Output = V>, U, V>(x: T, y: U) -> V {
    x & y
}

fn bit_or<T: BitOr<U, Output = V>, U, V>(x: T, y: U) -> V {
    x | y
}

fn bit_xor<T: BitXor<U, Output = V>, U, V>(x: T, y: U) -> V {
    x ^ y
}

fn lshift<T: Shl<U, Output = V>, U, V>(x: T, y: U) -> V {
    x << y
}

fn rshift<T: Shr<U, Output = V>, U, V>(x: T, y: U) -> V {
    x >> y
}

#[test]
fn test_bit_and() {
    assert_eq!(bit_and(true, true), true)
}

#[test]
fn test_bit_or() {
    assert_eq!(bit_or(true, true), true)
}

#[test]
fn test_bit_xor() {
    assert_eq!(bit_xor(true, true), false)
}
