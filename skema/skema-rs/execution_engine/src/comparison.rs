use super::defined_types::{GrometBool, GrometInt, GrometNumber, Int};
use num_bigint::BigInt;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::cmp::{Ordering, PartialEq, PartialOrd};

impl PartialEq<Int> for f64 {
    fn eq(&self, other: &Int) -> bool {
        &other
            .0
            .to_f64()
            .expect(&format!("Unable to convert {} to f64!", other.0))
            == self
    }
}

impl PartialOrd<Int> for f64 {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        self.partial_cmp(
            &other
                .0
                .to_f64()
                .expect(&format!("Unable to convert {:?} to f64!", &other)),
        )
    }
}

impl PartialEq<f64> for Int {
    fn eq(&self, other: &f64) -> bool {
        self.0
            .to_f64()
            .expect(&format!("Unable to convert {} to f64!", self.0))
            == *other
    }
}

impl PartialOrd<f64> for Int {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.0
            .to_f64()
            .expect(&format!("Unable to convert {} to f64!", self.0))
            .partial_cmp(other)
    }
}

pub fn gt<T: PartialOrd<U>, U>(x: T, y: U) -> bool {
    x > y
}

pub fn gte<T: PartialOrd<U>, U>(x: T, y: U) -> bool {
    x >= y
}

pub fn lt<T: PartialOrd<U>, U>(x: T, y: U) -> bool {
    x < y
}

pub fn lte<T: PartialOrd<U>, U>(x: T, y: U) -> bool {
    x <= y
}

#[test]
fn test_gt() {
    assert!(gt(3, 1));
    assert!(gt(Int::from(3), Int::from(1)));
    assert!(gt(Int::from(3), 1.1));
}

#[test]
fn test_gte() {
    assert!(gte(3, 1));
    assert!(gte(Int::from(3), Int::from(1)));
    assert!(gte(Int::from(3), 1.1));
}

#[test]
fn test_lt() {
    assert!(lt(1, 3));
    assert!(lt(Int::from(1), Int::from(3)));
    assert!(lt(1.1, Int::from(3)));
}

#[test]
fn test_lte() {
    assert!(lte(1, 3));
    assert!(lte(Int::from(1), Int::from(3)));
    assert!(lte(1.1, Int::from(3)));
}
