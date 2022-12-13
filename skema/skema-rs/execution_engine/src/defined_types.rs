use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::cast::FromPrimitive;

#[derive(Debug, PartialEq)]
pub struct Int(pub BigInt);
#[derive(Debug, PartialEq)]
pub struct Float(pub f64);
#[derive(Debug, PartialEq)]
pub enum Number {
    Int(Int), // Have to use Bigint here to prevent overflow
    Float(Float) // Python float size
    // The num::complex::Complex struct is generic, here we choose to use f64 as the generic type
    // parameter. However, it is unclear if this is the best choice, or whether we should also
    // explicitly have a version with an integer type as the generic type parameter.
}
impl From<i64> for Number {
    fn from(n: i64) -> Self {
        Number::Int(Int(BigInt::from_i64(n).unwrap()))
    }
}
impl From<f64> for Number {
    fn from(n: f64) -> Self {
        Number::Float(Float(n))
    }
}
impl From<BigInt> for Number {
    fn from(n: BigInt) -> Self {
        Number::Int(Int(n))
    }
}

#[derive(Debug, PartialEq)]
pub struct Bool(pub bool);

#[derive(Debug, PartialEq)]
pub enum Sequence {
    Array(Vec<Any>), // We can't know the size of this at compile time, so it must be implemented as a Vec
    List(Vec<Any>)
}

#[derive(Debug, PartialEq)]
pub enum Any {
    Number(Number),
    Sequence(Sequence),
    Bool(Bool),
}
