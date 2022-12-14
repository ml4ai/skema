use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::cast::FromPrimitive;

#[derive(Debug, PartialEq)]
pub struct GrometInt{
    pub value: BigInt
}
#[derive(Debug, PartialEq)]
pub struct GrometFloat{
    pub value: f64
}
#[derive(Debug, PartialEq)]
pub enum GrometNumber {
    Int(GrometInt), // Have to use Bigint here to prevent overflow
    Float(GrometFloat) // Python float size
    // The num::complex::Complex struct is generic, here we choose to use f64 as the generic type
    // parameter. However, it is unclear if this is the best choice, or whether we should also
    // explicitly have a version with an integer type as the generic type parameter.
}
impl From<i64> for GrometNumber {
    fn from(n: i64) -> Self {
        let j: BigInt = BigInt::from_i64(n).unwrap();
        GrometNumber::Int(GrometInt{value: j})
    }
}
impl From<f64> for GrometNumber {
    fn from(n: f64) -> Self {
        GrometNumber::Float(GrometFloat{value: n})
    }
}
impl From<BigInt> for GrometNumber {
    fn from(n: BigInt) -> Self {
        GrometNumber::Int(GrometInt{value: n})
    }
}

#[derive(Debug, PartialEq)]
pub struct GrometBool{
    pub value: bool
}

#[derive(Debug, PartialEq)]
pub enum GrometSequence {
    Array(Vec<GrometAny>), // We can't know the size of this at compile time, so it must be implemented as a Vec
    List(Vec<GrometAny>)
}

#[derive(Debug, PartialEq)]
pub enum GrometAny {
    Number(GrometNumber),
    Sequence(GrometSequence),
    Bool(GrometBool),
}
