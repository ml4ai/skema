use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::cast::FromPrimitive;

/*#[derive(Debug)]
enum GrometAny {
    Void,
    Number(Gromet_Number),
    Boolean(Gromet_Boolean),
    Character(Gromet_Character)
}

#[derive(Debug)]
enum GrometBoolean {
    Boolean(bool)
}

#[derive(Debug)]
enum GrometCharacter {
    Character(char)
}

#[derive(Debug)]
enum GrometReal {
    
}

#[derive(Debug)]
enum GrometComplex {
    
}

#[derive(Debug)]
enum GrometNumber {
    Real(Gromet_Real),
    Complex(Gromet_Complex)
}
*/

#[derive(Debug)]
#[derive(PartialEq)]
pub enum Number {
    Int(BigInt), // Have to use Bigint here to prevent overflow
    Float(f64)
    // The num::complex::Complex struct is generic, here we choose to use f64 as the generic type
    // parameter. However, it is unclear if this is the best choice, or whether we should also
    // explicitly have a version with an integer type as the generic type parameter.
}

impl From<i64> for Number {
    fn from(n: i64) -> Self {
        Number::Int(BigInt::from_i64(n).unwrap())
    }
}
impl From<f64> for Number {
    fn from(n: f64) -> Self {
        Number::Float(n)
    }
}

/*enum Sequence {
    Range(),
    List(Vec),
    Array(),
    Tuple(),
    Dataframe(),

}*/