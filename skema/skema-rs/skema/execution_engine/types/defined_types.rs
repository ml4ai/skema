use num::complex::Complex;
use num_bigint::BigInt;

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
enum Number {
    Int(BigInt), // Have to use Bigint here to prevent overflow
    Float(f64)
    // The num::complex::Complex struct is generic, here we choose to use f64 as the generic type
    // parameter. However, it is unclear if this is the best choice, or whether we should also
    // explicitly have a version with an integer type as the generic type parameter.
}

enum Sequence {
    List(),
    Array(),
    Tuple(),
    Dataframe(),

}
fn unwrap_number(number: Number) {
    match number {
        Int(value) => return value ,
        Float(value) => return value,
        Complex(value) => return value,
        _ => println!("Failure")
    }
}

fn unwrap_int(int: Bigint) -> Bigint{
    
}