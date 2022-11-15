use crate::execution_engine::types::defined_types::Field;
use num::complex::Complex;

enum Number {
    Int(isize),
    // The num::complex::Complex struct is generic, here we choose to use f64 as the generic type
    // parameter. However, it is unclear if this is the best choice, or whether we should also
    // explicitly have a version with an integer type as the generic type parameter.
    Complex(Complex<f64>),
    Float(f64),
}

trait Operator {
    fn shorthand() -> &'static str;
    fn documentation() -> &'static str;
    fn outputs() -> Vec<Field>;
}

struct Add {}

/// Note: This implementation of exec for the Add struct works even without specifying the type of
/// the arguments to be Number, due to the generic type argument T, which is constrained to
/// implement the std::ops::Add trait. If we want to restrict the input type to Number, we will need
/// to implement the std::ops::Add trait for Number.
impl Add {
    fn exec<T: std::ops::Add<Output = T>>(augend: T, addend: T) -> T {
        augend + addend
    }
}

impl Operator for Add {
    fn shorthand() -> &'static str {
        "+"
    }
    fn documentation() -> &'static str {
        "Add is the numerical addition operator. For a general addition operation (For example, the case of concatanation with +) see GenAdd."
    }
    fn outputs() -> Vec<Field> {
        vec![
            Field::new("augend", "Number"),
            Field::new("addend", "Number"),
        ]
    }
}

#[test]
fn test_add() {
    // Test integer addition
    assert_eq!(Add::exec(1, 2), 3);

    // Test floating point addition
    assert_eq!(Add::exec(1.2, 2.3), 3.5);

    // Test complex addition
    assert_eq!(
        Add::exec(Complex::new(1.2, 2.3), Complex::new(4.5, 6.7)),
        Complex::new(5.7, 9.0)
    );

    // Test trait implementation
    assert_eq!(Add::shorthand(), "+");
    assert_eq!(Add::documentation(),"Add is the numerical addition operator. For a general addition operation (For example, the case of concatanation with +) see GenAdd.");
    assert_eq!(
        Add::outputs(),
        vec![
            Field::new("augend", "Number"),
            Field::new("addend", "Number"),
        ]
    )
}
