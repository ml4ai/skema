#[macro_export]
macro_rules! new_List {
    ( $( $x:expr ),* ) => {
        {
            vec![$($x),*]
        }
    };
}

#[test]
fn test_new_List() {
    assert_eq!(vec![1, 2, 3], new_List![1, 2, 3]);
}
