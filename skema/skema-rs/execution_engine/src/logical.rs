fn and(x: bool, y: bool) -> bool {
    x && y
}

fn or(x: bool, y: bool) -> bool {
    x || y
}

#[test]
fn test_and() {
    assert_eq!(and(true, false), false);
    assert_eq!(and(true, true), true);
}

#[test]
fn test_or() {
    assert_eq!(or(true, false), true);
    assert_eq!(or(false, false), false);
}
