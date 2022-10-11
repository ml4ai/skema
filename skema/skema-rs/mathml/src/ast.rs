#[derive(Debug, PartialEq, Clone)]
pub enum MathExpression<'a> {
    Mi(&'a str),
    Mo(&'a str),
    Mrow(Vec<MathExpression<'a>>),
    Mfrac(Box<MathExpression<'a>>, Box<MathExpression<'a>>),
    Msub(Box<MathExpression<'a>>, Box<MathExpression<'a>>),
}

//impl Iterator for MathExpression {
    //type Item = MathExpression;

    //fn next(&mut self) -> Option<Self::Item> {
        //match 

    //}
//}

#[derive(Debug, PartialEq)]
pub struct Math<'a> {
    pub content: Vec<MathExpression<'a>>,
}
