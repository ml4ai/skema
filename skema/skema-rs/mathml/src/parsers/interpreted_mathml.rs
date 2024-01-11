//! This module contains parsers that perform some amount of preliminary domain-specific
//! interpretation of presentation MathML (e.g., globbing S(t) to an identifier S of type
//! 'function'). This is in contrast to the `generic_mathml.rs` module that contains parsers that
//! do not attempt to perform any interpretation but instead simply preserve the original MathML
//! document structure.

use crate::{
    ast::{
        operator::{Derivative, HatOp, Operator, PartialDerivative, SumUnderOver, GradSub},
        Ci, Differential, HatComp, Math, MathExpression, Mi, Mrow, SummationMath, Type,
    },
    parsers::generic_mathml::{
        add, attribute, cross, divide, dot, elem_many0, equals, etag, grad, hat, lparen, mean, mi,
        mn, msub, msubsup, mtext, multiply, rparen, stag, subtract, tag_parser, ws,
        xml_declaration, IResult, ParseError, Span,
    },
};

use nom::sequence::terminated;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt, value},
    multi::{many0, many1, separated_list1},
    sequence::{delimited, pair, preceded, tuple},
};

/// Function to parse operators. This function differs from the one in parsers::generic_mathml by
/// disallowing operators besides +, -, =, (, ) and ,.
pub fn operator(input: Span) -> IResult<Operator> {
    let (s, op) = ws(delimited(
        stag!("mo"),
        alt((
            add, subtract, multiply, divide, equals, lparen, rparen, mean, dot, cross, hat,
        )),
        etag!("mo"),
    ))(input)?;
    Ok((s, op))
}

/// Parses function of identifiers
/// Example: S(t,x) identifies (t,x) as identifiers.
fn parenthesized_identifier(input: Span) -> IResult<Vec<Mi>> {
    let mo_lparen = delimited(stag!("mo"), lparen, etag!("mo"));
    let mo_rparen = delimited(stag!("mo"), rparen, etag!("mo"));
    let mo_comma = delimited(stag!("mo"), ws(tag(",")), etag!("mo"));
    let (s, bound_vars) = delimited(mo_lparen, separated_list1(mo_comma, mi), mo_rparen)(input)?;
    Ok((s, bound_vars))
}

/// Parses function of identifiers
/// Example: Q_i ( t_{i-1}, s_{i-1} ) identifies ( t_{i-1}, s_{i-1} ) as identifiers.
fn parenthesized_msub_identifier(input: Span) -> IResult<Vec<Mi>> {
    let mo_lparen = delimited(stag!("mo"), lparen, etag!("mo"));
    let mo_rparen = delimited(stag!("mo"), rparen, etag!("mo"));
    let mo_mrow_lparen = delimited(tag("<mrow><mo>"), lparen, tag("</mo>"));
    let mo_mrow_rparen = delimited(tag("<mo>"), rparen, tag("</mo></mrow>"));
    let mo_comma = delimited(stag!("mo"), ws(tag(",")), etag!("mo"));
    let (s, bound_vars) = delimited(
        alt((mo_lparen, mo_mrow_lparen)),
        separated_list1(mo_comma, msub),
        alt((mo_mrow_rparen, mo_rparen)),
    )(input)?;
    let mut mi_func_of: Vec<Mi> = Vec::new();
    for bvar in bound_vars {
        let b = Mi(bvar.to_string());
        mi_func_of.push(b.clone());
    }
    Ok((s, mi_func_of))
}

/// Parse empty univariate function.
/// Example: S
fn empty_parenthesis(input: Span) -> IResult<Vec<Mi>> {
    let empty = vec![Mi("".to_string())];
    Ok((input, empty))
}

/// Parse content identifiers corresponding to univariate functions.
/// Example: S(t)
pub fn ci_univariate_with_bounds(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), bound_vars)) = tuple((
        mi,
        alt((parenthesized_identifier, parenthesized_msub_identifier)),
    ))(input)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in bound_vars {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    Ok((
        s,
        Ci::new(
            Some(Type::Function),
            Box::new(MathExpression::Mi(Mi(x.trim().to_string()))),
            Some(ci_func_of),
        ),
    ))
}

/// Parse content identifiers
/// Example: S
pub fn ci_univariate_without_bounds(input: Span) -> IResult<Ci> {
    let (s, Mi(x)) = mi(input)?;
    Ok((
        s,
        Ci::new(
            None,
            Box::new(MathExpression::Mi(Mi(x.trim().to_string()))),
            None,
        ),
    ))
}

/// Parse identifiers corresponding to univariate functions for ordinary derivatives
/// such that it can identify content identifiers with and without parenthesis identifiers.
pub fn ci_univariate_func(input: Span) -> IResult<Ci> {
    let (s, (x, bound_vars)) = tuple((
        mi,
        alt((
            parenthesized_identifier,
            empty_parenthesis,
            parenthesized_msub_identifier,
        )),
    ))(input)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in bound_vars {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    Ok((
        s,
        Ci::new(
            Some(Type::Function),
            Box::new(MathExpression::Mi(x)),
            Some(ci_func_of),
        ),
    ))
}

/// Parse content identifier for Msub
pub fn ci_subscript(input: Span) -> IResult<Ci> {
    let (s, x) = msub(input)?;
    Ok((s, Ci::new(None, Box::new(x), None)))
}

/// Parse contest identifier for Msub corresponding to univariate functions for ordinary
/// derivatives
pub fn ci_subscript_func(input: Span) -> IResult<Ci> {
    let (s, (x, bound_vars)) = tuple((
        msub,
        alt((
            parenthesized_msub_identifier,
            parenthesized_identifier,
            empty_parenthesis,
        )),
    ))(input)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in bound_vars {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    Ok((
        s,
        Ci::new(Some(Type::Function), Box::new(x), Some(ci_func_of)),
    ))
}

/// Parse content identifier for Msup
pub fn superscript(input: Span) -> IResult<MathExpression> {
    let (s, sup) = ws(map(
        tag_parser!("msup", pair(math_expression, math_expression)),
        |(x, y)| MathExpression::Msup(Box::new(x), Box::new(y)),
    ))(input)?;
    Ok((s, sup))
}

/// Parse Mover
pub fn over_term(input: Span) -> IResult<MathExpression> {
    let (s, over) = ws(map(
        alt((
            tag_parser!("mover", pair(math_expression, math_expression)),
            delimited(
                tag("<mrow><mover>"),
                pair(math_expression, math_expression),
                tag("</mover></mrow>"),
            ),
        )),
        |(x, y)| MathExpression::Mover(Box::new(x), Box::new(y)),
    ))(input)?;
    if let MathExpression::Mover(ref x, ref y) = over.clone() {
        if MathExpression::Mo(Operator::Hat) == **y {
            let new_op = Operator::HatOp(HatOp::new(x.clone()));
            return Ok((s, MathExpression::Mo(new_op)));
        } else {
            return Ok((s, over));
        }
    }
    Err(nom::Err::Error(ParseError::new(
        "Unable to obtain Mover term".to_string(),
        input,
    )))
}

/// Parse Hat operator with components. Example: r \hat{x}
pub fn hat_operator(input: Span) -> IResult<(MathExpression, MathExpression)> {
    let (s, (comp, op)) = pair(mi, over_term)(input)?;
    Ok((s, (op, MathExpression::Mi(comp))))
}

/// Parse the identifier 'd'
fn d(input: Span) -> IResult<()> {
    let (s, Mi(x)) = mi(input)?;
    if let "d" = x.as_ref() {
        Ok((s, ()))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify Mi('d')".to_string(),
            input,
        )))
    }
}

/// Parse the identifier '∂'
fn partial(input: Span) -> IResult<()> {
    let (s, Mi(x)) = mi(input)?;
    if let "∂" = x.as_ref() {
        Ok((s, ()))
    } else if let "&#x2202;" = x.as_ref() {
        Ok((s, ()))
    } else {
        Err(nom::Err::Error(ParseError::new(
            "Unable to identify Mi('∂')".to_string(),
            input,
        )))
    }
}

/// Parse a content identifier with function of elements.
/// Example: S(t,x)
pub fn ci_unknown_with_bounds(input: Span) -> IResult<Ci> {
    let (s, (x, bound_vars)) = pair(mi, parenthesized_identifier)(input)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in bound_vars {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    Ok((
        s,
        Ci::new(None, Box::new(MathExpression::Mi(x)), Some(ci_func_of)),
    ))
}

/// Parse a content identifier of unknown type.
/// Example: S
pub fn ci_unknown_without_bounds(input: Span) -> IResult<Ci> {
    let (s, x) = mi(input)?;
    Ok((s, Ci::new(None, Box::new(MathExpression::Mi(x)), None)))
}

/// Parse a content identifier of unknown type for ordinary derivatives.
/// such that it can identify content identifiers with and without parenthesis identifiers.
pub fn ci_unknown(input: Span) -> IResult<Ci> {
    let (s, (x, bound_vars)) = pair(mi, alt((parenthesized_identifier, empty_parenthesis)))(input)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in bound_vars {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    Ok((
        s,
        Ci::new(None, Box::new(MathExpression::Mi(x)), Some(ci_func_of)),
    ))
}

/// Parse first order derivative where the function of derivative is within a parenthesis
/// e.g. d/dt ( S(t)* I(t) )
pub fn first_order_with_func_in_parenthesis(input: Span) -> IResult<(Derivative, Mrow)> {
    let (s, _) = pair(stag!("mfrac"), d)(input)?;
    let (s, with_respect_to) = delimited(
        tuple((stag!("mrow"), d)),
        mi,
        pair(etag!("mrow"), etag!("mfrac")),
    )(s)?;
    let (s, mut func) = ws(preceded(
        tuple((stag!("mo"), lparen, etag!("mo"))),
        many0(math_expression),
    ))(s)?;
    let Some(_) = func.pop() else { todo!() };
    Ok((
        s,
        (
            Derivative::new(
                1,
                1,
                Ci::new(
                    Some(Type::Real),
                    Box::new(MathExpression::Mi(with_respect_to)),
                    None,
                ),
            ),
            Mrow(func),
        ),
    ))
}

/// Parse first order partial derivative where the function of derivative is within a parenthesis
/// e.g. ∂/∂t ( S(t)* I(t) )
pub fn first_order_partial_with_func_in_parenthesis(
    input: Span,
) -> IResult<(PartialDerivative, Mrow)> {
    let (s, _) = pair(stag!("mfrac"), partial)(input)?;
    let (s, with_respect_to) = delimited(
        tuple((stag!("mrow"), partial)),
        mi,
        pair(etag!("mrow"), etag!("mfrac")),
    )(s)?;
    let (s, mut func) = ws(preceded(
        tuple((stag!("mo"), lparen, etag!("mo"))),
        many0(math_expression),
    ))(s)?;
    let Some(_) = func.pop() else { todo!() };
    Ok((
        s,
        (
            PartialDerivative::new(
                1,
                1,
                Ci::new(
                    Some(Type::Real),
                    Box::new(MathExpression::Mi(with_respect_to)),
                    None,
                ),
            ),
            Mrow(func),
        ),
    ))
}

/// Parse a first-order ordinary derivative written in Leibniz notation.
pub fn first_order_derivative_leibniz_notation(input: Span) -> IResult<(Derivative, Ci)> {
    let (s, _) = tuple((stag!("mfrac"), stag!("mrow"), alt((d, partial))))(input)?;
    let (s, func) = ws(alt((
        ci_univariate_func,
        map(
            ci_unknown,
            |Ci {
                 content, func_of, ..
             }| {
                Ci {
                    r#type: Some(Type::Function),
                    content,
                    func_of,
                }
            },
        ),
        ci_subscript_func,
    )))(s)?;
    let (s, with_respect_to) = delimited(
        tuple((etag!("mrow"), stag!("mrow"), alt((d, partial)))),
        mi,
        pair(etag!("mrow"), etag!("mfrac")),
    )(s)?;

    if let Some(ref ci_vec) = func.func_of {
        for (indx, bvar) in ci_vec.iter().enumerate() {
            if Some(bvar.content.clone())
                == Some(Box::new(MathExpression::Mi(with_respect_to.clone())))
            {
                return Ok((
                    s,
                    (
                        Derivative::new(
                            1,
                            (indx + 1) as u8,
                            Ci::new(
                                Some(Type::Real),
                                Box::new(MathExpression::Mi(with_respect_to)),
                                None,
                            ),
                        ),
                        func,
                    ),
                ));
            } else if Some(bvar.content.clone())
                == Some(Box::new(MathExpression::Mi(Mi("".to_string()))))
            {
                return Ok((
                    s,
                    (
                        Derivative::new(
                            1,
                            1,
                            Ci::new(
                                Some(Type::Real),
                                Box::new(MathExpression::Mi(with_respect_to)),
                                None,
                            ),
                        ),
                        func,
                    ),
                ));
            }
        }
    }

    Err(nom::Err::Error(ParseError::new(
        "Unable to match  function_of  with with_respect_to".to_string(),
        input,
    )))
}

/// Parse first order partial derivative. Example: ∂_{t} S
pub fn first_order_partial_derivative_partial_func(
    input: Span,
) -> IResult<(PartialDerivative, Ci)> {
    let (s, _) = tuple((stag!("msub"), partial))(input)?;
    let (s, with_respect_to) = ws(terminated(mi, etag!("msub")))(s)?;
    let (s, func) = ws(alt((
        ci_univariate_func,
        map(
            ci_unknown,
            |Ci {
                 content, func_of, ..
             }| {
                Ci {
                    r#type: Some(Type::Function),
                    content,
                    func_of,
                }
            },
        ),
        ci_subscript_func,
    )))(s)?;
    if let Some(ref ci_vec) = func.func_of {
        for (indx, bvar) in ci_vec.iter().enumerate() {
            if Some(bvar.content.clone())
                == Some(Box::new(MathExpression::Mi(with_respect_to.clone())))
            {
                return Ok((
                    s,
                    (
                        PartialDerivative::new(
                            1,
                            (indx + 1) as u8,
                            Ci::new(
                                Some(Type::Real),
                                Box::new(MathExpression::Mi(with_respect_to)),
                                None,
                            ),
                        ),
                        func,
                    ),
                ));
            } else if Some(bvar.content.clone())
                == Some(Box::new(MathExpression::Mi(Mi("".to_string()))))
            {
                return Ok((
                    s,
                    (
                        PartialDerivative::new(
                            1,
                            1,
                            Ci::new(
                                Some(Type::Real),
                                Box::new(MathExpression::Mi(with_respect_to)),
                                None,
                            ),
                        ),
                        func,
                    ),
                ));
            }
        }
    }

    Err(nom::Err::Error(ParseError::new(
        "Unable to match  function_of  with with_respect_to in ∂_{t} S".to_string(),
        input,
    )))
}

/// Parse a first-order partial ordinary derivative written in Leibniz notation.
pub fn first_order_partial_derivative_leibniz_notation(
    input: Span,
) -> IResult<(PartialDerivative, Ci)> {
    let (s, _) = tuple((stag!("mfrac"), stag!("mrow"), partial))(input)?;
    let (s, func) = ws(alt((
        ci_univariate_func,
        map(
            ci_unknown,
            |Ci {
                 content, func_of, ..
             }| {
                Ci {
                    r#type: Some(Type::Function),
                    content,
                    func_of,
                }
            },
        ),
        ci_subscript_func,
    )))(s)?;
    let (s, with_respect_to) = delimited(
        tuple((etag!("mrow"), stag!("mrow"), partial)),
        mi,
        pair(etag!("mrow"), etag!("mfrac")),
    )(s)?;

    if let Some(ref ci_vec) = func.func_of {
        for (indx, bvar) in ci_vec.iter().enumerate() {
            if Some(bvar.content.clone())
                == Some(Box::new(MathExpression::Mi(with_respect_to.clone())))
            {
                return Ok((
                    s,
                    (
                        PartialDerivative::new(
                            1,
                            (indx + 1) as u8,
                            Ci::new(
                                Some(Type::Real),
                                Box::new(MathExpression::Mi(with_respect_to)),
                                None,
                            ),
                        ),
                        func,
                    ),
                ));
            } else if Some(bvar.content.clone())
                == Some(Box::new(MathExpression::Mi(Mi("".to_string()))))
            {
                return Ok((
                    s,
                    (
                        PartialDerivative::new(
                            1,
                            1,
                            Ci::new(
                                Some(Type::Real),
                                Box::new(MathExpression::Mi(with_respect_to)),
                                None,
                            ),
                        ),
                        func,
                    ),
                ));
            }
        }
    }

    Err(nom::Err::Error(ParseError::new(
        "Unable to match  function_of  with with_respect_to".to_string(),
        input,
    )))
}

pub fn newtonian_derivative(input: Span) -> IResult<(Derivative, Ci)> {
    // Get number of dots to recognize the order of the derivative
    let n_dots = delimited(
        stag!("mo"),
        alt((
            map(many1(nom::character::complete::char('˙')), |x| {
                x.len() as u8
            }),
            value(1_u8, tag("&#x02D9;")),
        )),
        etag!("mo"),
    );

    let (s, (func, order)) = delimited(
        alt((tag("<mrow><mover>"), tag("<mover>"))),
        pair(
            map(
                ci_unknown,
                |Ci {
                     content, func_of, ..
                 }| {
                    Ci {
                        r#type: Some(Type::Function),
                        content,
                        func_of,
                    }
                },
            ),
            n_dots,
        ),
        alt((tag("</mover></mrow>"), etag!("mover"))),
    )(input)?;
    let (s, mi_func_of) = alt((parenthesized_identifier, empty_parenthesis))(s)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in mi_func_of {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    let func_mi = ci_func_of.get(0).unwrap().content.clone();
    let new_with_respect_to: Box<MathExpression> = Box::new(MathExpression::Ci(Ci {
        r#type: None,
        content: Box::new(MathExpression::Mi(Mi(func_mi.to_string()))),
        func_of: None,
    }));

    // Loop over vector of `func_of` of type Ci
    if let Some(ref ci_vec) = func.func_of {
        for (_, bvar) in ci_vec.iter().enumerate() {
            if let Some(with_respect_to) = Some(bvar.content.clone()) {
                let _new_with_respect_to = with_respect_to;
            }
        }
    }
    Ok((
        s,
        (
            Derivative::new(
                order,
                1,
                Ci::new(Some(Type::Real), new_with_respect_to, None),
            ),
            func,
        ),
    ))
}

// We reimplement the mfrac and mrow parsers in this file (instead of importing them from
// the generic_mathml module) to work with the specialized version of the math_expression parser
// (also in this file).
pub fn mfrac(input: Span) -> IResult<MathExpression> {
    let (s, frac) = ws(map(
        tag_parser!("mfrac", pair(math_expression, math_expression)),
        |(x, y)| MathExpression::Mfrac(Box::new(x), Box::new(y)),
    ))(input)?;

    Ok((s, frac))
}

pub fn mrow(input: Span) -> IResult<Mrow> {
    let (s, elements) = ws(delimited(
        stag!("mrow"),
        many0(math_expression),
        etag!("mrow"),
    ))(input)?;
    Ok((s, Mrow(elements)))
}

///Absolute value
pub fn absolute(input: Span) -> IResult<MathExpression> {
    let (s, elements) = ws(delimited(
        tag("<mo>|</mo>"),
        many0(math_expression),
        tag("<mo>|</mo>"),
    ))(input)?;
    let components = MathExpression::Absolute(
        Box::new(MathExpression::Mo(Operator::Abs)),
        Box::new(MathExpression::Mrow(Mrow(elements))),
    );

    Ok((s, components))
}

/// Example: Divergence
pub fn div(input: Span) -> IResult<Operator> {
    let (s, _op) = ws(pair(gradient, ws(delimited(stag!("mo"), dot, etag!("mo")))))(input)?;
    let div = Operator::Div;
    Ok((s, div))
}

pub fn gradient(input: Span) -> IResult<Operator> {
    let (s, op) = ws(alt((
        delimited(stag!("mo"), grad, etag!("mo")),
        delimited(stag!("mi"), grad, etag!("mi")),
    )))(input)?;
    Ok((s, op))
}

/// Gradient sub  E.g. ∇_{x}
pub fn gradient_subscript(input: Span) -> IResult<Operator> {
    let (s, _) = tuple((stag!("msub"), gradient))(input)?;
    let (s, mi) = ws(terminated(mi, etag!("msub")))(s)?;
    let grad_sub = Operator::GradSub(GradSub::new(Box::new(MathExpression::Mi(mi.clone()))));
    Ok((s, grad_sub))
}

pub fn grad_func(input: Span) -> IResult<(Operator, Ci)> {
    let (s, (op, id)) = ws(pair(gradient, mi))(input)?;
    let ci = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(id)), None);
    Ok((s, (op, ci)))
}

pub fn functions_of_grad(input: Span) -> IResult<(Operator, Mrow)> {
    let (s, (op, id)) = ws(pair(gradient, map(ws(many0(math_expression)), Mrow)))(input)?;
    Ok((s, (op, id)))
}

///Absolute with Msup value
pub fn absolute_with_msup(input: Span) -> IResult<MathExpression> {
    let (s, sup) = ws(map(
        ws(delimited(
            ws(tuple((stag!("mo"), tag("|"), etag!("mo")))),
            //ws(tag("<mo>|</mo>")),
            ws(tuple((
                //math_expression,
                map(ws(many0(math_expression)), Mrow),
                preceded(ws(tag("<msup><mo>|</mo>")), ws(math_expression)),
            ))),
            ws(tag("</msup>")),
        )),
        |(x, y)| MathExpression::AbsoluteSup(Box::new(MathExpression::Mrow(x)), Box::new(y)),
    ))(input)?;
    Ok((s, sup))
}

///Parenthesis with Msup value
pub fn paren_as_msup(input: Span) -> IResult<MathExpression> {
    let (s, sup) = ws(map(
        ws(delimited(
            tag("<mo>(</mo>"),
            tuple((
                map(many0(math_expression), Mrow),
                preceded(tag("<msup><mo>)</mo>"), math_expression),
            )),
            tag("</msup>"),
        )),
        |(x, y)| MathExpression::Msup(Box::new(MathExpression::Mrow(x)), Box::new(y)),
    ))(input)?;
    Ok((s, sup))
}

/// Parser for multiple elements in square root
pub fn sqrt(input: Span) -> IResult<MathExpression> {
    let (s, elements) = ws(delimited(
        stag!("msqrt"),
        many0(math_expression),
        etag!("msqrt"),
    ))(input)?;
    Ok((
        s,
        MathExpression::Msqrt(Box::new(MathExpression::Mrow(Mrow(elements)))),
    ))
}

/// Parser for change in a variable :
/// Example: Δx
pub fn change_in_variable(input: Span) -> IResult<Ci> {
    let (s, elements) = ws(preceded(
        alt((tag("<mi>Δ</mi>"), tag("<mi>&#x0394;</mi>"))),
        math_expression,
    ))(input)?;
    let temp_sum = format!("Δ{}", elements.to_string());
    let change_in_var = Ci::new(
        Some(Type::Real),
        Box::new(MathExpression::Mi(Mi(temp_sum.to_string()))),
        None,
    );
    Ok((s, change_in_var))
}

/// Parser handles vector identity notation.
/// E.g. (v ⋅ ∇) u
pub fn gradient_with_closed_paren(input: Span) -> IResult<Vec<MathExpression>> {
    /*let(s,(mi, mi2)) = ws(tuple((
        preceded(tag("<mo>(</mo>"), mi),
        preceded(ws(tag("<mo>&#x22c5;</mo><mi>&#x2207;</mi><mo>)</mo>")), mi)
        ))
    )(input)?;*/
    println!(".");
    //println!("input={:?}", input);
    /*for bvar in bound_vars {
        let b = Mi(bvar.to_string());
        expression.push(b.clone());
    }*/
    /*let(s, (mi, dd)) = ws(delimited(tag("<mo>(</mo>"),
                                         pair(mi, dot),
                                         pair(gradient,tag("<mo>)</mo>")))
    )(input)?;*/
    let (s, (lp, (mi, (op, gg)))) = ws(pair(tag("<mo>(</mo>"),
                                      pair(mi, pair(ws(delimited(stag!("mo"), dot, etag!("mo"))), terminated(gradient, tag("<mo>)</mo>"))))
    ))(input)?;
    println!("..");
    println!("mi={:?}", mi);
    //let (s, _) = ws(delimited(stag!("mo"), dot, etag!("mo")))(input)?;
    println!("cdot={:?}", op);
    println!("gg={:?}", gg);
    println!("...");

    //let(s, (lp,mi, cdot, grad,rp)) = ws(tuple(( delimited(stag!("mo"), lparen, etag!("mo")),
                                      //        mi, dot,  delimited(stag!("mi"), tag("∇"), etag!("mi")),
                                                //delimited(stag!("mo"), rparen, etag!("mo")))))(input)?;
    //let (s, (op, _)) = ws(pair(gradient, rparen))(input)?;
    //println!("mi={:?}", mi);
   // println!("cdot={:?}", dd);
    //println!("mi2={:?}", mi2);
    let mut expression: Vec<MathExpression> = Vec::new();
    expression.push(MathExpression::Mi(mi));
    println!("expression={:?}", expression);
    expression.push(MathExpression::Mo(op));
    println!("expression={:?}", expression);
    println!("---");
    let ci = Ci::new(
        Some(Type::Real),
        Box::new(MathExpression::Mi(Mi("Grad".to_string()))),
        None,
    );
    expression.push(MathExpression::Ci(ci.clone()));
    println!("expression={:?}", expression);
    //let mi = Mi::new("Grad".to_string());
    Ok((s, expression))
}

/// Parser handles e.g. `...` in the equation
pub fn multiple_dots(input: Span) -> IResult<Ci> {
    let (s, x) = ws(delimited(tag("<mo>"), tag("…"), tag("</mo>")))(input)?;
    let ci = Ci::new(
        Some(Type::List),
        Box::new(MathExpression::Mi(Mi(x.to_string()))),
        None,
    );
    Ok((s, ci))
}

/// Handles summation as operator
pub fn munderover_summation(input: Span) -> IResult<(SumUnderOver, Mrow)> {
    let (s, (under, over)) = ws(delimited(
        alt((
            tag("<munderover><mo>∑</mo>"),
            tag("<munderover><mo>&#x2211;</mo>"),
        )),
        pair(
            ws(delimited(
                tag("<mrow>"),
                many0(math_expression),
                tag("</mrow>"),
            )),
            many0(math_expression),
        ),
        tag("</munderover>"),
    ))(input)?;
    let (s, comps) = many0(math_expression)(s)?;
    let sum_operator = Operator::Sum;
    let under_comp = MathExpression::Mrow(Mrow(under));
    let over_comp = MathExpression::Mrow(Mrow(over));
    let other_comps = Mrow::new(comps);
    let operator = SumUnderOver::new(
        Box::new(MathExpression::Mo(sum_operator)),
        Box::new(under_comp),
        Box::new(over_comp),
    );
    Ok((s, (operator, other_comps)))
}

/// Parser for math expressions. This varies from the one in the generic_mathml module, since it
/// assumes that expressions such as S(t) are actually univariate functions.
pub fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((
        alt((map(gradient_with_closed_paren, |row| MathExpression::Mrow(Mrow(row))),
        map(gradient_subscript, MathExpression::Mo),
        )),
        alt((
            map(div, MathExpression::Mo),
            map(hat_operator, |(op, row)| {
                MathExpression::HatComp(HatComp {
                    op: Box::new(op),
                    comp: Box::new(row),
                })
            }),
        )),
        map(
            first_order_partial_derivative_partial_func,
            |(
                PartialDerivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::PartialDerivative(
                        PartialDerivative {
                            order,
                            var_index,
                            bound_var,
                        },
                    ))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        alt((
            ws(absolute_with_msup),
            ws(paren_as_msup),
            map(change_in_variable, MathExpression::Ci),
            //sqrt,
            map(
                munderover_summation,
                |(SumUnderOver { op, under, over }, comp)| {
                    MathExpression::SummationMath(SummationMath {
                        op: Box::new(MathExpression::Mo(Operator::SumUnderOver(SumUnderOver {
                            op,
                            under,
                            over,
                        }))),
                        func: Box::new(MathExpression::Mrow(comp)),
                    })
                },
            ),
        )),
        map(
            grad_func,
            |(
                op,
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(op)),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            first_order_derivative_leibniz_notation,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            first_order_partial_derivative_leibniz_notation,
            |(
                PartialDerivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::PartialDerivative(
                        PartialDerivative {
                            order,
                            var_index,
                            bound_var,
                        },
                    ))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            newtonian_derivative,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            ci_univariate_with_bounds,
            |Ci {
                 content, func_of, ..
             }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Real),
                    content,
                    func_of,
                })
            },
        ),
        map(
            ci_subscript_func,
            |Ci {
                 content, func_of, ..
             }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Real),
                    content,
                    func_of,
                })
            },
        ),
        map(
            first_order_with_func_in_parenthesis,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                comp,
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Mrow(comp)),
                })
            },
        ),
        map(
            first_order_partial_with_func_in_parenthesis,
            |(
                PartialDerivative {
                    order,
                    var_index,
                    bound_var,
                },
                comp,
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::PartialDerivative(
                        PartialDerivative {
                            order,
                            var_index,
                            bound_var,
                        },
                    ))),
                    func: Box::new(MathExpression::Mrow(comp)),
                })
            },
        ),
        map(functions_of_grad, |(op, comp)| {
            MathExpression::Differential(Differential {
                diff: Box::new(MathExpression::Mo(op)),
                func: Box::new(MathExpression::Mrow(comp)),
            })
        }),
        map(ci_univariate_without_bounds, MathExpression::Ci),
        map(ci_subscript, MathExpression::Ci),
        map(multiple_dots, MathExpression::Ci),
        map(
            ci_unknown_with_bounds,
            |Ci {
                 content, func_of, ..
             }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Real),
                    content,
                    func_of,
                })
            },
        ),
        map(ci_unknown_without_bounds, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(Type::Real),
                content,
                func_of: None,
            })
        }),
        alt((
            absolute,
            sqrt,
            map(operator, MathExpression::Mo),
            map(gradient, MathExpression::Mo),
            mn,
            msub,
            superscript,
            mfrac,
            mtext,
            over_term,
        )),
        map(mrow, MathExpression::Mrow),
        msubsup,
    )))(input)
}
/*pub fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((
        map(div, MathExpression::Mo),
        map(
            first_order_partial_derivative_partial_func,
            |(
                PartialDerivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::PartialDerivative(
                        PartialDerivative {
                            order,
                            var_index,
                            bound_var,
                        },
                    ))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        alt((
            ws(absolute_with_msup),
            ws(paren_as_msup),
            map(change_in_variable, MathExpression::Ci),
            //sqrt,
            map(
                munderover_summation,
                |(SumUnderOver { op, under, over }, comp)| {
                    MathExpression::SummationMath(SummationMath {
                        op: Box::new(MathExpression::Mo(Operator::SumUnderOver(SumUnderOver {
                            op,
                            under,
                            over,
                        }))),
                        func: Box::new(MathExpression::Mrow(comp)),
                    })
                },
            ),
            map(
                grad_func,
                |(
                    op,
                    Ci {
                        r#type,
                        content,
                        func_of,
                    },
                )| {
                    MathExpression::Differential(Differential {
                        diff: Box::new(MathExpression::Mo(op)),
                        func: Box::new(MathExpression::Ci(Ci {
                            r#type,
                            content,
                            func_of,
                        })),
                    })
                },
            ),
            map(functions_of_grad, |(op, comp)| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(op)),
                    func: Box::new(MathExpression::Mrow(comp)),
                })
            }),
            map(hat_operator, |(op, row)| {
                MathExpression::HatComp(HatComp {
                    op: Box::new(op),
                    comp: Box::new(row),
                })
            }),
        )),
        map(
            first_order_derivative_leibniz_notation,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            first_order_partial_derivative_leibniz_notation,
            |(
                PartialDerivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::PartialDerivative(
                        PartialDerivative {
                            order,
                            var_index,
                            bound_var,
                        },
                    ))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            newtonian_derivative,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                Ci {
                    r#type,
                    content,
                    func_of,
                },
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Ci(Ci {
                        r#type,
                        content,
                        func_of,
                    })),
                })
            },
        ),
        map(
            ci_univariate_with_bounds,
            |Ci {
                 content, func_of, ..
             }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Real),
                    content,
                    func_of,
                })
            },
        ),
        map(
            ci_subscript_func,
            |Ci {
                 content, func_of, ..
             }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Real),
                    content,
                    func_of,
                })
            },
        ),
        map(
            first_order_with_func_in_parenthesis,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                comp,
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Mrow(comp)),
                })
            },
        ),
        map(
            first_order_partial_with_func_in_parenthesis,
            |(
                PartialDerivative {
                    order,
                    var_index,
                    bound_var,
                },
                comp,
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::PartialDerivative(
                        PartialDerivative {
                            order,
                            var_index,
                            bound_var,
                        },
                    ))),
                    func: Box::new(MathExpression::Mrow(comp)),
                })
            },
        ),
        map(ci_univariate_without_bounds, MathExpression::Ci),
        map(ci_subscript, MathExpression::Ci),
        map(multiple_dots, MathExpression::Ci),
        map(
            ci_unknown_with_bounds,
            |Ci {
                 content, func_of, ..
             }| {
                MathExpression::Ci(Ci {
                    r#type: Some(Type::Real),
                    content,
                    func_of,
                })
            },
        ),
        map(ci_unknown_without_bounds, |Ci { content, .. }| {
            MathExpression::Ci(Ci {
                r#type: Some(Type::Real),
                content,
                func_of: None,
            })
        }),
        alt((
            map(
                grad_func,
                |(
                    op,
                    Ci {
                        r#type,
                        content,
                        func_of,
                    },
                )| {
                    MathExpression::Differential(Differential {
                        diff: Box::new(MathExpression::Mo(op)),
                        func: Box::new(MathExpression::Ci(Ci {
                            r#type,
                            content,
                            func_of,
                        })),
                    })
                },
            ),
            absolute,
            sqrt,
            map(operator, MathExpression::Mo),
            map(gradient, MathExpression::Mo),
            mn,
            msub,
            superscript,
            mfrac,
            mtext,
            over_term,
            map(mrow, MathExpression::Mrow),
            msubsup,
        )),
    )))(input)
}*/

/// Parser for interpreted math expressions.
/// testing MathML documents
pub fn interpreted_math(input: Span) -> IResult<Math> {
    let (s, elements) = preceded(
        opt(xml_declaration),
        alt((
            ws(delimited(
                ws(tag("<math>")),
                ws(many0(math_expression)),
                ws(tag("<mo>,</mo></math>")),
            )),
            ws(delimited(
                ws(tag("<math>")),
                ws(many0(math_expression)),
                ws(tag("<mo>.</mo></math>")),
            )),
            elem_many0!("math"),
        )),
    )(input)?;
    Ok((s, Math { content: elements }))
}
