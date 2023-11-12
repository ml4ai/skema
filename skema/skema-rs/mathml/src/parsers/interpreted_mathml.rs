//! This module contains parsers that perform some amount of preliminary domain-specific
//! interpretation of presentation MathML (e.g., globbing S(t) to an identifier S of type
//! 'function'). This is in contrast to the `generic_mathml.rs` module that contains parsers that
//! do not attempt to perform any interpretation but instead simply preserve the original MathML
//! document structure.

use crate::{
    ast::{
        operator::{Derivative, Operator},
        Ci, Differential, Math, MathExpression, Mi, Mrow, Type,
    },
    parsers::generic_mathml::{
        add, attribute, dot, elem_many0, equals, etag, grad, lparen, mean, mi, mn, msqrt, msub,
        msubsup, multiply, rparen, stag, subtract, tag_parser, ws, xml_declaration, IResult,
        ParseError, Span,
    },
};

use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt, value},
    multi::{many0, many1, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, tuple},
};

/// Function to parse operators. This function differs from the one in parsers::generic_mathml by
/// disallowing operators besides +, -, =, (, ) and ,.
pub fn operator(input: Span) -> IResult<Operator> {
    let (s, op) = ws(delimited(
        stag!("mo"),
        alt((
            add, subtract, multiply, equals, lparen, rparen, mean, grad, dot,
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

/// Parse empty univariate function.
/// Example: S
fn empty_parenthesis(input: Span) -> IResult<Vec<Mi>> {
    let empty = vec![Mi("".to_string())];
    Ok((input, empty))
}

/// Parse content identifiers corresponding to univariate functions.
/// Example: S(t)
pub fn ci_univariate_with_bounds(input: Span) -> IResult<Ci> {
    let (s, (Mi(x), bound_vars)) = tuple((mi, parenthesized_identifier))(input)?;
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
    let (s, (x, bound_vars)) =
        tuple((mi, alt((parenthesized_identifier, empty_parenthesis))))(input)?;
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
    let (s, (x, bound_vars)) =
        tuple((msub, alt((parenthesized_identifier, empty_parenthesis))))(input)?;
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
        tag_parser!("mover", pair(math_expression, math_expression)),
        |(x, y)| MathExpression::Mover(Box::new(x), Box::new(y)),
    ))(input)?;
    Ok((s, over))
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

pub fn first_order_with_func_in_parenthesis(input: Span) -> IResult<(Derivative, Mrow)> {
    let (s, _) = tuple((stag!("mfrac"), alt((d, partial))))(input)?;
    let (s, with_respect_to) = delimited(
        tuple((stag!("mrow"), alt((d, partial)))),
        mi,
        pair(etag!("mrow"), etag!("mfrac")),
    )(s)?;

    let (s, _) = tuple((stag!("mo"), tag("("), etag!("mo")))(s)?;
    let (s, func) = many0(math_expression)(s)?;
    let (s, _) = tuple((stag!("mo"), tag(")"), etag!("mo")))(s)?;
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
        stag!("mover"),
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
        etag!("mover"),
    )(input)?;
    let (s, mi_func_of) = alt((parenthesized_identifier, empty_parenthesis))(s)?;
    let mut ci_func_of: Vec<Ci> = Vec::new();
    for bvar in mi_func_of {
        let b = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(bvar)), None);
        ci_func_of.push(b.clone());
    }
    let new_with_respect_to: Box<MathExpression> = Box::new(MathExpression::Ci(Ci {
        r#type: None,
        content: Box::new(MathExpression::Mi(Mi("".to_string()))),
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
pub fn absolute(input: Span) -> IResult<MathExpression>  {
    let (s, elements) = ws(delimited(
        tag("<mo>|</mo>"),
        many0(math_expression),
        tag("<mo>|</mo>"),
    ))(input)?;
   let components = MathExpression::Absolute(Box::new(MathExpression::Mo(Operator::Abs)), Box::new(MathExpression::Mrow(Mrow(elements))));

    Ok((s, components))
}

/// Example: Divergence
pub fn div(input: Span) -> IResult<Operator> {
    let (s, op) = ws(pair(gradient, ws(delimited(stag!("mo"), dot, etag!("mo")))))(input)?;
    let div = Operator::Div;
    Ok((s, div))
}

pub fn gradient(input: Span) -> IResult<Operator> {
    let (s, op) = ws(delimited(stag!("mo"), grad, etag!("mo")))(input)?;
    Ok((s, op))
}

pub fn grad_func(input: Span) -> IResult<(Operator, Ci)> {
    let (s, (op, id)) = ws(pair(gradient, mi))(input)?;
    let ci = Ci::new(Some(Type::Real), Box::new(MathExpression::Mi(id)), None);
    Ok((s, (op, ci)))
}

///Absolute with Msup value
pub fn absolute_with_msup(input: Span) -> IResult<MathExpression> {
    let (s, sup) = ws(map(
        ws(delimited(
            tag("<mo>|</mo>"),
            tuple((
                //math_expression,
                map(many0(math_expression), |z| Mrow(z)),
                preceded(tag("<msup><mo>|</mo>"), math_expression),
            )),
            tag("</msup>"),
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
                map(many0(math_expression), |z| Mrow(z)),
                preceded(tag("<msup><mo>)</mo>"), math_expression),
            )),
            tag("</msup>"),
        )),
        |(x, y)| MathExpression::Msup(Box::new(MathExpression::Mrow(x)), Box::new(y)),
    ))(input)?;
    Ok((s, sup))
}

/// Parser for math expressions. This varies from the one in the generic_mathml module, since it
/// assumes that expressions such as S(t) are actually univariate functions.
pub fn math_expression(input: Span) -> IResult<MathExpression> {
    ws(alt((
        map(div, MathExpression::Mo),
        absolute_with_msup,
        paren_as_msup,
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
        map(ci_univariate_without_bounds, MathExpression::Ci),
        map(ci_subscript, MathExpression::Ci),
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
        map(
            first_order_with_func_in_parenthesis,
            |(
                Derivative {
                    order,
                    var_index,
                    bound_var,
                },
                y,
            )| {
                MathExpression::Differential(Differential {
                    diff: Box::new(MathExpression::Mo(Operator::Derivative(Derivative {
                        order,
                        var_index,
                        bound_var,
                    }))),
                    func: Box::new(MathExpression::Mrow(y)),
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
        //map(gradient, MathExpression::Mo),
        absolute,
        map(operator, MathExpression::Mo),
        mn,
        superscript,
        over_term,
        msub,
        msqrt,
        mfrac,
        map(mrow, MathExpression::Mrow),
        msubsup,
    )))(input)
}

/// Parser for interpreted math expressions.
/// testing MathML documents
pub fn interpreted_math(input: Span) -> IResult<Math> {
    let (s, elements) = preceded(
        opt(xml_declaration),
        alt((
            ws(delimited(
                tag("<math>"),
                many0(math_expression),
                ws(tag("<mo>,</mo></math>")),
            )),
            ws(delimited(
                tag("<math>"),
                many0(math_expression),
                ws(tag("<mo>.</mo></math>")),
            )),
            elem_many0!("math"),
        )),
    )(input)?;
    Ok((s, Math { content: elements }))
}
