use actix_web::{put, web, HttpResponse};
use mathml::parsers::decapodes_serialization::{
    to_wiring_diagram, DecapodesCollection, WiringDiagram,
};
use mathml::parsers::first_order_ode::flatten_mults;
use mathml::parsers::generic_mathml::math;
use mathml::parsers::math_expression_tree::MathExpressionTree;
use mathml::parsers::math_expression_tree::{replace_unicode_with_symbols, preprocess_mathml_for_to_latex};
use mathml::{
    acset::{AMRmathml, PetriNet, RegNet},
    ast::Math,
    expression::{preprocess_content, wrap_math},
    parsers::first_order_ode::{first_order_ode, FirstOrderODE},
};
use petgraph::dot::{Config, Dot};
use utoipa;

/// Parse MathML and return a DOT representation of the abstract syntax tree (AST)
#[utoipa::path(
    request_body = String,
    responses(
        (
            status = 200,
            body = String
        )
    )
)]
#[put("/mathml/ast-graph")]
pub async fn get_ast_graph(payload: String) -> String {
    let contents = &payload;
    let math_results = math(contents.as_str().into());
    match math_results {
        Ok((_, math)) => {
            let g = math.to_graph();
            let dot_representation = Dot::with_config(&g, &[Config::EdgeNoLabel]);
            dot_representation.to_string()
        }
        Err(err) => err.to_string(),
    }
}

/// Parse a MathML representation of an equation and return a DOT representation of the math
/// expression graph (MEG), which can be used to perform structural alignment with the scientific
/// model code that corresponds to the equation.
#[utoipa::path(
    request_body = String,
    responses(
        (
            status = 200,
            body = String
        )
    )
)]
#[put("/mathml/math-exp-graph")]
pub async fn get_math_exp_graph(payload: String) -> String {
    let mut contents = payload;
    contents = preprocess_content(contents);
    let mut math = contents.parse::<Math>().unwrap();
    math.normalize();
    let new_math = wrap_math(math);
    let g = new_math.to_graph();
    let dot_representation = Dot::new(&g);
    dot_representation.to_string()
}

/// Parse a presentation MathML representation of an equation and
/// return the corresponding LaTeX representation
#[utoipa::path(
request_body = String,
responses(
(
status = 200,
body = String
)
)
)]
#[put("/mathml/latex")]
pub async fn get_latex(payload: String) -> String {
    let modified_input1 = &replace_unicode_with_symbols(&payload).to_string();
    let modified_input2 = &preprocess_mathml_for_to_latex(modified_input1).to_string();
    let exp = modified_input2.parse::<MathExpressionTree>().unwrap();
    exp.to_latex()
}

/// Parse presentation MathML and return a content MathML representation. Currently limited to
/// first-order ODEs.
#[utoipa::path(
    request_body = String,
    responses(
        (
            status = 200,
            body = String
        )
    )
)]
#[put("/mathml/content-mathml")]
pub async fn get_content_mathml(payload: String) -> String {
    let ode = payload.parse::<FirstOrderODE>().unwrap();
    ode.to_cmml()
}

/// Return a JSON representation of a DecapodeCollection, which should be the foundation of a DecapodeCollection AMR, from
/// an array of MathML strings.
#[utoipa::path(
    request_body = Vec<String>,
    responses(
        (
            status = 200,
            body = DecapodeCollection
        )
    )
)]
#[put("/mathml/decapodes")]
pub async fn get_decapodes(payload: web::Json<Vec<String>>) -> HttpResponse {
    let met_vec: Vec<MathExpressionTree> = payload
        .iter()
        .map(|x| x.parse::<MathExpressionTree>().unwrap())
        .collect();
    let mut deca_vec = Vec::<WiringDiagram>::new();
    for term in met_vec.iter() {
        deca_vec.push(to_wiring_diagram(term));
    }
    let decapodes_collection = DecapodesCollection {
        decapodes: deca_vec.clone(),
    };
    HttpResponse::Ok().json(web::Json(decapodes_collection))
}

/// Return a JSON representation of a PetriNet ModelRep constructed from an array of MathML strings.
#[utoipa::path(
    request_body = Vec<String>,
    responses(
        (
            status = 200,
            body = PetriNet
        )
    )
)]
#[put("/mathml/petrinet")]
pub async fn get_acset(payload: web::Json<Vec<String>>) -> HttpResponse {
    let asts_result: Result<Vec<_>, _> = payload
        .iter()
        .map(|x| first_order_ode(x.as_str().into()))
        .collect();

    match asts_result {
        Ok(asts) => {
            let mut flattened_asts = Vec::<FirstOrderODE>::new();
            for (_, mut eq) in asts {
                eq.rhs = flatten_mults(eq.rhs.clone());
                flattened_asts.push(eq.clone());
            }
            HttpResponse::Ok().json(web::Json(PetriNet::from(flattened_asts)))
        }
        Err(err) => HttpResponse::BadRequest()
            .content_type("text/plain")
            .body(err.to_string()),
    }
}

/// Return a JSON representation of a RegNet ModelRep constructed from an array of MathML strings.
#[utoipa::path(
    request_body = Vec<String>,
    responses(
        (
            status = 200,
            body = RegNet
        )
    )
)]
#[put("/mathml/regnet")]
pub async fn get_regnet(payload: web::Json<Vec<String>>) -> HttpResponse {
    let asts: Vec<Math> = payload.iter().map(|x| x.parse::<Math>().unwrap()).collect();
    HttpResponse::Ok().json(web::Json(RegNet::from(asts)))
}

/// Return a JSON representation of an AMR constructed from an array of MathML strings and a string
/// for the AMR subtype.
#[utoipa::path(
    request_body = AMRmathml,
    responses(
        (
            status = 200,
            body = EitherBody<PetriNet, RegNet>,
        )
    )
)]
#[put("/mathml/amr")]
pub async fn get_amr(payload: web::Json<AMRmathml>) -> HttpResponse {
    let mt_asts: Result<Vec<_>, _> = payload
        .mathml
        .iter()
        .map(|x| first_order_ode(x.as_str().into()))
        .collect();

    match mt_asts {
        Ok(mt_asts) => {
            let mut flattened_asts = Vec::<FirstOrderODE>::new();

            for (_, mut eq) in mt_asts {
                eq.rhs = flatten_mults(eq.rhs.clone());
                flattened_asts.push(eq.clone());
            }
            let asts: Vec<Math> = payload
                .mathml
                .clone()
                .iter()
                .map(|x| x.parse::<Math>().unwrap())
                .collect();
            let model_type = payload.model.clone();
            if model_type == *"regnet" {
                HttpResponse::Ok().json(web::Json(RegNet::from(asts)))
            } else if model_type == *"petrinet" {
                HttpResponse::Ok().json(web::Json(PetriNet::from(flattened_asts)))
            } else {
                HttpResponse::BadRequest()
                    .content_type("text/plain")
                    .body("Please specify a valid model.")
            }
        }
        Err(err) => HttpResponse::BadRequest()
            .content_type("text/plain")
            .body(err.to_string()),
    }
}
