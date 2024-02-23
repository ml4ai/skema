//! REST API endpoints related to CRUD operations and other queries on GroMEt objects.
use crate::config::Config;
use crate::database::{parse_gromet_queries, run_queries};
use crate::model_extraction::module_id2mathml_MET_ast;
use crate::ModuleCollection;
use actix_web::web::ServiceConfig;
use actix_web::{delete, get, post, put, web, HttpResponse};
use mathml::acset::{GeneralizedAMR, PetriNet, RegNet};

use mathml::ast::{self, MathExpression};
use mathml::parsers::math_expression_tree::MathExpressionTree;
use neo4rs;
use neo4rs::{query, Error, Node};
use std::collections::HashMap;
use std::sync::Arc;

use mathml::ast::operator::DerivativeNotation;
use utoipa;

pub fn configure() -> impl FnOnce(&mut ServiceConfig) {
    |config: &mut ServiceConfig| {
        config
            .service(post_model)
            .service(delete_model)
            .service(get_named_opos)
            .service(get_named_opis)
            .service(get_named_ports)
            .service(get_model_ids);
    }
}
#[allow(non_snake_case)]
pub async fn model_to_RN(gromet: ModuleCollection, config: Config) -> Result<RegNet, Error> {
    let module_id = push_model_to_db(gromet, config.clone()).await; // pushes model to db and gets id
    let ref_module_id1 = module_id.as_ref();
    let ref_module_id2 = module_id.as_ref();
    let mathml_ast = module_id2mathml_MET_ast(*ref_module_id1.unwrap(), config.clone()).await; // turns model into mathml ast equations
    let _del_response = delete_module(*ref_module_id2.unwrap(), config.clone()).await; // deletes model from db
    Ok(RegNet::from(mathml_ast))
}

// this is updated to mathexpressiontrees
#[allow(non_snake_case)]
pub async fn model_to_PN(gromet: ModuleCollection, config: Config) -> Result<PetriNet, Error> {
    let module_id = push_model_to_db(gromet, config.clone()).await; // pushes model to db and gets id
    let ref_module_id1 = module_id.as_ref();
    let ref_module_id2 = module_id.as_ref();
    let mathml_ast = module_id2mathml_MET_ast(*ref_module_id1.unwrap(), config.clone()).await; // turns model into mathml ast equations
    let _del_response = delete_module(*ref_module_id2.unwrap(), config.clone()).await; // deletes model from db
    Ok(PetriNet::from(mathml_ast))
}

pub async fn push_model_to_db(gromet: ModuleCollection, config: Config) -> Result<i64, Error> {
    // parse gromet into vec of queries
    let queries = parse_gromet_queries(gromet);

    // need to make the whole query list one line, individual executions are treated as different graphs for each execution.
    run_queries(queries, config.clone()).await?;
    let model_ids = module_query(config.clone()).await?;
    let last_model_id = model_ids[model_ids.len() - 1];
    Ok(last_model_id)
}

pub async fn delete_module(module_id: i64, config: Config) -> Result<(), Error> {
    // construct the query that will delete the module with a given unique identifier

    let query = format!(
        "MATCH (n)-[r:Contains|Port_Of|Wire|Metadata*1..7]->(m) WHERE id(n) = {}\nDETACH DELETE n,m",
        module_id
    );
    run_queries(vec![query], config.clone()).await?;
    Ok(())
}

pub async fn named_opi_query(module_id: i64, config: Config) -> Result<Vec<String>, Error> {
    // construct the query that will delete the module with a given unique identifier

    let mut port_names = Vec::<String>::new();

    // Connect to Memgraph.
    let graph = Arc::new(config.graphdb_connection().await);
    let mut result = graph
        .execute(
            query(
                "MATCH (n)-[r:Contains|Port_Of|Wire*1..7]->(m) WHERE id(n) = $id
        \nwith DISTINCT m\nmatch (m:Opi) where not m.name = 'un-named'\nreturn m",
            )
            .param("id", module_id),
        )
        .await?;
    while let Ok(Some(row)) = result.next().await {
        let node: Node = row.get("m").unwrap();
        let name: String = node.get("name").unwrap();
        port_names.push(name.clone());
    }

    Ok(port_names)
}

pub async fn named_opo_query(module_id: i64, config: Config) -> Result<Vec<String>, Error> {
    // construct the query that will delete the module with a given unique identifier

    let mut port_names = Vec::<String>::new();

    // Connect to Memgraph.
    let graph = Arc::new(config.graphdb_connection().await);
    let mut result = graph
        .execute(
            query(
                "MATCH (n)-[r:Contains|Port_Of|Wire*1..5]->(m) WHERE id(n) = $id
        \nwith DISTINCT m\nmatch (m:Opo) where not m.name = 'un-named'\nreturn m",
            )
            .param("id", module_id),
        )
        .await?;
    while let Ok(Some(row)) = result.next().await {
        let node: Node = row.get("m").unwrap();
        let name: String = node.get("name").unwrap();
        port_names.push(name.clone());
    }

    Ok(port_names)
}

pub async fn named_port_query(
    module_id: i64,
    config: Config,
) -> Result<HashMap<&'static str, Vec<String>>, Error> {
    let mut result = HashMap::<&str, Vec<String>>::new();
    let opis = named_opi_query(module_id, config.clone()).await;
    let opos = named_opo_query(module_id, config.clone()).await;
    result.insert("opis", opis.unwrap());
    result.insert("opos", opos.unwrap());
    Ok(result)
}

pub async fn module_query(config: Config) -> Result<Vec<i64>, Error> {
    let mut ids = Vec::<i64>::new();
    // Connect to Memgraph.

    let graph = Arc::new(config.graphdb_connection().await);
    let mut result = graph.execute(query("MATCH (n:Module) RETURN n")).await?;
    while let Ok(Some(row)) = result.next().await {
        let node: Node = row.get("n").unwrap();
        ids.push(node.id());
    }

    Ok(ids)
}

/// This retrieves the model IDs.
#[utoipa::path(
    responses(
        (
            status = 200
        )
    )
)]
#[get("/models")]
pub async fn get_model_ids(config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let response = module_query(config1.clone()).await.unwrap();
    HttpResponse::Ok().json(web::Json(response.clone()))
}

/// Pushes a gromet JSON to the Memgraph database
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (status = 200, description = "Model successfully pushed")
    )
)]
#[post("/models")]
pub async fn post_model(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let model_id = push_model_to_db(payload.into_inner(), config1)
        .await
        .unwrap();
    HttpResponse::Ok().json(web::Json(model_id))
}

/// Deletes a model from the database based on its id.
#[utoipa::path(
    responses(
        (status = 200, description = "Model deleted")
    )
)]
#[delete("/models/{id}")]
pub async fn delete_model(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let id = path.into_inner();
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    delete_module(id, config1).await.unwrap();
    HttpResponse::Ok().body("Model deleted")
}

/// This retrieves named Opos based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved named outports")
    )
)]
#[get("/models/{id}/named_opos")]
pub async fn get_named_opos(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let response = named_opo_query(path.into_inner(), config1).await.unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves named ports based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved named ports")
    )
)]
#[get("/models/{id}/named_ports")]
pub async fn get_named_ports(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let response = named_port_query(path.into_inner(), config1).await.unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves named Opis based on model id.
#[utoipa::path(
    responses(
        (status = 200, description = "Successfully retrieved named input ports")
    )
)]
#[get("/models/{id}/named_opis")]
pub async fn get_named_opis(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let response = named_opi_query(path.into_inner(), config1).await.unwrap();
    HttpResponse::Ok().json(web::Json(response))
}

/// This retrieves a RegNet AMR based on model id.
#[allow(non_snake_case)]
#[utoipa::path(
    responses(
        (
            status = 200, description = "Successfully retrieved RN AMR",
            body = RegNet
        )
    )
)]
#[get("/models/{id}/RN")]
pub async fn get_model_RN(path: web::Path<i64>, config: web::Data<Config>) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let mathml_ast = module_id2mathml_MET_ast(path.into_inner(), config1).await;
    HttpResponse::Ok().json(web::Json(RegNet::from(mathml_ast)))
}

/// This returns a PetriNet AMR from a gromet.
#[allow(non_snake_case)]
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (
            status = 200, description = "Successfully retrieved PN AMR",
            body = ModelPetriNet
        )
    )
)]
#[put("/models/PN")]
pub async fn model2PN(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    HttpResponse::Ok().json(web::Json(
        model_to_PN(payload.into_inner(), config1).await.unwrap(),
    ))
}

/// This returns a RegNet AMR from a gromet.
#[allow(non_snake_case)]
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (
            status = 200, description = "Successfully retrieved RN AMR",
            body = ModelRegNet
        )
    )
)]
#[put("/models/RN")]
pub async fn model2RN(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    HttpResponse::Ok().json(web::Json(
        model_to_RN(payload.into_inner(), config1).await.unwrap(),
    ))
}

/// This returns a MET vector from a gromet.
#[allow(non_snake_case)]
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (
            status = 200, description = "Successfully retrieved MET"
        )
    )
)]
#[put("/models/MET")]
pub async fn model2MET(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let module_id = push_model_to_db(payload.into_inner(), config1.clone()).await; // pushes model to db and gets id
    let ref_module_id1 = module_id.as_ref();
    let ref_module_id2 = module_id.as_ref();
    let mathml_ast = module_id2mathml_MET_ast(*ref_module_id1.unwrap(), config1.clone()).await; // turns model into mathml ast equations
    let _del_response = delete_module(*ref_module_id2.unwrap(), config1.clone()).await; // deletes model from db
                                                                                        // now we convert each firstorderode into a MET. The RHS is just ported over, but we need to create the LHS as a derivative and put it into an equation
    let mut mets = Vec::<MathExpressionTree>::new();
    for equation in mathml_ast.iter() {
        let mut equal_args = Vec::<MathExpressionTree>::new();
        let lhs_mi1 = mathml::ast::Mi("".to_string()); // blank
        let lhs_mi2 = mathml::ast::Mi("t".to_string()); // differentiation variable
        let lhs_mi3 = mathml::ast::Mi(equation.lhs_var.to_string()); // state function
        let lhs_ci1 = mathml::ast::Ci {
            // blank
            r#type: Some(ast::Type::Real),
            content: Box::new(mathml::ast::MathExpression::Mi(lhs_mi1)),
            func_of: None,
        };
        let lhs_ci2 = mathml::ast::Ci {
            // differentiation variable
            r#type: Some(ast::Type::Real),
            content: Box::new(mathml::ast::MathExpression::Mi(lhs_mi2)),
            func_of: None,
        };
        let lhs_ci3 = mathml::ast::Ci {
            // state function
            r#type: Some(ast::Type::Function),
            content: Box::new(mathml::ast::MathExpression::Mi(lhs_mi3)),
            func_of: Some([lhs_ci1].to_vec()),
        };
        let lhs_deriv = mathml::ast::operator::Derivative {
            order: 1,
            var_index: 1,
            bound_var: lhs_ci2,
            notation: DerivativeNotation::LeibnizTotal,
        };
        let lhs = MathExpressionTree::Cons(
            mathml::ast::operator::Operator::Derivative(lhs_deriv),
            [MathExpressionTree::Atom(MathExpression::Ci(lhs_ci3))].to_vec(),
        );
        equal_args.push(lhs.clone());
        equal_args.push(equation.rhs.clone());
        let met =
            MathExpressionTree::Cons(mathml::ast::operator::Operator::Equals, equal_args.clone());
        mets.push(met.clone());
    }
    HttpResponse::Ok().json(web::Json(mets))
}

/// This returns a Generalized AMR from a gromet.
#[allow(non_snake_case)]
#[utoipa::path(
    request_body = ModuleCollection,
    responses(
        (
            status = 200, description = "Successfully retrieved MET",
            body = GeneralizedAMR
        )
    )
)]
#[put("/models/G-AMR")]
pub async fn model2GAMR(
    payload: web::Json<ModuleCollection>,
    config: web::Data<Config>,
) -> HttpResponse {
    let config1 = Config {
        db_host: config.db_host.clone(),
        db_port: config.db_port,
        db_protocol: config.db_protocol.clone(),
    };
    let module_id = push_model_to_db(payload.into_inner(), config1.clone()).await; // pushes model to db and gets id
    let ref_module_id1 = module_id.as_ref();
    let ref_module_id2 = module_id.as_ref();
    let mathml_ast = module_id2mathml_MET_ast(*ref_module_id1.unwrap(), config1.clone()).await; // turns model into mathml ast equations
    let _del_response = delete_module(*ref_module_id2.unwrap(), config1.clone()).await; // deletes model from db
                                                                                        // now we convert each firstorderode into a MET. The RHS is just ported over, but we need to create the LHS as a derivative and put it into an equation
    let mut mets = Vec::<MathExpressionTree>::new();
    for equation in mathml_ast.iter() {
        let mut equal_args = Vec::<MathExpressionTree>::new();
        let lhs_mi1 = mathml::ast::Mi("".to_string()); // blank
        let lhs_mi2 = mathml::ast::Mi("t".to_string()); // differentiation variable
        let lhs_mi3 = mathml::ast::Mi(equation.lhs_var.to_string()); // state function
        let lhs_ci1 = mathml::ast::Ci {
            // blank
            r#type: Some(ast::Type::Real),
            content: Box::new(mathml::ast::MathExpression::Mi(lhs_mi1)),
            func_of: None,
        };
        let lhs_ci2 = mathml::ast::Ci {
            // differentiation variable
            r#type: Some(ast::Type::Real),
            content: Box::new(mathml::ast::MathExpression::Mi(lhs_mi2)),
            func_of: None,
        };
        let lhs_ci3 = mathml::ast::Ci {
            // state function
            r#type: Some(ast::Type::Function),
            content: Box::new(mathml::ast::MathExpression::Mi(lhs_mi3)),
            func_of: Some([lhs_ci1].to_vec()),
        };
        let lhs_deriv = mathml::ast::operator::Derivative {
            order: 1,
            var_index: 1,
            bound_var: lhs_ci2,
            notation: DerivativeNotation::LeibnizTotal,
        };
        let lhs = MathExpressionTree::Cons(
            mathml::ast::operator::Operator::Derivative(lhs_deriv),
            [MathExpressionTree::Atom(MathExpression::Ci(lhs_ci3))].to_vec(),
        );
        equal_args.push(lhs.clone());
        equal_args.push(equation.rhs.clone());
        let met =
            MathExpressionTree::Cons(mathml::ast::operator::Operator::Equals, equal_args.clone());
        mets.push(met.clone());
    }
    HttpResponse::Ok().json(web::Json(GeneralizedAMR::from(mets)))
}
