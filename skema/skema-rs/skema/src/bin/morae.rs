use mathml::acset::{InputArc, OutputArc, PN_to_ModelRepPN, Specie, Transition};
use mathml::ast::{Math, Operator};
use mathml::expression::wrap_math;
use mathml::expression::Atom;
use mathml::expression::{Expr, PreExp};
use mathml::mml2pn::get_mathml_asts_from_file;
pub use mathml::mml2pn::{ACSet, Term};
use petgraph::dot::{Config, Dot};
use petgraph::matrix_graph::IndexType;
use petgraph::prelude::*;
use petgraph::*;
use rsmgclient::{ConnectParams, Connection, MgError, Node, Relationship, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;

// new imports
use mathml::ast::MathExpression;
use mathml::ast::MathExpression::Mo;
use mathml::ast::MathExpression::Mrow;
use mathml::expression::preprocess_content;
use mathml::expression::Expression;
use mathml::parsing::parse;
use mathml::petri_net::recognizers::get_polarity;
use mathml::petri_net::recognizers::get_specie_var;
use mathml::petri_net::recognizers::is_add_or_subtract_operator;
use mathml::petri_net::recognizers::is_var_candidate;
use mathml::petri_net::{Polarity, Var};

// just for making schema's
use schemars::schema_for; // for printing
use serde_json; // need for printing
use skema::Metadata; // struct to make schema from

// overall this will be a function that takes in a module id and weather it is manually or auto extraction to use

// for now we will deal with manual module ids

fn main() {
    // setup command line argument on if the core dynamics has been found manually or needs to be found automatically
    /*
    Command line args;
        - auto -> This will attempt an automated search for the dynamics
        - manual -> This assumes the input is the function of the dynamics
    */
    let args: Vec<String> = env::args().collect();

    let mut module_id = 1755;
    // now to prototype an algorithm to find the function that contains the core dynamics

    if args[1] == "auto".to_string() {
        if args.clone().len() > 2 {
            module_id = args[2].parse::<i64>().unwrap();
        }

        let graph = subgraph2petgraph(module_id); // makes petgraph of graph

        let core_id = find_pn_dynamics(module_id); // gives back list of function nodes that might contain the dynamics

        let line_span = get_line_span(core_id[0].clone(), graph.clone()); // get's the line span of function id

        println!("\n{:?}", line_span);

        println!("function_core_id: {:?}", core_id[0].clone());
        println!("module_id: {:?}\n", module_id.clone());
        // 4.5 now to check if of those expressions, if they are arithmetric in nature

        // 5. pass id to subgrapg2_core_dyn to get core dynamics
        let (mut core_dynamics_ast, metadata_map_ast) =
            subgraph2_core_dyn_ast(core_id[0].clone()).unwrap();

        println!("\n{:?}", core_dynamics_ast[2].clone());

        println!("Testing of convseriton to new data rep");

        let mathml_ast =
            get_mathml_asts_from_file("../../../data/mml2pn_inputs/simple_sir_v1/mml_list.txt");
        let test_pn = ACSet::from(mathml_ast);

        println!("Pertri-Net rep: {:?}", test_pn.clone());

        let new_model = PN_to_ModelRepPN(test_pn.clone());

        println!("New Model Rep: {:?}", new_model.clone());

        let mod_serialized = serde_json::to_string(&new_model).unwrap();

        println!("Serialized Model Rep: {}", mod_serialized.clone());

        /*let (core_dynamics, metadata_map) = subgraph2_core_dyn_exp(core_id).unwrap();

        println!("{:?}", core_dynamics[0].clone());*/
    }
    // This is the graph id for the top level function for the core dynamics for our test case.
    else if args[1] == "manual".to_string() {
        // still need to grab the module id

        let (mut core_dynamics, metadata_map) = subgraph2_core_dyn_exp(module_id.clone()).unwrap();

        let (mut core_dynamics_ast, metadata_map_ast) =
            subgraph2_core_dyn_ast(module_id.clone()).unwrap();

        let mut named_core_dynamics = Vec::<PreExp>::new();

        for (i, expression) in core_dynamics.clone().iter().enumerate() {
            let test_pre_exp = match &expression {
                Expr::Expression { ops, args, name } => PreExp {
                    ops: (*ops.clone()).to_vec(),
                    args: (*args.clone()).to_vec(),
                    name: expression.clone().set_name(),
                },
                &Expr::Atom(_) => PreExp {
                    ops: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                },
            };
            named_core_dynamics.push(test_pre_exp.clone());
        }

        // expressions that are working: 5 (T), 6 (H)
        // 7 (E) should be fixable, the USub is not being subsituted and is at the end of the RHS instead of the start

        println!("Exp:\n {:?}", named_core_dynamics[5].clone());
        println!("\n Ast:\n {:?}", core_dynamics_ast[5].clone());
    }
    // This is for testing the converter and it will import from the test files since we should be able to make sure things are
    // parsing correctly
    else if args[1] == "test".to_string() {
        // load up the parsed file
        let mathml_ast =
            get_mathml_asts_from_file("../../../data/mml2pn_inputs/simple_sir_v1/mml_list.txt");

        // make my own converter to PreExp with an '=' as the anchor as I don't understand Liang's converter
        let mut core_dynamics_exp = ast2exp(mathml_ast.clone());

        // convert the parsed file into pre_exp
        let mut named_core_dynamics = Vec::<PreExp>::new();
        for equation in mathml_ast.iter() {
            println!("\nast: {:?}", equation.content[0].clone());
            // this next bit is for Liang's converter to expressions
            let mut pre_exp = PreExp {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: "".to_string(),
            };
            let new_math = wrap_math(equation.clone());
            new_math.clone().to_expr(&mut pre_exp);
            pre_exp.group_expr();
            pre_exp.collapse_expr();
            //pre_exp.to_graph();
            pre_exp.set_name();
            // pre_exp.distribute_expr();
            println!("\nLiang exp: {:?}", pre_exp.clone());
        }

        let acset_dyn = exp2pn(named_core_dynamics);

        // printing the ACSet for future comparison
        println!("\nACSet: {:?}", ACSet::from(mathml_ast));
        println!("\n exp ACSet: {:?}", acset_dyn);

        // printing json schema's for Ben
        /*let schema = schema_for!(Metadata);
        println!("{}", serde_json::to_string_pretty(&schema).unwrap());*/
    } else if args[1] == "test_liang".to_string() {
        let input = "../../../skema/skema-rs/mathml/tests/seir_eq1.xml";
        let mut contents = std::fs::read_to_string(input)
            .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
        contents = preprocess_content(contents);
        let (_, mut math) =
            parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
        math.normalize();
        let new_math = wrap_math(math);
        let mut pre_exp = PreExp {
            ops: Vec::<Operator>::new(),
            args: Vec::<Expr>::new(),
            name: "".to_string(),
        };
        new_math.clone().to_expr(&mut pre_exp);
        pre_exp.group_expr();
        pre_exp.collapse_expr();
        pre_exp.group_expr();
        pre_exp.collapse_expr();
        pre_exp.set_name();
        println!("\nLiang exp: {:?}", pre_exp.clone());
    } else {
        println!("Unknown Command!");
    }

    // FOR DEBUGGING
    /*for node_idx in expressions_wiring[1].clone().node_indices() {
        if expressions_wiring[1].clone()[node_idx].properties["name"].to_string()
            == "'un-named'".to_string()
        {
            println!("{:?}", node_idx);
        }
    }*/
    /*for i in 0..trimmed_expressions_wiring.len() {
        println!("{:?}", graph[expression_nodes[i]].id);
        println!(
            "Nodes in wiring subgraph: {}",
            trimmed_expressions_wiring[i].node_count()
        );
        println!(
            "Edges in wiring subgraph: {}",
            trimmed_expressions_wiring[i].edge_count()
        );
    }*/
    //let (nodes1, edges1) = expressions_wiring[1].clone().into_nodes_edges();
    /*for edge in edges1 {
        let source = edge.source();
        let target = edge.target();
        let edge_weight = edge.weight;
        println!(
            "Edge from {:?} to {:?} with weight {:?}",
            source, target, edge_weight
        );
    }*/
    /*for node in nodes1 {
        println!("{:?}", node);
    }*/
    /*println!(
        "{:?}",
        Dot::with_config(&expressions_wiring[1], &[Config::EdgeNoLabel])
    );*/
}

// struct for returning line spans
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize)]
pub struct LineSpan {
    line_begin: i64,
    line_end: i64,
}

// this function returns the line numbers of the function node id provided
pub fn get_line_span(
    node_id: i64,
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
) -> LineSpan {
    // extract line numbers of function which contains the dynamics
    let mut line_nums = Vec::<i64>::new();
    for node in graph.node_indices() {
        if graph[node].id == node_id.clone() {
            for n_node in graph.neighbors_directed(node.clone(), Outgoing) {
                if graph[n_node.clone()].labels == ["Metadata"] {
                    match &graph[n_node].clone().properties["line_begin"] {
                        Value::List(x) => match x[0] {
                            Value::Int(y) => {
                                //println!("line_begin: {:?}", y);
                                line_nums.push(y.clone());
                            }
                            _ => println!("error metadata type"),
                        },
                        _ => println!("error metadata type"),
                    }
                    match &graph[n_node].clone().properties["line_end"] {
                        Value::List(x) => match x[0] {
                            Value::Int(y) => {
                                //println!("line_end: {:?}", y);
                                line_nums.push(y.clone());
                            }
                            _ => println!("error metadata type"),
                        },
                        _ => println!("error metadata type"),
                    }
                }
            }
        }
    }
    let line_span = LineSpan {
        line_begin: line_nums[0].clone(),
        line_end: line_nums[1].clone(),
    };

    return line_span;
}

// this function finds the core dynamics and returns a vector of
// node id's that meet the criteria for identification
pub fn find_pn_dynamics(module_id: i64) -> Vec<i64> {
    let graph = subgraph2petgraph(module_id);
    // 1. find each function node
    let mut function_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Function"] {
            function_nodes.push(node.clone());
        }
    }
    // 2. check and make sure only expressions in function
    // 3. check number of expressions and decide off that
    let mut functions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..function_nodes.len() {
        // grab the subgraph of the given expression
        functions.push(subgraph2petgraph(graph[function_nodes[i]].id.clone()));
    }
    // get a sense of the number of expressions in each function
    let mut func_counter = 0;
    let mut core_func = Vec::<usize>::new();
    for func in functions.clone() {
        let mut expression_counter = 0;
        let mut primitive_counter = 0;
        for node in func.node_indices() {
            if func[node].labels == ["Expression"] {
                expression_counter += 1;
            }
            if func[node].labels == ["Primitive"] {
                if func[node].properties["name"].to_string() == "'*'".to_string() {
                    primitive_counter += 1;
                } else if func[node].properties["name"].to_string() == "'+'".to_string() {
                    primitive_counter += 1;
                } else if func[node].properties["name"].to_string() == "'-'".to_string() {
                    primitive_counter += 1;
                } else if func[node].properties["name"].to_string() == "'USub'".to_string() {
                    primitive_counter += 1;
                }
            }
        }
        if expression_counter >= 3 && primitive_counter >= 12 {
            core_func.push(func_counter);
        }
        func_counter += 1;
    }
    // 4. get the id of functions with enough expressions
    let mut core_id = Vec::<i64>::new();
    for c_func in core_func.iter() {
        for node in functions[c_func.clone()].clone().node_indices() {
            if functions[c_func.clone()][node].labels == ["Function"] {
                core_id.push(functions[c_func.clone()][node].id.clone());
            }
        }
    }

    return core_id;
}

// This will parse the mathml based on a '=' anchor
fn ast2exp(mathml_ast: Vec<Math>) -> Vec<PreExp> {
    let mut named_core_dynamics = Vec::<PreExp>::new();

    // This is a lot of adarsh's code on somehow iteratoring over an enum
    let mut equals_index = 0;
    for equation in mathml_ast {
        let mut pre_exp = PreExp {
            ops: Vec::<Operator>::new(),
            args: Vec::<Expr>::new(),
            name: "".to_string(),
        };
        if let Mrow(expr_1) = &equation.content[0] {
            // Get the index of the equals term
            for (i, expr_2) in (*expr_1).iter().enumerate() {
                if let Mo(Operator::Equals) = expr_2 {
                    equals_index = i;
                    println!("{:?}", expr_1.clone());
                    let lhs = &expr_1[0]; // does this not imply equals_index is always 1?
                    println!("lhs: {:?}", lhs);
                    pre_exp.ops.push(Operator::Other("".to_string()));
                    pre_exp.ops.push(Operator::Equals);
                    let mut pre_exp_args = Vec::<Expression>::new();
                    let lhs_string = String::new();
                    let lhs_expr = Expression {
                        ops: [Operator::Other("Derivative".to_string())].to_vec(),
                        args: [Expr::Atom(Atom::Identifier("S".to_string()))].to_vec(),
                        name: "".to_string(),
                    };
                    let mut rhs_math = expr_1[2..].to_vec();
                    pre_exp_args.push(lhs_expr.clone());

                    pre_exp.args.push(Expr::Expression {
                        ops: pre_exp_args[0].ops.clone(),
                        args: pre_exp_args[0].args.clone(),
                        name: "".to_string(),
                    });
                    println!("\nContrstucted pre_exp: {:?}", pre_exp.clone());
                }
            }
        }
    }

    return named_core_dynamics;
}

fn exp2pn(named_core_dynamics: Vec<PreExp>) -> ACSet {
    let ascet = ACSet {
        S: Vec::<Specie>::new(),
        T: Vec::<Transition>::new(),
        I: Vec::<InputArc>::new(),
        O: Vec::<OutputArc>::new(),
    };

    return ascet;
}

fn subgraph2_core_dyn_ast(
    root_node_id: i64,
) -> Result<(Vec<Vec<MathExpression>>, HashMap<String, rsmgclient::Node>), MgError> {
    // get the petgraph of the subgraph
    let graph = subgraph2petgraph(root_node_id);

    /* MAKE THIS A FUNCTION THAT TAKES IN A PETGRAPH */
    // create the metadata rust rep
    // this will be a map of the name of the node and the metadata node it's attached to with the mapping to our standard metadata struct
    // grab metadata nodes
    let mut metadata_map = HashMap::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Metadata"] {
            for neighbor_node in graph.neighbors_directed(node, Incoming) {
                // NOTE: these names have slightly off formating, the key is: "'name'"
                metadata_map.insert(
                    graph[neighbor_node].properties["name"].to_string().clone(),
                    graph[node].clone(),
                );
            }
        }
    }

    // find all the expressions
    let mut expression_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Expression"] {
            expression_nodes.push(node);
        }
    }

    // initialize vector to collect all expressions
    let mut expressions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the subgraph of the given expression
        expressions.push(subgraph2petgraph(graph[expression_nodes[i]].id.clone()));
    }

    // initialize vector to collect all expression wiring graphs
    let mut expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        expressions_wiring.push(subgraph_wiring(graph[expression_nodes[i]].id.clone()).unwrap());
    }

    // now to trim off the un-named filler nodes and filler expressions
    let mut trimmed_expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expressions_wiring.clone().len() {
        let (nodes1, _edges1) = expressions_wiring[i].clone().into_nodes_edges();
        if nodes1.len() > 3 {
            trimmed_expressions_wiring.push(trim_un_named(expressions_wiring[i].clone()));
        }
    }

    // now we convert the following into a Expr to get converted into a petri net
    // first we have to get the parent node index to pass into the function
    /*let mut root_node = Vec::<NodeIndex>::new();
    for node_index in trimmed_expressions_wiring[0].clone().node_indices() {
        if trimmed_expressions_wiring[0].clone()[node_index].labels == ["Opo"] {
            root_node.push(node_index);
        }
    }
    if root_node.len() >= 2 {
        panic!("More than one Opo!");
    }*/

    // this is the actual convertion
    let mut core_dynamics = Vec::<Vec<MathExpression>>::new();

    for expr in trimmed_expressions_wiring.clone() {
        let mut root_node = Vec::<NodeIndex>::new();
        for node_index in expr.clone().node_indices() {
            if expr.clone()[node_index].labels == ["Opo"] {
                root_node.push(node_index);
            }
        }
        if root_node.len() >= 2 {
            println!("More than one Opo! Skipping Expression!");
        } else {
            core_dynamics.push(tree_2_ast(expr.clone(), root_node[0].clone()).unwrap());
        }
    }

    return Ok((core_dynamics, metadata_map));
}

fn subgraph2_core_dyn_exp(
    root_node_id: i64,
) -> Result<(Vec<Expr>, HashMap<String, rsmgclient::Node>), MgError> {
    // get the petgraph of the subgraph
    let graph = subgraph2petgraph(root_node_id);

    /* MAKE THIS A FUNCTION THAT TAKES IN A PETGRAPH */
    // create the metadata rust rep
    // this will be a map of the name of the node and the metadata node it's attached to with the mapping to our standard metadata struct
    // grab metadata nodes
    let mut metadata_map = HashMap::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Metadata"] {
            for neighbor_node in graph.neighbors_directed(node, Incoming) {
                // NOTE: these names have slightly off formating, the key is: "'name'"
                metadata_map.insert(
                    graph[neighbor_node].properties["name"].to_string().clone(),
                    graph[node].clone(),
                );
            }
        }
    }

    // find all the expressions
    let mut expression_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Expression"] {
            expression_nodes.push(node);
        }
    }

    // initialize vector to collect all expressions
    let mut expressions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the subgraph of the given expression
        expressions.push(subgraph2petgraph(graph[expression_nodes[i]].id.clone()));
    }

    // initialize vector to collect all expression wiring graphs
    let mut expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        expressions_wiring.push(subgraph_wiring(graph[expression_nodes[i]].id.clone()).unwrap());
    }

    // now to trim off the un-named filler nodes and filler expressions
    let mut trimmed_expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expressions_wiring.clone().len() {
        let (nodes1, _edges1) = expressions_wiring[i].clone().into_nodes_edges();
        if nodes1.len() > 3 {
            trimmed_expressions_wiring.push(trim_un_named(expressions_wiring[i].clone()));
        }
    }

    // now we convert the following into a Expr to get converted into a petri net
    // first we have to get the parent node index to pass into the function
    let mut root_node = Vec::<NodeIndex>::new();
    for node_index in trimmed_expressions_wiring[0].clone().node_indices() {
        if trimmed_expressions_wiring[0].clone()[node_index].labels == ["Opo"] {
            root_node.push(node_index);
        }
    }
    if root_node.len() >= 2 {
        panic!("More than one Opo!");
    }

    // this is the actual convertion
    let mut core_dynamics = Vec::<Expr>::new();

    for expr in trimmed_expressions_wiring.clone() {
        let mut root_node = Vec::<NodeIndex>::new();
        for node_index in expr.clone().node_indices() {
            if expr.clone()[node_index].labels == ["Opo"] {
                root_node.push(node_index);
            }
        }
        if root_node.len() >= 2 {
            panic!("More than one Opo!");
        }

        core_dynamics.push(tree_2_expr(expr.clone(), root_node[0].clone()).unwrap());
    }

    return Ok((core_dynamics, metadata_map));
}

fn tree_2_ast(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Result<Vec<MathExpression>, MgError> {
    let mut math_vec = Vec::<MathExpression>::new();

    if graph.clone()[root_node].labels == ["Opo"] {
        // we first construct the derivative of the first node
        let mut deriv_name: &str = &graph.clone()[root_node].properties["name"].to_string();
        // this will let us know if additional trimming is needed to handle the code implementation of the equations
        let mut step_impl = false;
        // This is very bespoke right now
        // this check is for if it's leibniz notation or not, will need to expand as more cases are creating,
        // currently we convert to leibniz form
        if deriv_name[1..2].to_string().clone() == "d" {
            let deriv = MathExpression::Mfrac(
                Box::new(Mrow(
                    [
                        MathExpression::Mi(deriv_name[1..2].to_string().clone()),
                        MathExpression::Mi(deriv_name[2..3].to_string().clone()),
                    ]
                    .to_vec(),
                )),
                Box::new(Mrow(
                    [
                        MathExpression::Mi(deriv_name[3..4].to_string().clone()),
                        MathExpression::Mi(deriv_name[4..5].to_string().clone()),
                    ]
                    .to_vec(),
                )),
            );
            math_vec.push(deriv.clone());
        } else {
            step_impl = true;
            let deriv = MathExpression::Mfrac(
                Box::new(Mrow(
                    [
                        MathExpression::Mi("d".to_string().clone()),
                        MathExpression::Mi(deriv_name[1..2].to_string().clone()),
                    ]
                    .to_vec(),
                )),
                Box::new(Mrow(
                    [
                        MathExpression::Mi("d".to_string().clone()),
                        MathExpression::Mi("t".to_string().clone()),
                    ]
                    .to_vec(),
                )),
            );
            math_vec.push(deriv.clone());
        }
        // we also push an Mo('=') here before traversing the tree to parse the rhs
        math_vec.push(MathExpression::Mo(Operator::Equals));
        // now we walk through the tree to parse the rest
        let mut rhs_eq = Vec::<MathExpression>::new();
        let mut first_op = Vec::<MathExpression>::new();

        for node in graph.neighbors_directed(root_node, Outgoing) {
            if graph.clone()[node].labels == ["Primitive"]
                && !(graph.clone()[node].properties["name"].to_string() == "'USub'".to_string())
            {
                first_op.push(get_operator(graph.clone(), node.clone()));
                let mut arg1 = get_args(graph.clone(), node.clone());
                if graph.clone()[node].properties["name"].to_string() == "'*'".to_string() {
                    let arg1_mult = is_multiple_terms(arg1[0].clone());
                    let arg2_mult = is_multiple_terms(arg1[1].clone());
                    if arg1_mult {
                        let arg_dist = distribute_args(arg1[1].clone(), arg1[0].clone());
                        math_vec.extend_from_slice(&arg_dist.clone());
                    } else if arg2_mult {
                        let arg_dist = distribute_args(arg1[0].clone(), arg1[1].clone());
                        math_vec.extend_from_slice(&arg_dist.clone());
                    } else {
                        math_vec.extend_from_slice(&arg1[0].clone());
                        math_vec.push(first_op[0].clone());
                        math_vec.extend_from_slice(&arg1[1].clone());
                    }
                } else {
                    math_vec.extend_from_slice(&arg1[0].clone());
                    math_vec.push(first_op[0].clone());
                    math_vec.extend_from_slice(&arg1[1].clone());
                }
            } else {
                println!("Not supported or Trivial case");
            }
        }

        // we now need to handle the case where it's step implementation
        // we find the Mi of the state variable that doesn't have a multiplication next to it (including only one, if at the end of the vec)
        // we then remove it and the one of the addition operators next to it
        if step_impl {
            let ref_name = deriv_name[1..2].to_string().clone();
            for (idx, obj) in math_vec.clone().iter().enumerate() {
                if *obj == MathExpression::Mi(ref_name.clone()) {
                    // find the index of the extra state variable
                    println!("found idx: {:?}", idx.clone());
                    println!("obj: {:?}", obj.clone());
                }
            }
        }
    } else {
        println!("Error! Starting node is not Opo!");
    }

    println!("Not reversed mathml: {:?}", math_vec.clone());

    let mut reversed_final_math = Vec::<MathExpression>::new();
    let vec_len_temp = math_vec.clone().len();

    reversed_final_math.extend_from_slice(&math_vec.clone()[0..2]);

    for (i, j) in math_vec.clone().iter().rev().enumerate() {
        if i != vec_len_temp && i != (vec_len_temp - 1) && i != (vec_len_temp - 2) {
            reversed_final_math.push(j.clone());
        }
    }

    return Ok(reversed_final_math);
}

// this function returns a bool of if an expression has multiple terms (conjuction of +/-'s)
fn is_multiple_terms(arg: Vec<MathExpression>) -> bool {
    let mut add_sub_index = 0;

    for (i, expression) in arg.clone().iter().enumerate() {
        if is_add_or_subtract_operator(expression) {
            add_sub_index = i.clone();
        }
    }
    if add_sub_index.clone() != 0 {
        return true;
    } else {
        return false;
    }
}

// This function returns a vector of indicies of where the operators that seperate terms in an expression are
fn terms_indicies(arg: Vec<MathExpression>) -> Vec<i32> {
    let mut add_sub_index_vec = Vec::<i32>::new();

    for (i, expression) in arg.clone().iter().enumerate() {
        if is_add_or_subtract_operator(expression) {
            add_sub_index_vec.push(i.clone().try_into().unwrap());
        }
    }
    return add_sub_index_vec;
}

// the function takes two arguments and distributes them, made so the first argument is distributed over the second
fn distribute_args(
    arg1: Vec<MathExpression>,
    mut arg2: Vec<MathExpression>,
) -> Vec<MathExpression> {
    let mut arg_dist = Vec::<MathExpression>::new();
    let mut arg2_term_ind = Vec::<i32>::new();

    arg2_term_ind = terms_indicies(arg2.clone());

    // check if need to swap operator signs
    if arg1[0] == Mo(Operator::Other("USub".to_string())) {
        println!("USub dist happens"); // This is never running
                                       // operator starts at begining of arg2
        if arg2_term_ind[0] == 0 {
            for (i, ind) in arg2_term_ind.clone().iter().enumerate() {
                if arg2[(*ind as usize)].clone() == Mo(Operator::Add) {
                    arg2[(*ind as usize)] = Mo(Operator::Subtract);
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = (arg1.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = (arg1.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                } else {
                    arg2[(*ind as usize)] = Mo(Operator::Add);
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = (arg1.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    }
                }
            }
        } else {
            // operator doesn't start at beginning so have to add it manually
            arg_dist.push(Mo(Operator::Subtract));
            arg_dist.extend_from_slice(&arg2.clone()[0..(arg2_term_ind.clone()[0] - 1) as usize]);
            //arg_dist.push(Mo(Operator::Multiply));
            let vec_len1 = (arg1.clone().len() - 1) as usize;
            arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
            for (i, ind) in arg2_term_ind.clone().iter().enumerate() {
                if arg2[(*ind as usize)].clone() == Mo(Operator::Add) {
                    arg2[(*ind as usize)] = Mo(Operator::Subtract);
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = (arg1.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                } else {
                    arg2[(*ind as usize)] = Mo(Operator::Add);
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = (arg1.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                }
            }
        }
    } else {
        // don't have to swap operators
        if arg2_term_ind[0] == 0 {
            for (i, ind) in arg2_term_ind.clone().iter().enumerate() {
                if arg2[(*ind as usize)].clone() == Mo(Operator::Add) {
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                } else {
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                }
            }
        } else {
            // don't swap operators manual beginning push
            arg_dist.extend_from_slice(&arg2.clone()[0..(arg2_term_ind.clone()[0] - 1) as usize]);
            //arg_dist.push(Mo(Operator::Multiply));
            let vec_len1 = (arg1.clone().len() - 1) as usize;
            arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
            for (i, ind) in arg2_term_ind.clone().iter().enumerate() {
                if arg2[*ind as usize].clone() == Mo(Operator::Add) {
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                } else {
                    if (i + 1) != arg2_term_ind.clone().len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = (arg2.clone().len() - 1) as usize;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                }
            }
        }
    }
    return arg_dist;
}

// this will get the arguments for an operator/primitive of the graph, assumes it's binary
fn get_args(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Vec<Vec<MathExpression>> {
    let mut op = Vec::<MathExpression>::new();
    // need to construct the vector of length 2 with temporary filling since we might get the second
    // element first and tuples can't be dynamically indexed in rust
    let temp_op = Vec::<MathExpression>::new();
    let mut args = vec![temp_op.clone(); 2];

    for (i, node) in graph.neighbors_directed(root_node, Outgoing).enumerate() {
        if graph.clone()[node].labels == ["Primitive"]
            && graph.clone()[node].properties["name"].to_string() == "'USub'".to_string()
        {
            op.push(get_operator(graph.clone(), node.clone()));
            for node1 in graph.neighbors_directed(node.clone(), Outgoing) {
                let temp_mi = MathExpression::Mi(
                    graph.clone()[node1].properties["name"].to_string()[1..(graph.clone()[node1]
                        .properties["name"]
                        .to_string()
                        .len()
                        - 1 as usize)]
                        .to_string()
                        .clone(),
                );
                args[i].push(op[0].clone());
                args[i].push(temp_mi.clone());
            }
        } else if graph.clone()[node].labels == ["Opi"] || graph.clone()[node].labels == ["Literal"]
        {
            let temp_mi = MathExpression::Mi(
                graph.clone()[node].properties["name"].to_string()
                    [1..(graph.clone()[node].properties["name"].to_string().len() - 1 as usize)]
                    .to_string()
                    .clone(),
            );
            args[i].push(temp_mi.clone());
        } else {
            let n_args = get_args(graph.clone(), node.clone());
            let mut temp_vec = Vec::<MathExpression>::new();
            temp_vec.extend_from_slice(&n_args[0]);
            temp_vec.push(get_operator(graph.clone(), node.clone()));
            temp_vec.extend_from_slice(&n_args[1]);
            args[i].extend_from_slice(&temp_vec.clone());
        }
    }

    return args;
}

// this gets the operator from the node name
fn get_operator(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> MathExpression {
    let mut op = Vec::<MathExpression>::new();
    if graph.clone()[root_node].properties["name"].to_string() == "'*'".to_string() {
        op.push(Mo(Operator::Multiply));
    } else if graph.clone()[root_node].properties["name"].to_string() == "'+'".to_string() {
        op.push(Mo(Operator::Add));
    } else if graph.clone()[root_node].properties["name"].to_string() == "'-'".to_string() {
        op.push(Mo(Operator::Subtract));
    } else if graph.clone()[root_node].properties["name"].to_string() == "'/'".to_string() {
        op.push(Mo(Operator::Divide));
    } else {
        op.push(Mo(Operator::Other(
            graph.clone()[root_node].properties["name"].to_string(),
        )));
    }
    return op[0].clone();
}

fn tree_2_expr(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Result<Expr, MgError> {
    // initialize struct properties
    let mut op_vec = Vec::<Operator>::new();
    op_vec.push(Operator::Other("".to_string())); // empty first element to initialize
    let mut args_vec = Vec::<Expr>::new();
    let mut expr_name = String::from("");

    if graph.clone()[root_node].labels == ["Opo"] {
        // starting node in expression tree, traverse down one node and then start parse
        for node in graph.neighbors_directed(root_node, Outgoing) {
            if graph.clone()[node].labels == ["Primitive"]
                && !(graph.clone()[node].properties["name"].to_string() == "'unpack'".to_string())
            {
                // make an operator type based on operation
                if graph.clone()[node].properties["name"].to_string() == "'+'".to_string() {
                    op_vec.push(Operator::Add);
                } else if graph.clone()[node].properties["name"].to_string() == "'-'".to_string() {
                    op_vec.push(Operator::Subtract);
                } else if graph.clone()[node].properties["name"].to_string() == "'*'".to_string() {
                    op_vec.push(Operator::Multiply);
                } else if graph.clone()[node].properties["name"].to_string() == "'/'".to_string() {
                    op_vec.push(Operator::Divide);
                } else {
                    panic!("Unknown Primitive!");
                }
                // name expression
                expr_name = graph.clone()[root_node].properties["name"].to_string();

                // now for the more complicated part, getting the arguments
                for node1 in graph.neighbors_directed(node, Outgoing) {
                    if graph.clone()[node1].properties["name"].to_string() == "'USub'".to_string() {
                        // this means there is a '-' outside one of the arguments and we need to traverse
                        // a node deeper to get the argument
                        for node2 in graph.neighbors_directed(node1, Outgoing) {
                            if graph.clone()[node2].labels == ["Opi"] {
                                let arg_string = format!(
                                    "{}",
                                    graph.clone()[node2].properties["name"].to_string()
                                );
                                args_vec.push(Expr::Atom(Atom::Identifier(arg_string)));
                            } else {
                                panic!("Unsupported edge case where 'USub' preceeds something besides an 'Opi'!");
                            }
                        }
                        op_vec[0] = Operator::Subtract;
                    // add this as a new unary operator to the expression
                    } else if graph.clone()[node1].labels == ["Primitive"] {
                        // this is the case where there are more operators and will likely require a recursive call.
                        let expr1 = tree_2_expr(graph.clone(), node1).unwrap();
                        args_vec.push(expr1);
                    } else if graph.clone()[node1].labels == ["Opi"] {
                        // nice and straight to an argument
                        args_vec.push(Expr::Atom(Atom::Identifier(
                            graph.clone()[node1].properties["name"].to_string(),
                        )));
                    } else {
                        panic!("Encoutered node that is not an 'Opi' or 'Primitive'!");
                    }
                }
            }
        }
    } else {
        if graph.clone()[root_node].labels == ["Primitive"]
            && !(graph.clone()[root_node].properties["name"].to_string() == "'unpack'".to_string())
        {
            // make an operator type based on operation
            if graph.clone()[root_node].properties["name"].to_string() == "'+'".to_string() {
                op_vec.push(Operator::Add);
            } else if graph.clone()[root_node].properties["name"].to_string() == "'-'".to_string() {
                op_vec.push(Operator::Subtract);
            } else if graph.clone()[root_node].properties["name"].to_string() == "'*'".to_string() {
                op_vec.push(Operator::Multiply);
            } else if graph.clone()[root_node].properties["name"].to_string() == "'/'".to_string() {
                op_vec.push(Operator::Divide);
            } else {
                panic!("Unknown Primitive!");
            }
            // name expression
            expr_name = graph.clone()[root_node].properties["name"].to_string();

            // now for the more complicated part, getting the arguments
            for node1 in graph.neighbors_directed(root_node, Outgoing) {
                if graph.clone()[node1].properties["name"].to_string() == "'USub'".to_string() {
                    // this means there is a '-' outside one of the arguments and we need to traverse
                    // a node deeper to get the argument
                    for node2 in graph.neighbors_directed(node1, Outgoing) {
                        if graph.clone()[node2].labels == ["Opi"] {
                            let arg_string =
                                format!("{}", graph.clone()[node2].properties["name"].to_string());
                            args_vec.push(Expr::Atom(Atom::Identifier(arg_string)));
                        } else {
                            panic!("Unsupported edge case where 'USub' preceeds something besides an 'Opi'!");
                        }
                    }
                    op_vec[0] = Operator::Subtract; // add this as a new unary operator to the expression
                } else if graph.clone()[node1].labels == ["Primitive"] {
                    // this is the case where there are more operators and will likely require a recursive call.
                    let expr1 = tree_2_expr(graph.clone(), node1).unwrap();
                    args_vec.push(expr1);
                } else if (graph.clone()[node1].labels == ["Opi"]
                    || graph.clone()[node1].labels == ["Literal"])
                {
                    // nice and straight to an argument
                    args_vec.push(Expr::Atom(Atom::Identifier(
                        graph.clone()[node1].properties["name"].to_string(),
                    )));
                } else {
                    println!("{:?}", graph.clone()[node1].labels);
                    panic!("Encoutered node that is not an 'Opi' or 'Primitive'!");
                }
            }
        }
    }

    // now to construct the Expr
    let temp_expr = Expr::Expression {
        ops: op_vec.clone(),
        args: args_vec.clone(),
        name: expr_name,
    };

    return Ok(temp_expr);
}

// this currently only works for un-named nodes that are not chained or have multiple incoming/outgoing edges
fn trim_un_named(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
) -> petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> {
    // first create a cloned version of the graph we can modify while iterating over it.
    let mut bypass_graph = graph.clone();

    // iterate over the graph and add a new edge to bypass the un-named nodes
    for node_index in graph.clone().node_indices() {
        if graph.clone()[node_index].properties["name"].to_string() == "'un-named'".to_string() {
            let mut bypass = Vec::<NodeIndex>::new();
            for node1 in graph.neighbors_directed(node_index, Incoming) {
                bypass.push(node1);
            }
            for node2 in graph.neighbors_directed(node_index, Outgoing) {
                bypass.push(node2);
            }
            if bypass.len() == 2 {
                bypass_graph.add_edge(
                    bypass[0].clone(),
                    bypass[1].clone(),
                    graph
                        .edge_weight(graph.find_edge(node_index, bypass[1]).unwrap())
                        .unwrap()
                        .clone(),
                );
            }
        }
    }

    // now we perform a filter_map to remove the un-named nodes and only the bypass edge will remain to connect the nodes
    // we also remove the unpack node if it is present here as well
    let trimmed_graph = bypass_graph.filter_map(
        |node_index, _edge_index| {
            if !(bypass_graph.clone()[node_index].properties["name"].to_string()
                == "'un-named'".to_string()
                || bypass_graph.clone()[node_index].properties["name"].to_string()
                    == "'unpack'".to_string())
            {
                Some(graph[node_index].clone())
            } else {
                None
            }
        },
        |_node_index, edge_index| Some(edge_index.clone()),
    );

    return trimmed_graph;
}

fn subgraph_wiring(
    module_id: i64,
) -> Result<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>, MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some("localhost".to_string()),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // create query for nodes
    let query1 = format!(
        "MATCH (n)-[*]->(m) WHERE id(n) = {}
        MATCH q = (l)<-[r:Wire]-(m)
        WITH reduce(output = [], m IN nodes(q) | output + m ) AS nodes1
        UNWIND nodes1 AS nodes2
        WITH DISTINCT nodes2
        return collect(nodes2)",
        module_id
    );

    // Run Query for nodes
    connection.execute(&query1, None)?;

    // collect nodes into list
    let mut node_list = Vec::<Node>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        node_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Node(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    // create query for edges
    let query2 = format!(
        "MATCH (n)-[*]->(m) WHERE id(n) = {}
        MATCH q = (l)<-[r:Wire]-(m)
        WITH reduce(output = [], m IN relationships(q) | output + m ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return collect(edges2)",
        module_id
    );

    // Run Query for edges
    connection.execute(&query2, None)?;

    // collect edges into list
    let mut edge_list = Vec::<Relationship>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        edge_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Relationship(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    // Create a petgraph graph
    let mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> = Graph::new();

    // Add nodes to the petgraph graph and collect their indexes
    let mut nodes = Vec::<NodeIndex>::new();
    for node in node_list {
        let n1 = graph.add_node(node);
        nodes.push(n1.clone());
    }

    // add the edges to the petgraph
    for edge in edge_list {
        let mut src = Vec::<NodeIndex>::new();
        let mut tgt = Vec::<NodeIndex>::new();
        for node_idx in nodes.clone() {
            if graph[node_idx].id == edge.start_id {
                src.push(node_idx.clone());
            }
            if graph[node_idx].id == edge.end_id {
                tgt.push(node_idx.clone());
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0].clone(), tgt[0].clone(), edge);
        }
    }
    return Ok(graph);
}

fn subgraph2petgraph(
    module_id: i64,
) -> petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> {
    let (x, y) = get_subgraph(module_id).unwrap();

    // Create a petgraph graph
    let mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> = Graph::new();

    // Add nodes to the petgraph graph and collect their indexes
    let mut nodes = Vec::<NodeIndex>::new();
    for node in x {
        let n1 = graph.add_node(node);
        nodes.push(n1.clone());
    }

    // add the edges to the petgraph
    for edge in y {
        let mut src = Vec::<NodeIndex>::new();
        let mut tgt = Vec::<NodeIndex>::new();
        for node_idx in nodes.clone() {
            if graph[node_idx].id == edge.start_id {
                src.push(node_idx.clone());
            }
            if graph[node_idx].id == edge.end_id {
                tgt.push(node_idx.clone());
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0].clone(), tgt[0].clone(), edge);
        }
    }
    return graph;
}

pub fn get_subgraph(module_id: i64) -> Result<(Vec<Node>, Vec<Relationship>), MgError> {
    // construct the query that will delete the module with a given unique identifier

    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some("localhost".to_string()),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // create query for nodes
    let query1 = format!(
        "MATCH p = (n)-[r*]->(m) WHERE id(n) = {}
    WITH reduce(output = [], n IN nodes(p) | output + n ) AS nodes1
    UNWIND nodes1 AS nodes2
    WITH DISTINCT nodes2
    return collect(nodes2)",
        module_id
    );

    // Run Query for nodes
    connection.execute(&query1, None)?;

    // collect nodes into list
    let mut node_list = Vec::<Node>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        node_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Node(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    // create query for edges
    let query2 = format!(
        "MATCH p = (n)-[r*]->(m) WHERE id(n) ={}
        WITH reduce(output = [], n IN relationships(p) | output + n ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return collect(edges2)",
        module_id
    );

    // Run Query for edges
    connection.execute(&query2, None)?;

    // collect edges into list
    let mut edge_list = Vec::<Relationship>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        edge_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Relationship(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    return Ok((node_list, edge_list));
}
