use mathml::acset::{InputArc, OutputArc, Specie, Transition};
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
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;

// new imports
use mathml::ast::MathExpression::Mo;
use mathml::ast::MathExpression::Mrow;
use mathml::petri_net::recognizers::get_polarity;
use mathml::petri_net::recognizers::get_specie_var;
use mathml::petri_net::recognizers::is_add_or_subtract_operator;
use mathml::petri_net::recognizers::is_var_candidate;
use mathml::petri_net::Var;

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
    println!("{:?}", args[1]);

    let module_id = 1525;
    // now to prototype an algorithm to find the function that contains the core dynamics

    if args[1] == "auto".to_string() {
        println!("auto branch");
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
        let mut functions =
            Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
        for i in 0..function_nodes.len() {
            // grab the subgraph of the given expression
            functions.push(subgraph2petgraph(graph[function_nodes[i]].id.clone()));
        }
        // get a sense of the number of expressions in each function
        let mut func_counter = 0;
        let mut core_func = 0;
        for func in functions.clone() {
            let mut expression_counter = 0;
            for node in func.node_indices() {
                if func[node].labels == ["Expression"] {
                    expression_counter += 1;
                }
            }
            if expression_counter >= 4 {
                core_func = func_counter;
            }
            func_counter += 1;
        }
        // 4. get the id of the core dynamics function
        let mut core_id = module_id;
        for node in functions[core_func].clone().node_indices() {
            if functions[core_func][node].labels == ["Function"] {
                core_id = functions[core_func][node].id.clone();
            }
        }

        // 5. pass id to subgrapg2_core_dyn to get core dynamics
        let (core_dynamics, metadata_map) = subgraph2_core_dyn(core_id).unwrap();

        println!("{:?}", core_dynamics[0].clone());
    }
    // This is the graph id for the top level function for the core dynamics for our test case.
    else if args[1] == "manual".to_string() {
        // still need to grab the module id

        let (mut core_dynamics, metadata_map) = subgraph2_core_dyn(module_id).unwrap();

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

        println!("{:?}", named_core_dynamics.clone());
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
    } else {
        println!("Unknown command!");
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

// This will parse the mathml based on a '=' anchor
fn ast2exp(mathml_ast: Vec<Math>) -> Vec<PreExp> {
    let mut named_core_dynamics = Vec::<PreExp>::new();

    // This is a lot of adarsh's code on somehow iteratoring over an enum
    let mut terms = Vec::<Term>::new();
    let mut current_term = Term::default();
    let mut lhs_specie: Option<Var> = None;
    let mut species: HashSet<Var>;
    let mut vars: HashSet<Var> = Default::default();
    let mut eqns: HashMap<Var, Vec<Term>>;

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
                    let mut pre_exp_args = Vec::<Expr>::new();
                    pre_exp.ops.push(Operator::Other("".to_string()));
                    pre_exp.ops.push(Operator::Equals);
                    pre_exp.args.push(Expr::Expression {
                        ops: [Operator::Other("Derivative".to_string())].to_vec(),
                        args: pre_exp_args,
                        name: "".to_string(),
                    })
                }
            }

            println!("rhs: {:?}", expr_1[equals_index + 1]);
            // Iterate over MathExpressions in the RHS
            for (_i, expr_2) in expr_1[equals_index + 1..].iter().enumerate() {
                if is_add_or_subtract_operator(expr_2) {
                    if current_term.vars.is_empty() {
                        current_term.polarity = get_polarity(expr_2);
                    } else {
                        terms.push(current_term);
                        current_term = Term {
                            vars: vec![],
                            polarity: get_polarity(expr_2),
                            ..Default::default()
                        };
                    }
                } else if is_var_candidate(expr_2) {
                    current_term.vars.push(Var(expr_2.clone()));
                    vars.insert(Var(expr_2.clone()));
                } else {
                    panic!("Unhandled rhs element {:?}", expr_2);
                }
            }
            if !current_term.vars.is_empty() {
                terms.push(current_term.clone());
            }
        }
    }

    return named_core_dynamics;
}

fn exp2pn(named_core_dynamics: Vec<PreExp>) -> ACSet {
    let mut ascet = ACSet {
        S: Vec::<Specie>::new(),
        T: Vec::<Transition>::new(),
        I: Vec::<InputArc>::new(),
        O: Vec::<OutputArc>::new(),
    };

    return ascet;
}

fn subgraph2_core_dyn(
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
