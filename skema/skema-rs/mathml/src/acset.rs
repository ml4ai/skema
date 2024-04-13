//! Structs to represent elements of ACSets (Annotated C-Sets, a concept from category theory).
//! JSON-serialized ACSets are the form of model exchange between TA1 and TA2.
use crate::ast::operator::Operator;
use crate::ast::{Ci, MathExpression, Mi, Type};
use crate::parsers::first_order_ode::{get_terms, FirstOrderODE, PnTerm};
use crate::parsers::math_expression_tree::MathExpressionTree;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use utoipa;
use utoipa::ToSchema;

// We keep our ACSet representation in addition to the new SKEMA model representation since it is
// more compact and easy to work with for development.

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema, JsonSchema,
)]
pub struct Specie {
    pub sname: String,
    pub uid: usize,
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema, JsonSchema,
)]
pub struct ACSetTransition {
    pub tname: String,
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema, JsonSchema,
)]
pub struct InputArc {
    pub it: usize,
    pub is: usize,
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema, JsonSchema,
)]
pub struct OutputArc {
    pub ot: usize,
    pub os: usize,
}

#[allow(non_snake_case)]
#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct ACSet {
    pub S: Vec<Specie>,
    pub T: Vec<ACSetTransition>,
    pub I: Vec<InputArc>,
    pub O: Vec<OutputArc>,
}

// -------------------------------------------------------------------------------------------
// The following data structs are those requested by TA-4 as an exchange format for the models.
// the spec in json format can be found here: https://github.com/DARPA-ASKEM/Model-Representations/blob/main/petrinet/petrinet_schema.json
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct PetriNet {
    pub header: Header,
    pub model: ModelPetriNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantics: Option<Semantics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct RegNet {
    pub header: Header,
    pub model: ModelRegNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Header {
    pub name: String,
    pub schema: String,
    pub schema_name: String,
    pub description: String,
    pub model_version: String,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct ModelRegNet {
    pub vertices: BTreeSet<RegState>,
    pub edges: BTreeSet<RegTransition>,
    /// Note: parameters is a required field in the schema, but we make it optional since we want
    /// to reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Parameter>>,
}
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct ModelPetriNet {
    pub states: BTreeSet<State>,
    pub transitions: BTreeSet<Transition>,
    /// Note: parameters is a required field in the schema, but we make it optional since we want
    /// to reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Metadata {
    pub placeholder: String, // once we finalize the metadata data struct fill in this data struct
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Semantics {
    pub ode: Ode,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Ode {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rates: Option<Vec<Rate>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initials: Option<Vec<Initial>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observables: Option<Vec<Observable>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time: Option<Time>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Observable {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    states: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expression: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expression_mathml: Option<String>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct RegState {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_constant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sign: Option<bool>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct State {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<Units>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Units {
    pub expression: String,
    pub expression_mathml: String,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Grounding {
    pub identifiers: Identifier,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Identifier {
    pub ido: String,
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema, JsonSchema,
)]
pub struct Initial {
    pub target: String,
    pub expression: String,
    pub expression_mathml: String,
}

impl Default for Initial {
    fn default() -> Self {
        Initial {
            target: "temp".to_string(),
            expression: "0".to_string(),
            expression_mathml: "<math></math>".to_string(),
        }
    }
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Rate {
    pub target: String,
    pub expression: String,
    pub expression_mathml: Option<String>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct RegTransition {
    pub id: String,

    /// Note: source is a required field in the schema, but we make it optional since we want to
    /// reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Note: target is a required field in the schema, but we make it optional since we want to
    /// reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sign: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Properties>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Transition {
    pub id: String,

    /// Note: input is a required field in the schema, but we make it optional since we want to
    /// reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<String>>,

    /// Note: output is a required field in the schema, but we make it optional since we want to
    /// reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sign: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Properties>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Properties {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_constant: Option<String>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Parameter {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution: Option<Distribution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<Units>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Time {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<Units>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct Distribution {
    #[serde(rename = "type")]
    pub r#type: String,
    pub parameters: String,
}

// This is for the routing of mathml for various endpoints to extract the appropriate AMR
#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct AMRmathml {
    pub model: String,
    pub mathml: Vec<String>,
}

// -------------------------------------------------------------------------------------------
// These next structs are for Generalized AMR's
// -------------------------------------------------------------------------------------------
#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct GeneralizedAMR {
    pub header: Header,
    pub met: Vec<MathExpressionTree>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantics: Option<GeneralSemantics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    ToSchema,
    JsonSchema,
)]
pub struct GeneralSemantics {
    pub states: BTreeSet<State>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Parameter>>,
}

// -------------------------------------------------------------------------------------------
// This function takes our previous model form, the ACSet and transforms it to the new TA4 exchange format
// -------------------------------------------------------------------------------------------
impl From<Vec<MathExpressionTree>> for GeneralizedAMR {
    fn from(met_vec: Vec<MathExpressionTree>) -> GeneralizedAMR {
        let mut states_vec = BTreeSet::<State>::new();
        let mut parameter_vec = Vec::<Parameter>::new();
        let mut rhs_vec = Vec::<MathExpressionTree>::new();

        // construct state vector, under assumption that only differentialed LHS terms are states
        for equation in met_vec.iter() {
            match equation {
                MathExpressionTree::Cons(ref x, ref y) => match &x {
                    Operator::Equals => match &y[0] {
                        MathExpressionTree::Cons(Operator::Derivative(_d), ref x1) => {
                            let state_name = x1[0].to_string();
                            let state = State {
                                id: state_name.clone(),
                                name: state_name.clone(),
                                grounding: None,
                                units: None,
                            };
                            states_vec.insert(state.clone());
                            rhs_vec.push(y[1].clone());
                        }
                        _ => {
                            println!("Non-differential Equation");
                            rhs_vec.push(y[1].clone());
                        }
                    },
                    _ => {
                        println!("Expected an equation!")
                    }
                },
                _ => {
                    println!("Expected an equation!")
                }
            }
        }

        // now to construct the parameters vector
        // might be best to make a first order ODE and pass the get terms thing and then pull all terms from it
        // would need to flatten the mults and then pull make temp lhs
        let mut param_str_vec = Vec::<String>::new();
        let mut state_str_vec = Vec::<String>::new();
        for state in states_vec.iter() {
            state_str_vec.push(state.name.clone());
        }
        for (i, _equation) in met_vec.iter().enumerate() {
            let deriv = Ci {
                r#type: Some(Type::Function),
                content: Box::new(MathExpression::Mi(Mi("temp".to_string()))),
                func_of: None,
                notation: None,
            };
            let fode = FirstOrderODE {
                lhs_var: deriv.clone(),
                func_of: [deriv.clone()].to_vec(), // just place holders for construction
                with_respect_to: deriv.clone(),    // just place holders for construction
                rhs: rhs_vec[i].clone(),
            };
            let terms = get_terms(state_str_vec.clone(), fode);
            for term in terms.iter() {
                println!("{:?}", term.clone());
                println!("{:?}", term.parameters.clone());
                param_str_vec.extend(term.parameters.clone().into_iter());
            }
        }

        // dedup the parameters vector
        param_str_vec.sort();
        param_str_vec.dedup();

        // now to make the parameter vec from the strings
        for param in param_str_vec.iter() {
            let parameter = Parameter {
                id: param.clone(),
                name: Some(param.clone()),
                ..Default::default()
            };
            parameter_vec.push(parameter.clone());
        }

        // now to trim the numbers from the parameters field
        let mut nums = Vec::<usize>::new();
        for (k, param) in parameter_vec.iter().enumerate() {
            if param.id.parse::<f32>().is_ok() {
                nums.push(k);
            }
        }

        for num in nums.iter().rev() {
            parameter_vec.remove(num.clone());
        }

        let header = Header {
            name: "Model".to_string(),
            schema: "G-AMR".to_string(),
            schema_name: "Generalized AMR".to_string(),
            description: "Generalized AMR model from...".to_string(),
            model_version: "0.1".to_string(),
        };

        let semantics = GeneralSemantics {
            states: states_vec,
            parameters: Some(parameter_vec),
        };

        GeneralizedAMR {
            header,
            met: met_vec.clone(),
            semantics: Some(semantics),
            metadata: None,
        }
    }
}

impl From<Vec<FirstOrderODE>> for PetriNet {
    fn from(ode_vec: Vec<FirstOrderODE>) -> PetriNet {
        // initialize vecs
        let mut states_vec = BTreeSet::<State>::new();
        let mut transitions_vec = BTreeSet::<Transition>::new();
        let mut initial_vec = Vec::<Initial>::new();
        let mut parameter_vec = Vec::<Parameter>::new();
        let mut rate_vec = Vec::<Rate>::new();
        let mut state_string_list = Vec::<String>::new();
        let mut terms = Vec::<PnTerm>::new();
        let mut dirty_terms = Vec::<PnTerm>::new();

        // this first for loop is for the creation state related parameters in the AMR
        for ode in ode_vec.iter() {
            let states = State {
                id: ode.lhs_var.to_string().clone(),
                name: ode.lhs_var.to_string().clone(),
                ..Default::default()
            };
            let initials = Initial {
                target: ode.lhs_var.to_string().clone(),
                ..Default::default()
            };
            /*let parameters = Parameter {
                id: initials.expression.clone(),
                name: Some(initials.expression.clone()),
                description: Some(format!(
                    "The total {} population at timestep 0",
                    ode.lhs_var.to_string().clone()
                )),
                ..Default::default()
            };*/
            //parameter_vec.push(parameters.clone());
            initial_vec.push(initials.clone());
            states_vec.insert(states.clone());
            state_string_list.push(ode.lhs_var.to_string().clone()); // used later for transition parsing
            println!("ode.rhs: {:?}", ode.rhs.to_string().clone());
        }

        // now for the construction of the transitions and their results

        // this collects all the terms from the equations
        for ode in ode_vec.iter() {
            dirty_terms.append(&mut get_terms(state_string_list.clone(), ode.clone()));
        }
        // now to trim off terms that are for euler methods, dyn_state != exp_state && parameters.len() != 0
        // this conditional does nothing now, but is kept in case we need to turn it on later.
        for term in dirty_terms.iter() {
            println!("-----\nterm: {:?}\n-----", term.clone());
            if !term.exp_states.is_empty() {
                if term.dyn_state != term.exp_states[0] || !term.parameters.is_empty() {
                    terms.push(term.clone());
                } else {
                    terms.push(term.clone());
                }
            }
        }
        for term in terms.iter() {
            for param in &term.parameters {
                let parameters = Parameter {
                    id: param.clone(),
                    name: Some(param.clone()),
                    //description: Some(format!("{} rate", param.clone())),
                    ..Default::default()
                };
                parameter_vec.push(parameters.clone());
            }
        }

        // first is we need to replace any terms with more than 2 exp_states with subterms, these are simply
        // terms that need to be distributed (ASSUMPTION, MOST A TRANSTION CAN HAVE IS 2 IN AND 2 OUT)
        // but first we need to inherit the dynamic state to each sub term
        /*let mut composite_term_ind = Vec::<usize>::new();
        let mut sub_terms = Vec::<PnTerm>::new();
        for (j, t) in terms.clone().iter().enumerate() {
            if t.exp_states.len() > 2 && t.sub_terms.is_some() {
                for (i, _sub_t) in t.sub_terms.clone().unwrap().iter().enumerate() {
                    terms[j].sub_terms.as_mut().unwrap()[i].dyn_state = t.dyn_state.clone();
                    sub_terms.push(terms[j].sub_terms.as_mut().unwrap()[i].clone());
                }
                composite_term_ind.push(j);
            }
        }
        // delete composite terms we are replacing
        composite_term_ind.sort();
        for i in composite_term_ind.iter().rev() {
            terms.remove(*i);
        }
        // replace with subterms
        terms.append(&mut sub_terms);*/

        // now for polarity pairs of terms we need to construct the transistions
        let mut paired_term_indices = Vec::<usize>::new();
        let mut transition_pair = Vec::<(PnTerm, PnTerm)>::new();
        for (i, term1) in terms.clone().iter().enumerate() {
            for (j, term2) in terms.clone().iter().enumerate() {
                if term1.polarity != term2.polarity
                    && term1.parameters == term2.parameters
                    && term1.polarity
                    && term1.exp_states == term2.exp_states
                {
                    let temp_pair = (term1.clone(), term2.clone());
                    transition_pair.push(temp_pair);
                    paired_term_indices.push(i);
                    paired_term_indices.push(j);
                }
            }
        }

        // delete paired terms from list
        paired_term_indices.sort();
        for i in paired_term_indices.iter().rev() {
            terms.remove(*i);
        }
        // Now we replace unpaired terms with subterms, by their subterms and repeat the process

        // but first we need to inherit the dynamic state to each sub term
        let mut composite_term_ind = Vec::<usize>::new();
        let mut sub_terms = Vec::<PnTerm>::new();
        for (j, t) in terms.clone().iter().enumerate() {
            if t.sub_terms.is_some() {
                for (i, _sub_t) in t.sub_terms.clone().unwrap().iter().enumerate() {
                    terms[j].sub_terms.as_mut().unwrap()[i].dyn_state = t.dyn_state.clone();
                    sub_terms.push(terms[j].sub_terms.as_mut().unwrap()[i].clone());
                }
                composite_term_ind.push(j);
            }
        }

        // delete composite terms
        composite_term_ind.sort();
        for i in composite_term_ind.iter().rev() {
            terms.remove(*i);
        }

        // replace with subterms
        terms.append(&mut sub_terms);

        // now we attempt to pair again and delete paired terms again
        let mut paired_term_indices = Vec::<usize>::new();
        for (i, term1) in terms.clone().iter().enumerate() {
            for (j, term2) in terms.clone().iter().enumerate() {
                if term1.polarity != term2.polarity
                    && term1.parameters == term2.parameters
                    && term1.polarity
                    && term1.exp_states == term2.exp_states
                {
                    let temp_pair = (term1.clone(), term2.clone());
                    transition_pair.push(temp_pair);
                    paired_term_indices.push(i);
                    paired_term_indices.push(j);
                }
            }
        }
        paired_term_indices.sort();
        for i in paired_term_indices.iter().rev() {
            terms.remove(*i);
        }

        // now we construct transitions of all paired terms
        for (i, t) in transition_pair.iter().enumerate() {
            if t.0.exp_states.len() == 1 {
                // construct transtions for simple transtions
                let transitions = Transition {
                    id: format!("t{}", i.clone()),
                    input: Some([t.1.dyn_state.clone()].to_vec()),
                    output: Some([t.0.dyn_state.clone()].to_vec()),
                    ..Default::default()
                };
                transitions_vec.insert(transitions.clone());

                let rate = Rate {
                    target: transitions.id.clone(),
                    expression: t.0.expression_infix.clone()[1..t.0.expression_infix.clone().len()-1].to_string(),// the second 
                    expression_mathml: Some(t.0.expression.clone()),
                };
                rate_vec.push(rate.clone());
            } else {
                // construct transitions for complicated transitions
                // mainly need to construct the output specially,
                let mut output = [t.0.dyn_state.clone()].to_vec();

                for state in t.0.exp_states.iter() {
                    if *state != t.1.dyn_state {
                        output.push(state.clone());
                    }
                }

                let transitions = Transition {
                    id: format!("t{}", i.clone()),
                    input: Some(t.1.exp_states.clone()),
                    output: Some(output.clone()),
                    ..Default::default()
                };
                transitions_vec.insert(transitions.clone());

                let rate = Rate {
                    target: transitions.id.clone(),
                    expression: t.0.expression_infix.clone()[1..t.0.expression_infix.clone().len()-1].to_string(),// the second 
                    expression_mathml: Some(t.0.expression.clone()),
                };
                rate_vec.push(rate.clone());
            }
        }

        // now we construct transitions from unpaired terms, assuming them to be sources and sinks
        // This should also support sources and sinks that are state dependent.
        if !terms.is_empty() {
            for (i, term) in terms.iter().enumerate() {
                if term.polarity {
                    let mut input = Vec::<String>::new();
                    let mut exp_eq_dyn = false;

                    let mut output = [term.dyn_state.clone()].to_vec();

                    for state in term.exp_states.iter() {
                        input.push(state.clone());
                        if *state == term.dyn_state.clone() {
                            exp_eq_dyn = true;
                        }
                    }

                    // I think if the expression equals the dynamic state both in the input and output get
                    if exp_eq_dyn {
                        input.push(term.dyn_state.clone());
                        output.push(term.dyn_state.clone());
                    }

                    let transitions = Transition {
                        id: format!("s{}", i),
                        input: Some(input.clone()),
                        output: Some(output.clone()),
                        ..Default::default()
                    };
                    transitions_vec.insert(transitions.clone());

                    let rate = Rate {
                        target: transitions.id.clone(),
                        expression: term.expression_infix.clone()[1..term.expression_infix.clone().len()-1].to_string(),// the second term needs to be the product of the inputs
                        expression_mathml: Some(term.expression.clone()),
                    };
                    rate_vec.push(rate.clone());
                } else {
                    let mut input = [term.dyn_state.clone()].to_vec();

                    for state in term.exp_states.iter() {
                        input.push(state.clone());
                    }

                    let transitions = Transition {
                        id: format!("s{}", i),
                        input: Some(input.clone()),
                        output: None,
                        ..Default::default()
                    };
                    transitions_vec.insert(transitions.clone());

                    let rate = Rate {
                        target: transitions.id.clone(),
                        expression: term.expression_infix.clone()[1..term.expression_infix.clone().len()-1].to_string(),// the second 
                        expression_mathml: Some(term.expression.clone()),
                    };
                    rate_vec.push(rate.clone());
                }
            }
        }

        // trim duplicate parameters and (TODO)remove integer parameters

        parameter_vec.sort();
        parameter_vec.dedup();

        // now to trim the numbers from the parameters field
        let mut nums = Vec::<usize>::new();
        for (k, param) in parameter_vec.iter().enumerate() {
            if param.id.parse::<f32>().is_ok() {
                nums.push(k);
            }
        }

        for num in nums.iter().rev() {
            parameter_vec.remove(num.clone());
        }

        // construct the PetriNet
        let ode = Ode {
            rates: Some(rate_vec),
            initials: Some(initial_vec),
            parameters: Some(parameter_vec),
            ..Default::default()
        };

        let semantics = Semantics { ode };

        let model = ModelPetriNet {
            states: states_vec,
            transitions: transitions_vec,
            metadata: None,
        };

        let header = Header {
            name: "mathml model".to_string(),
            schema: "https://github.com/DARPA-ASKEM/Model-Representations/blob/main/petrinet/petrinet_schema.json".to_string(),
            schema_name: "PetriNet".to_string(),
            description: "This is a model from equations".to_string(),
            model_version: "0.1".to_string(),
        };

        PetriNet {
            header,
            model,
            semantics: Some(semantics),
            metadata: None,
        }
    }
}

// This impl will take a vector of FirstOrderODE and return the RegNet for it
impl From<Vec<FirstOrderODE>> for RegNet {
    fn from(ode_vec: Vec<FirstOrderODE>) -> RegNet {
        // get the terms
        let mut terms = Vec::<PnTerm>::new();
        let mut sys_states = Vec::<String>::new();

        for ode in ode_vec.iter() {
            sys_states.push(ode.lhs_var.to_string().clone());
        }

        for ode in ode_vec.iter() {
            terms.append(&mut get_terms(sys_states.clone(), ode.clone()));
        }
        // -----------------------------------------------------------
        // -----------------------------------------------------------

        let mut states_vec = BTreeSet::<RegState>::new();
        let mut transitions_vec = BTreeSet::<RegTransition>::new();
        let mut parameter_vec = Vec::<Parameter>::new();

        // construct the states

        for state in sys_states.iter() {
            // This constructs the intital state, without rate_constant or sign yet
            let mut r_state = RegState {
                id: state.clone(),
                name: state.clone(),
                grounding: None,
                initial: Some(format!("{}0", state)),
                rate_constant: None,
                sign: None,
            };
            // This finishes the construction of the state
            let mut counter = 0;
            for term in terms.iter() {
                if term.exp_states.len() == 1 && term.exp_states[0] == *state {
                    // note this is only grabbing one term. This is somewhat limited by the current AMR schema
                    // it assumes only a simple single parameter for this Date: 08/10/23
                    r_state.rate_constant = Some(term.parameters[0].clone());
                    r_state.sign = Some(term.polarity);
                    // This adds the edges for the environment couplings
                    //---DO WE INCLUDE THE SINGLE TRANSITION?---
                    /*let prop = Properties {
                        name: term.parameters[0].clone(),
                        rate_constant: None,
                    };
                    let self_trans = RegTransition {
                        id: format!("s{}", counter.clone()),
                        source: Some([term.dyn_state.clone()].to_vec()),
                        target: Some([term.dyn_state.clone()].to_vec()),
                        sign: Some(term.polarity),
                        grounding: None,
                        properties: Some(prop.clone()),
                    };
                    transitions_vec.insert(self_trans.clone()); */
                    counter += 1;
                }
            }
            // This adds the intial values from the state variables into the parameters vec
            let parameters = Parameter {
                id: r_state.initial.clone().unwrap(),
                name: r_state.initial.clone(),
                description: Some(format!(
                    "The total {} population at timestep 0",
                    state.clone()
                )),
                ..Default::default()
            };
            parameter_vec.push(parameters.clone());
            states_vec.insert(r_state.clone());
        }
        // construct the transitions

        // first for the polarity pairs of terms we need to construct the transistions
        let mut transition_pair = Vec::<(PnTerm, PnTerm)>::new();
        let mut paired_indicies = Vec::<usize>::new();
        for (i, term1) in terms.clone().iter().enumerate() {
            for (j, term2) in terms.clone().iter().enumerate() {
                if term1.polarity != term2.polarity
                    && term1.parameters == term2.parameters
                    && term1.polarity
                {
                    // first term is positive, second is negative
                    let temp_pair = (term1.clone(), term2.clone());
                    transition_pair.push(temp_pair);
                    paired_indicies.push(i);
                    paired_indicies.push(j);
                }
            }
        }

        paired_indicies.sort();
        paired_indicies.dedup();

        let mut unpaired_terms = terms.clone();
        for i in paired_indicies.iter().rev() {
            unpaired_terms.remove(*i);
        }

        let mut trans_num = 0;

        for (_i, t) in transition_pair.iter().enumerate() {
            if t.0.exp_states.len() == 1 {
                // construct transtions for simple transtions
                let prop = Properties {
                    // once again the assumption of only one parameters for transition
                    name: t.0.parameters[0].clone(),
                    rate_constant: None,
                };
                let trans = RegTransition {
                    id: format!("t{}", trans_num.clone()),
                    source: Some(t.1.dyn_state.clone()),
                    target: Some(t.0.dyn_state.clone()),
                    sign: Some(true),
                    grounding: None,
                    properties: Some(prop.clone()),
                };
                trans_num = trans_num + 1;
                transitions_vec.insert(trans.clone());
                let trans = RegTransition {
                    id: format!("t{}", trans_num.clone()),
                    source: Some(t.0.dyn_state.clone()),
                    target: Some(t.1.dyn_state.clone()),
                    sign: Some(false),
                    grounding: None,
                    properties: Some(prop.clone()),
                };
                trans_num = trans_num + 1;
                transitions_vec.insert(trans.clone());
            } else {
                // construct transitions for complicated transitions
                // mainly need to construct the output specially,
                // run by clay
                let mut output = [t.0.dyn_state.clone()].to_vec();

                for state in t.0.exp_states.iter() {
                    if *state != t.1.dyn_state {
                        output.push(state.clone());
                    }
                }

                let prop = Properties {
                    // once again the assumption of only one parameters for transition
                    name: t.0.parameters[0].clone(),
                    rate_constant: None,
                };
                for (j, _out) in output.iter().enumerate() {
                    let trans = RegTransition {
                        id: format!("t{}", trans_num.clone()),
                        source: Some(t.1.exp_states[j].clone()),
                        target: Some(output[j].clone()),
                        sign: Some(true),
                        grounding: None,
                        properties: Some(prop.clone()),
                    };
                    transitions_vec.insert(trans.clone());
                    trans_num = trans_num + 1;
                }
            }
        }

        for (i, term) in unpaired_terms.iter().enumerate() {
            println!("Term: {:?}", term.clone());
            if term.exp_states.len() > 1 {
                let mut output = term.dyn_state.clone();
                let mut input = term.exp_states.clone();

                let param_len = term.parameters.len();

                let prop = Properties {
                    // once again the assumption of only one parameters for transition
                    name: term.parameters[param_len - 1].clone(),
                    rate_constant: None,
                };

                input.sort();
                input.dedup();

                if input.clone().len() > 1 {
                    let old_input = input.clone();
                    input = [].to_vec();
                    for term in old_input.clone().iter() {
                        if *term != output {
                            input.push(term.clone());
                        }
                    }
                }
                for (j, _trm) in input.iter().enumerate() {
                    let trans = RegTransition {
                        id: format!("s{}", trans_num.clone()),
                        source: Some(input[j].clone()),
                        target: Some(output.clone()),
                        sign: Some(term.polarity),
                        grounding: None,
                        properties: Some(prop.clone()),
                    };
                    transitions_vec.insert(trans.clone());
                    trans_num = trans_num + 1;
                }
            }
        }

        // construct the remaining parameters

        for term in terms.iter() {
            for param in &term.parameters {
                let parameters = Parameter {
                    id: param.clone(),
                    name: Some(param.clone()),
                    description: Some(format!("{} rate", param.clone())),
                    ..Default::default()
                };
                parameter_vec.push(parameters.clone());
            }
        }

        parameter_vec.sort();
        parameter_vec.dedup();

        // now to trim the numbers from the parameters field
        let mut nums = Vec::<usize>::new();
        for (k, param) in parameter_vec.iter().enumerate() {
            if param.id.parse::<f32>().is_ok() {
                nums.push(k);
            }
        }

        for num in nums.iter().rev() {
            parameter_vec.remove(num.clone());
        }

        // ------------------------------------------

        let model = ModelRegNet {
            vertices: states_vec,
            edges: transitions_vec,
            parameters: Some(parameter_vec),
        };

        let header = Header {
            name: "Regnet mathml model".to_string(),
            schema: "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/regnet_v0.1/regnet/regnet_schema.json".to_string(),
            schema_name: "regnet".to_string(),
            description: "This is a Regnet model from mathml equations".to_string(),
            model_version: "0.1".to_string(),
        };

        RegNet {
            header,
            model,
            metadata: None,
        }
    }
}
