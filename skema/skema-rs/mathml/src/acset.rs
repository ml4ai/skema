//! Structs to represent elements of ACSets (Annotated C-Sets, a concept from category theory).
//! JSON-serialized ACSets are the form of model exchange between TA1 and TA2.
use crate::ast::{Math, MathExpression::Mi};
use crate::petri_net::{Polarity, Var};
use crate::{
    mml2pn::{group_by_operators, Term},
    parsing::parse,
};
use serde::{Deserialize, Serialize};

use std::collections::{BTreeSet, HashMap, HashSet};
use utoipa;
use utoipa::ToSchema;

// We keep our ACSet representation in addition to the new SKEMA model representation since it is
// more compact and easy to work with for development.

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema)]
pub struct Specie {
    pub sname: String,
    pub uid: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema)]
pub struct ACSetTransition {
    pub tname: String,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema)]
pub struct InputArc {
    pub it: usize,
    pub is: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema)]
pub struct OutputArc {
    pub ot: usize,
    pub os: usize,
}

#[allow(non_snake_case)]
#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
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
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct PetriNet {
    pub name: String,
    pub schema: String,
    pub schema_name: String,
    pub description: String,
    pub model_version: String,
    pub model: ModelPetriNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct RegNet {
    pub name: String,
    pub schema: String,
    pub schema_name: String,
    pub description: String,
    pub model_version: String,
    pub model: ModelRegNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelRegNet {
    pub vertices: BTreeSet<RegState>,
    pub edges: BTreeSet<RegTransition>,
    /// Note: parameters is a required field in the schema, but we make it optional since we want
    /// to reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Parameter>>,
}
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelPetriNet {
    pub states: BTreeSet<State>,
    pub transitions: BTreeSet<Transition>,
    /// Note: parameters is a required field in the schema, but we make it optional since we want
    /// to reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantics: Option<Semantics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Metadata {
    pub placeholder: String, // once we finalize the metadata data struct fill in this data struct
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Semantics {
    pub ode: Ode,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
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
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
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
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
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
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
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
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Units {
    pub expression: String,
    pub expression_mathml: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Grounding {
    pub identifiers: Identifier,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Identifier {
    pub ido: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Initial {
    pub target: String,
    pub expression: String,
    pub expression_mathml: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Rate {
    pub target: String,
    pub expression: String,
    pub expression_mathml: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct RegTransition {
    pub id: String,

    /// Note: source is a required field in the schema, but we make it optional since we want to
    /// reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Vec<String>>,

    /// Note: target is a required field in the schema, but we make it optional since we want to
    /// reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sign: Option<bool>,

    pub grounding: Option<Grounding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Properties>,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
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

    pub grounding: Option<Grounding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Properties>,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Properties {
    pub rate_constant: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
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
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Time {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<Units>,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Distribution {
    #[serde(rename = "type")]
    pub r#type: String,
    pub parameters: String,
}

// This is for the routing of mathml for various endpoints to extract the appropriate AMR
#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct AMRmathml {
    pub model: String,
    pub mathml: Vec<String>,
}

// -------------------------------------------------------------------------------------------
// This function takes our previous model form, the ACSet and transforms it to the new TA4 exchange format
// -------------------------------------------------------------------------------------------
impl From<ACSet> for PetriNet {
    fn from(pn: ACSet) -> PetriNet {
        let mut states_vec = BTreeSet::<State>::new();
        let mut transitions_vec = BTreeSet::<Transition>::new();

        // -----------------------------------------------------------

        for state in pn.S.iter() {
            let states = State {
                id: state.sname.clone(),
                name: state.sname.clone(),
                ..Default::default()
            };
            states_vec.insert(states.clone());
        }

        for (i, trans) in pn.T.iter().enumerate() {
            // convert transition index to base 1
            let transition = i + 1;

            // construct array of incoming states
            let mut string_vec1 = Vec::<String>::new();
            for incoming in pn.I.clone().iter() {
                if incoming.it == transition {
                    let state_idx = incoming.is - 1; // 0 based to index the rust vec
                    let state_name = pn.S[state_idx].sname.clone();
                    string_vec1.push(state_name.clone());
                }
            }
            // construct array of outgoing states
            let mut string_vec2 = Vec::<String>::new();
            for outgoing in pn.O.clone().iter() {
                if outgoing.ot == transition {
                    let state_idx = outgoing.os - 1; // 0 based to index the rust vec
                    let state_name = pn.S[state_idx].sname.clone();
                    string_vec2.push(state_name.clone());
                }
            }

            let transitions = Transition {
                id: trans.tname.clone(),
                input: Some(string_vec1.clone()),
                output: Some(string_vec2.clone()),
                ..Default::default()
            };

            transitions_vec.insert(transitions.clone());
        }

        // -----------------------------------------------------------

        let model = ModelPetriNet {
            states: states_vec,
            transitions: transitions_vec,
            semantics: None,
            metadata: None,
        };

        PetriNet {
        name: "mathml model".to_string(),
        schema: "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json".to_string(),
        schema_name: "PetriNet".to_string(),
        description: "This is a model from mathml equations".to_string(),
        model_version: "0.1".to_string(),
        model,
        metadata: None,
    }
    }
}

// This function takes in a mathml string and returns a Regnet
impl From<Vec<Math>> for RegNet {
    fn from(mathml_asts: Vec<Math>) -> RegNet {
        // this algorithm to follow should be refactored into a seperate function once it is functional

        let mut specie_vars = HashSet::<Var>::new();
        let mut vars = HashSet::<Var>::new();
        let mut eqns = HashMap::<Var, Vec<Term>>::new();

        for ast in mathml_asts.into_iter() {
            group_by_operators(ast, &mut specie_vars, &mut vars, &mut eqns);
        }

        // Get the rate variables
        let _rate_vars: HashSet<&Var> = vars.difference(&specie_vars).collect();

        // -----------------------------------------------------------
        // -----------------------------------------------------------

        let mut states_vec = BTreeSet::<RegState>::new();
        let mut transitions_vec = BTreeSet::<RegTransition>::new();

        for state in specie_vars.clone().into_iter() {
            // state bits
            let mut rate_const = "temp".to_string();
            let mut state_name = "temp".to_string();
            let mut term_idx = 0;
            let mut rate_sign = false;

            //transition bits
            let mut trans_name = "temp".to_string();
            let mut trans_sign = false;
            let _trans_tgt = "temp".to_string();
            let mut trans_src = "temp".to_string();

            for (i, term) in eqns[&state].iter().enumerate() {
                for variable in term.vars.clone().iter() {
                    if state == variable.clone() && term.vars.len() == 2 {
                        term_idx = i;
                    }
                }
            }

            if eqns[&state.clone()][term_idx].polarity == Polarity::Positive {
                rate_sign = true;
            }

            for variable in eqns[&state][term_idx].vars.iter() {
                if state.clone() != variable.clone() {
                    match variable.clone() {
                        Var(Mi(x)) => {
                            rate_const = x.clone();
                        }
                        _ => {
                            println!("Error in rate extraction");
                        }
                    };
                } else {
                    match variable.clone() {
                        Var(Mi(x)) => {
                            state_name = x.clone();
                        }
                        _ => {
                            println!("Error in rate extraction");
                        }
                    };
                }
            }

            let states = RegState {
                id: state_name.clone(),
                name: state_name.clone(),
                sign: Some(rate_sign),
                rate_constant: Some(rate_const.clone()),
                ..Default::default()
            };
            states_vec.insert(states.clone());

            // now to make the transition part ----------------------------------

            for (i, term) in eqns[&state].iter().enumerate() {
                if i != term_idx {
                    if term.polarity == Polarity::Positive {
                        trans_sign = true;
                    }
                    let mut state_indx = 0;
                    let mut other_state_indx = 0;
                    for (j, var) in term.vars.iter().enumerate() {
                        if state.clone() == var.clone() {
                            state_indx = j;
                        }
                        for other_states in specie_vars.clone().into_iter() {
                            if *var != state && *var == other_states {
                                // this means it is not the state, but is another state
                                other_state_indx = j;
                            }
                        }
                    }
                    for (j, var) in term.vars.iter().enumerate() {
                        if j == other_state_indx {
                            match var.clone() {
                                Var(Mi(x)) => {
                                    trans_src = x.clone();
                                }
                                _ => {
                                    println!("error in trans src extraction");
                                }
                            };
                        } else if j != other_state_indx && j != state_indx {
                            match var.clone() {
                                Var(Mi(x)) => {
                                    trans_name = x.clone();
                                }
                                _ => {
                                    println!("error in trans name extraction");
                                }
                            };
                        }
                    }
                }
            }

            let prop = Properties {
                rate_constant: trans_name.clone(),
                ..Default::default()
            };

            let transitions = RegTransition {
                id: trans_name.clone(),
                target: Some([state_name.clone()].to_vec()), // tgt
                source: Some([trans_src.clone()].to_vec()),  // src
                sign: Some(trans_sign),
                properties: Some(prop.clone()),
                ..Default::default()
            };

            transitions_vec.insert(transitions.clone());
        }

        // -----------------------------------------------------------

        let model = ModelRegNet {
            vertices: states_vec,
            edges: transitions_vec,
            parameters: None,
        };

        RegNet {
        name: "Regnet mathml model".to_string(),
        schema: "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/regnet_v0.1/regnet/regnet_schema.json".to_string(),
        schema_name: "regnet".to_string(),
        description: "This is a Regnet model from mathml equations".to_string(),
        model_version: "0.1".to_string(),
        model,
        metadata: None,
        }
    }
}

#[test]
fn test_lotka_volterra_mml_to_regnet() {
    let input: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("tests/mml2amr_input_1.json").unwrap())
            .unwrap();

    let elements: Vec<Math> = input["mathml"]
        .as_array()
        .unwrap()
        .into_iter()
        .map(|x| parse(x.as_str().unwrap()).unwrap().1)
        .collect();

    let regnet = RegNet::from(elements);

    let desired_output: RegNet =
        serde_json::from_str(&std::fs::read_to_string("tests/mml2amr_output_1.json").unwrap())
            .unwrap();

    assert_eq!(regnet, desired_output);
}
