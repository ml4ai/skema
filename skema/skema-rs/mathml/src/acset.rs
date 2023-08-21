//! Structs to represent elements of ACSets (Annotated C-Sets, a concept from category theory).
//! JSON-serialized ACSets are the form of model exchange between TA1 and TA2.
use crate::parsers::first_order_ode::{get_terms, FirstOrderODE, PnTerm};
use crate::{
    ast::{Math, MathExpression, Mi},
    mml2pn::{group_by_operators, Term},
    petri_net::{Polarity, Var},
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
    pub semantics: Option<Semantics>,
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
    pub expression_mathml: Option<String>,
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

    #[serde(skip_serializing_if = "Option::is_none")]
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

    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Properties>,
}

#[derive(
    Debug, Default, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
pub struct Properties {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_constant: Option<String>,
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
        let mut initial_vec = Vec::<Initial>::new();
        let mut parameter_vec = Vec::<Parameter>::new();
        let mut rate_vec = Vec::<Rate>::new();

        // -----------------------------------------------------------

        for state in pn.S.iter() {
            let states = State {
                id: state.sname.clone(),
                name: state.sname.clone(),
                ..Default::default()
            };
            let initials = Initial {
                target: state.sname.clone(),
                expression: format!("{}0", state.sname.clone()),
                ..Default::default()
            };
            let parameters = Parameter {
                id: initials.expression.clone(),
                name: Some(initials.expression.clone()),
                description: Some(format!(
                    "The total {} population at timestep 0",
                    state.sname.clone()
                )),
                ..Default::default()
            };
            parameter_vec.push(parameters.clone());
            initial_vec.push(initials.clone());
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
                id: format!("t{}", i.clone()),
                input: Some(string_vec1.clone()),
                output: Some(string_vec2.clone()),
                ..Default::default()
            };
            let parameters = Parameter {
                id: trans.tname.clone(),
                name: Some(trans.tname.clone()),
                description: Some(format!("{} rate", trans.tname.clone())),
                ..Default::default()
            };

            let mut terms = String::new();
            for term in transitions.input.clone().unwrap() {
                let terms_temp = terms.clone();
                terms = format!("{}*{}", terms_temp.clone(), term.clone());
            }

            let rate = Rate {
                target: format!("t{}", i.clone()),
                expression: format!("{}{}", trans.tname.clone(), terms.clone()), // the second term needs to be the product of the inputs
                ..Default::default()
            };

            rate_vec.push(rate.clone());
            parameter_vec.push(parameters.clone());
            transitions_vec.insert(transitions.clone());
        }

        // -----------------------------------------------------------

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

        PetriNet {
        name: "mathml model".to_string(),
        schema: "https://github.com/DARPA-ASKEM/Model-Representations/blob/main/petrinet/petrinet_schema.json".to_string(),
        schema_name: "PetriNet".to_string(),
        description: "This is a model from mathml equations".to_string(),
        model_version: "0.1".to_string(),
        model,
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

        // this first for loop is for the creation state related parameters in the AMR
        for ode in ode_vec.iter() {
            let states = State {
                id: ode.lhs_var.to_string().clone(),
                name: ode.lhs_var.to_string().clone(),
                ..Default::default()
            };
            let initials = Initial {
                target: ode.lhs_var.to_string().clone(),
                expression: format!("{}0", ode.lhs_var.to_string().clone()),
                ..Default::default()
            };
            let parameters = Parameter {
                id: initials.expression.clone(),
                name: Some(initials.expression.clone()),
                description: Some(format!(
                    "The total {} population at timestep 0",
                    ode.lhs_var.to_string().clone()
                )),
                ..Default::default()
            };
            parameter_vec.push(parameters.clone());
            initial_vec.push(initials.clone());
            states_vec.insert(states.clone());
            state_string_list.push(ode.lhs_var.to_string().clone()); // used later for transition parsing
        }

        // now for the construction of the transitions and their results

        // this collects all the terms from the equations
        for ode in ode_vec.iter() {
            terms.append(&mut get_terms(state_string_list.clone(), ode.clone()));
        }

        for term in terms.iter() {
            println!("term: {:?}\n", term.clone());
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

        // first is we need to replace any terms with more than 2 exp_states with subterms, these are simply
        // terms that need to be distributed (ASSUMPTION, MOST A TRANSTION CAN HAVE IS 2 IN AND 2 OUT)
        // but first we need to inherit the dynamic state to each sub term
        let mut composite_term_ind = Vec::<usize>::new();
        let mut sub_terms = Vec::<PnTerm>::new();
        for (j, t) in terms.clone().iter().enumerate() {
            if t.exp_states.len() > 2 {
                if t.sub_terms.is_some() {
                    for (i, _sub_t) in t.sub_terms.clone().unwrap().iter().enumerate() {
                        terms[j].sub_terms.as_mut().unwrap()[i].dyn_state = t.dyn_state.clone();
                        sub_terms.push(terms[j].sub_terms.as_mut().unwrap()[i].clone());
                    }
                    composite_term_ind.push(j);
                }
            }
        }
        // delete composite terms we are replacing
        composite_term_ind.sort();
        for i in composite_term_ind.iter().rev() {
            terms.remove(*i);
        }
        // replace with subterms
        terms.append(&mut sub_terms);

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

        // all that should be left are unpaired terms
        for t in terms.clone().iter() {
            println!("\nunpaired t: {:?}\n", t.clone());
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

                let mut expression_string = "".to_string();

                for param in t.0.parameters.clone().iter() {
                    expression_string = format!("{}{}*", expression_string.clone(), param.clone());
                }

                let exp_len = t.0.exp_states.len();
                for (i, exp) in t.0.exp_states.clone().iter().enumerate() {
                    if i != exp_len - 1 {
                        expression_string =
                            format!("{}{}*", expression_string.clone(), exp.clone());
                    } else {
                        expression_string = format!("{}{}", expression_string.clone(), exp.clone());
                    }
                }

                let rate = Rate {
                    target: transitions.id.clone(),
                    expression: expression_string.clone(), // the second term needs to be the product of the inputs
                    expression_mathml: Some(t.0.expression.clone()),
                };
                rate_vec.push(rate.clone());
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

                let transitions = Transition {
                    id: format!("t{}", i.clone()),
                    input: Some(t.1.exp_states.clone()),
                    output: Some(output.clone()),
                    ..Default::default()
                };
                transitions_vec.insert(transitions.clone());

                let mut expression_string = "".to_string();

                for param in t.0.parameters.clone().iter() {
                    expression_string = format!("{}{}*", expression_string.clone(), param.clone());
                }

                let exp_len = t.0.exp_states.len() - 1;
                for (i, exp) in t.0.exp_states.clone().iter().enumerate() {
                    if i != exp_len {
                        expression_string =
                            format!("{}{}*", expression_string.clone(), exp.clone());
                    } else {
                        expression_string = format!("{}{}", expression_string.clone(), exp.clone());
                    }
                }

                let rate = Rate {
                    target: transitions.id.clone(),
                    expression: expression_string.clone(), // the second term needs to be the product of the inputs
                    expression_mathml: Some(t.0.expression.clone()),
                };
                rate_vec.push(rate.clone());
            }
        }

        // now we construct transitions from unpaired terms, assuming them to be sources and sinks

        // trim duplicate parameters and (TODO)remove integer parameters

        parameter_vec.sort();
        parameter_vec.dedup();

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

        PetriNet {
        name: "mathml model".to_string(),
        schema: "https://github.com/DARPA-ASKEM/Model-Representations/blob/main/petrinet/petrinet_schema.json".to_string(),
        schema_name: "PetriNet".to_string(),
        description: "This is a model from mathml equations".to_string(),
        model_version: "0.1".to_string(),
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
                if term.exp_states.len() == 1 && term.exp_states[0] == term.dyn_state {
                    // note this is only grabbing one term. This is somewhat limited by the current AMR schema
                    // it assumes only a simple single parameter for this Date: 08/10/23
                    r_state.rate_constant = Some(term.parameters[0].clone());
                    r_state.sign = Some(term.polarity);
                    // This adds the edges for the environment couplings
                    let prop = Properties {
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
                    transitions_vec.insert(self_trans.clone());
                    counter += 1;
                }
            }
            // This adds the intial values from the state variables into the parameters vec
            let parameters = Parameter {
                id: state.clone(),
                name: Some(state.clone()),
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
        for term1 in terms.clone().iter() {
            for term2 in terms.clone().iter() {
                if term1.polarity != term2.polarity
                    && term1.parameters == term2.parameters
                    && term1.polarity
                {
                    // first term is positive, second is negative
                    let temp_pair = (term1.clone(), term2.clone());
                    transition_pair.push(temp_pair);
                }
            }
        }

        for (i, t) in transition_pair.iter().enumerate() {
            if t.0.exp_states.len() == 1 {
                // construct transtions for simple transtions
                let prop = Properties {
                    // once again the assumption of only one parameters for transition
                    name: t.0.parameters[0].clone(),
                    rate_constant: None,
                };
                let trans = RegTransition {
                    id: format!("t{}", i.clone()),
                    source: Some([t.1.dyn_state.clone()].to_vec()),
                    target: Some([t.0.dyn_state.clone()].to_vec()),
                    sign: Some(true),
                    grounding: None,
                    properties: Some(prop.clone()),
                };
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

                let _transitions = Transition {
                    id: format!("t{}", i.clone()),
                    input: Some(t.1.exp_states.clone()),
                    output: Some(output.clone()),
                    ..Default::default()
                };
                let prop = Properties {
                    // once again the assumption of only one parameters for transition
                    name: t.0.parameters[0].clone(),
                    rate_constant: None,
                };
                let trans = RegTransition {
                    id: format!("t{}", i.clone()),
                    source: Some(t.1.exp_states.clone()),
                    target: Some(output.clone()),
                    sign: Some(true),
                    grounding: None,
                    properties: Some(prop.clone()),
                };
                transitions_vec.insert(trans.clone());
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

        // ------------------------------------------

        let model = ModelRegNet {
            vertices: states_vec,
            edges: transitions_vec,
            parameters: Some(parameter_vec),
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

            // Positive rate sign: source, negative => sink.
            if eqns[&state.clone()][term_idx].polarity == Polarity::Positive {
                rate_sign = true;
            }

            for variable in eqns[&state][term_idx].vars.iter() {
                if state.clone() != variable.clone() {
                    match variable.clone() {
                        Var(MathExpression::Mi(Mi(x))) => {
                            rate_const = x.clone();
                        }
                        _ => {
                            println!("Error in rate extraction");
                        }
                    };
                } else {
                    match variable.clone() {
                        Var(MathExpression::Mi(Mi(x))) => {
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
                                Var(MathExpression::Mi(Mi(x))) => {
                                    trans_src = x.clone();
                                }
                                _ => {
                                    println!("error in trans src extraction");
                                }
                            };
                        } else if j != other_state_indx && j != state_indx {
                            match var.clone() {
                                Var(MathExpression::Mi(Mi(x))) => {
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
                name: trans_name.clone(),
                rate_constant: Some(trans_name.clone()),
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

/*#[test]
fn test_lotka_volterra_mml_to_regnet() {
    let input: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("tests/mml2amr_input_1.json").unwrap())
            .unwrap();

    let elements: Vec<Math> = input["mathml"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().parse::<Math>().unwrap())
        .collect();

    let regnet = RegNet::from(elements);

    let desired_output: RegNet =
        serde_json::from_str(&std::fs::read_to_string("tests/mml2amr_output_1.json").unwrap())
            .unwrap();

    assert_eq!(regnet, desired_output);
}*/
