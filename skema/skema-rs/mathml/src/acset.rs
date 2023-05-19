//! Structs to represent elements of ACSets (Annotated C-Sets, a concept from category theory).
//! JSON-serialized ACSets are the form of model exchange between TA1 and TA2.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
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
#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelRepPn {
    pub name: String,
    pub schema: String,
    pub description: String,
    pub model_version: String,
    pub model: Model,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct Model {
    pub states: Vec<State>,
    pub transitions: Vec<Transition>,

    /// Note: parameters is a required field in the schema, but we make it optional since we want
    /// to reuse this schema for partial extractions as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Parameter>>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct State {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_constant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sign: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial: Option<Initial>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct Grounding {
    pub identifiers: serde_json::Map<String, Value>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Initial {
    pub expression: String,
    pub expression_mathml: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Rate {
    pub expression: String,
    pub expression_mathml: String,
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
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

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct Properties {
    pub name: String,
    pub grounding: Option<Grounding>,
    pub rate: Rate,
    pub rate_constant: Option<String>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct Parameter {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution: Option<Distribution>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Serialize, Deserialize, ToSchema)]
pub struct Distribution {
    #[serde(rename = "type")]
    pub r#type: String,
    pub parameters: serde_json::Map<String, Value>,
}

// -------------------------------------------------------------------------------------------
// This function takes our previous model form, the ACSet and transforms it to the new TA4 exchange format
// -------------------------------------------------------------------------------------------
impl ModelRepPn {
    pub fn from(pn: ACSet) -> ModelRepPn {
        let mut states_vec = Vec::<State>::new();
        let mut transitions_vec = Vec::<Transition>::new();

        // -----------------------------------------------------------

        for state in pn.S.iter() {
            let states = State {
                id: state.sname.clone(),
                name: state.sname.clone(),
                ..Default::default()
            };
            states_vec.push(states.clone());
        }

        for (i, trans) in pn.T.clone().iter().enumerate() {
            // convert transition index to base 1
            let transition = i.clone() + 1;

            // construct array of incoming states
            let mut string_vec1 = Vec::<String>::new();
            for incoming in pn.I.clone().iter() {
                if incoming.it.clone() == transition {
                    let state_idx = incoming.is.clone() - 1; // 0 based to index the rust vec
                    let state_name = pn.S[state_idx as usize].sname.clone();
                    string_vec1.push(state_name.clone());
                }
            }
            // construct array of outgoing states
            let mut string_vec2 = Vec::<String>::new();
            for outgoing in pn.O.clone().iter() {
                if outgoing.ot.clone() == transition {
                    let state_idx = outgoing.os.clone() - 1; // 0 based to index the rust vec
                    let state_name = pn.S[state_idx as usize].sname.clone();
                    string_vec2.push(state_name.clone());
                }
            }

            let transitions = Transition {
                id: trans.tname.clone(),
                input: Some(string_vec1.clone()),
                output: Some(string_vec2.clone()),
                ..Default::default()
            };

            transitions_vec.push(transitions.clone());
        }

        // -----------------------------------------------------------

        let model = Model {
            states: states_vec,
            transitions: transitions_vec,
            ..Default::default()
        };

        let mrp = ModelRepPn {
        name: "mathml model".to_string(),
        schema: "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json".to_string(),
        description: "This is a model from mathml equations".to_string(),
        model_version: "0.1".to_string(),
        model: model.clone(),
        ..Default::default()
    };

        return mrp;
    }
}
