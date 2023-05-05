//! Structs to represent elements of ACSets (Annotated C-Sets, a concept from category theory).
//! JSON-serialized ACSets are the form of model exchange between TA1 and TA2.
use serde::{Deserialize, Serialize};
use utoipa;
use utoipa::ToSchema;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema)]
pub struct Specie {
    pub sname: String,
    pub uid: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema)]
pub struct Transition {
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
    pub T: Vec<Transition>,
    pub I: Vec<InputArc>,
    pub O: Vec<OutputArc>,
}

// -------------------------------------------------------------------------------------------
// The following data structs are those requested by TA-4 as an exchange format for the models.
// the spec in json format can be found here: https://github.com/DARPA-ASKEM/Model-Representations/blob/main/petrinet/petrinet_schema.json
#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct ModelRepPn {
    pub name: String,
    pub schema: String,
    pub description: String,
    pub model_verison: String,
    pub model: Model,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadatas>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Model {
    pub states: Vec<States>,
    pub transitions: Vec<Transitions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<Parameters>>,
}
#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct States {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial: Option<Initial>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Grounding {
    pub identifiers: Identifier,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Identifier {
    pub ido: i64,
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
pub struct Transitions {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Properties>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Properties {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<Initial>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Parameters {
    pub id: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding: Option<Grounding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution: Option<Distribution>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Distribution {
    #[serde(rename = "type")]
    pub r#type: String,
    pub parameters: ParametersD,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct ParametersD {
    minimum: i64,
    maximum: i64,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Metadatas {
    pub processed_at: i64,
    pub processed_by: String,
}

// -------------------------------------------------------------------------------------------
// This function takes our previous model form, the ACSet and transforms it to the new TA4 exchange format
// -------------------------------------------------------------------------------------------

    let mut states_vec = Vec::<States>::new();
    let mut transitions_vec = Vec::<Transitions>::new();

        // -----------------------------------------------------------

    for state in pn.S.iter() {
        let states = States {
            id: state.sname.clone(),
            name: state.sname.clone(),
            grounding: None,
            initial: None,
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

        let transitions = Transitions {
            id: trans.tname.clone(),
            input: Some(string_vec1.clone()),
            output: Some(string_vec2.clone()),
            properties: None,
            model,
        };

        transitions_vec.push(transitions.clone());
    }

    // -----------------------------------------------------------

    let model = Model {
        states: states_vec,
        transitions: transitions_vec,
        parameters: None,
    };

    let mrp = ModelRepPn {
        name: "mathml model".to_string(),
        schema: "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json".to_string(),
        description: "This is a model from mathml equations".to_string(),
        model_verison: "0.1".to_string(),
        model: model.clone(),
        metadata: None,
    };

    return mrp;
}
