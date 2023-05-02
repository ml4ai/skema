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
    pub uid: usize,
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
    #[serde(rename = "$schema")]
    pub schema: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub properties: Properties,
    #[serde(rename = "$defs", skip_serializing_if = "Option::is_none")]
    pub defs: Option<Defs>,
    #[serde(rename = "additionalProperties")]
    pub additionalproperties: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct StringType {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct ArrayType {
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<StringType>>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Ref {
    #[serde(rename = "$ref")]
    pub r#ref: String,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct NumType {
    pub r#type: i32,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Model {
    #[serde(rename = "type")]
    r#type: String,
    properties: ModelProperties,
    required: Option<Vec<String>>,
    #[serde(rename = "additionalProperties")]
    additionalproperties: Option<bool>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct ModelProperties {
    #[serde(skip_serializing_if = "Option::is_none")]
    states: Option<States>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transitions: Option<Transitions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Parameters>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct States {
    #[serde(rename = "type")]
    r#type: String,
    items: Vec<StateItems>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct TransitionItems {
    #[serde(rename = "type")]
    r#type: String,
    properties: TransitionProperties,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct ParameterItems {
    #[serde(rename = "type")]
    r#type: String,
    properties: ParameterProperties,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct StateItems {
    #[serde(rename = "type")]
    r#type: String,
    properties: StateProperties,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct StateProperties {
    id: StringType,
    name: StringType,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounding: Option<Ref>,
    #[serde(skip_serializing_if = "Option::is_none")]
    initial: Option<Ref>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Transitions {
    #[serde(rename = "type")]
    r#type: String,
    items: Vec<TransitionItems>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct TransitionProperties {
    id: StringType,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<ArrayType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<ArrayType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounding: Option<Ref>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<Ref>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Parameters {
    #[serde(rename = "type")]
    r#type: String,
    items: ParameterItems,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct ParameterProperties {
    id: StringType,
    description: StringType,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<NumType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounding: Option<Ref>,
    #[serde(skip_serializing_if = "Option::is_none")]
    distribution: Option<Ref>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Properties {
    name: StringType,
    schema: StringType,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<StringType>,
    model_version: StringType,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<StringType>,
    model: Model,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Defs {
    #[serde(skip_serializing_if = "Option::is_none")]
    initial: Option<Initial>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<DefProperties>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate: Option<Rate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    distribution: Option<Distribution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grounding: Option<Groundings>,
    #[serde(
        rename = "additionalProperties",
        skip_serializing_if = "Option::is_none"
    )]
    additionalproperties: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct DefProperties {
    #[serde(rename = "type")]
    r#type: String,
    properties: DefPropertiesProperties,
    required: Vec<String>,
    #[serde(rename = "additionalProperties")]
    additionalproperties: bool,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Initial {
    #[serde(rename = "type")]
    r#type: String,
    properties: InitialProperties,
    required: Vec<String>,
    #[serde(rename = "additionalProperties")]
    additionalproperties: bool,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Rate {
    #[serde(rename = "type")]
    r#type: String,
    properties: InitialProperties,
    required: Vec<String>,
    #[serde(rename = "additionalProperties")]
    additionalproperties: bool,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Distribution {
    #[serde(rename = "type")]
    r#type: String,
    properties: DistributionProperties,
    required: Vec<String>,
    #[serde(rename = "additionalProperties")]
    additionalproperties: bool,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct Groundings {
    #[serde(rename = "type")]
    r#type: String,
    properties: GroundingProperties,
    required: Vec<String>,
    #[serde(rename = "additionalProperties")]
    additionalproperties: bool,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct InitialProperties {
    expression: StringType,
    expression_mathml: StringType,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct DefPropertiesProperties {
    name: StringType,
    grounding: Ref,
    rate: Ref,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct DistributionProperties {
    #[serde(rename = "type")]
    r#type: StringType,
    parameters: StringType,
}

#[derive(
    Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, ToSchema,
)]
pub struct GroundingProperties {
    identifiers: StringType,
    context: StringType,
}

// -------------------------------------------------------------------------------------------
// This function takes our previous model form, the ACSet and transforms it to the new TA4 exchange format
pub fn PN_to_ModelRepPN(pn: ACSet) -> ModelRepPn {

    // -----------------------------------------------------

    let mut state_items_vec = Vec::<StateItems>::new();
    let mut transition_items_vec = Vec::<TransitionItems>::new();
    let mut string_type_vec = Vec::<StringType>::new();

    // -----------------------------------------------------

    for state in pn.S.iter() {
        let string_type1 = StringType {
            r#type: Some(state.sname.clone()),
            format: None,
        };
        let string_type2 = StringType {
            r#type: Some(state.sname.clone()),
            format: None,
        };
        let state_properties = StateProperties {
            id: string_type1.clone(),
            name: string_type2.clone(),
            grounding: None,
            initial: None,
        };
        let state_items = StateItems {
            r#type: "object".to_string(),
            properties: state_properties.clone(),
            required: None,
        };
        state_items_vec.push(state_items.clone());
    }

    for (i, trans) in pn.T.clone().iter().enumerate() {

        // convert transition index to base 1
        let transition = i.clone() + 1; 

        let string_type1 = StringType {
            r#type: Some(trans.tname.clone()),
            format: None,
        };
        // construct array of incoming states
        let mut string_type_vec1 = Vec::<StringType>::new();
        for incoming in pn.I.clone().iter() {
            if incoming.it.clone() == transition {
                let state_idx = incoming.is.clone() - 1; // 0 based to index the rust vec
                let state_name = pn.S[state_idx as usize].sname.clone();
                let string_type_temp = StringType {
                    r#type: Some(state_name.clone()),
                    format: None,
                };
                string_type_vec1.push(string_type_temp.clone());
            }
        }
        let array_type1 = ArrayType {
            pub r#type: "array".to_string(),
            pub items: Some(string_type_vec.clone()),
        };
        // construct array of outgoing states
        let mut string_type_vec2 = Vec::<StringType>::new();
        for outgoing in pn.O.clone().iter() {
            if outgoing.ot.clone() == transition {
                let state_idx = outgoing.os.clone() - 1; // 0 based to index the rust vec
                let state_name = pn.S[state_idx as usize].sname.clone();
                let string_type_temp = StringType {
                    r#type: Some(state_name.clone()),
                    format: None,
                };
                string_type_vec2.push(string_type_temp.clone());
            }
        }
        let array_type2 = ArrayType {
            pub r#type: "array".to_string(),
            pub items: Some(string_type_vec2.clone()),
        };
        let transition_properties = TransitionProperties {
            id: string_type1,
            input: Some(array_type1),
            output: Some(array_type2),
            grounding: None,
            properties: None,
        };
        let transition_items = TransitionItems {
            r#type: "object".to_string(),
            properties: transition_properties,
            required: None,
        };
        transition_items_vec.push(transition_items.clone());
    }

    // -----------------------------------------------------------

    let states = States {
        r#type: "array".to_string(),
        items: state_items_vec,
    };

    let transtitions = Transitions {
        r#type: "array".to_string(),
        items: transition_items_vec,
    };

    let model_properties = ModelProperties {
        states: Some(states),
        transitions: Some(transitions),
        parameters: None,
    };

    let model = Model {
        r#type: "object".to_string(),
        properties: model_properties,
        required: None,
        additionalproperties: None,
    };

    let properties = Properties {
        name: "PetriNet".to_string(),
        schema: "PertriNet".to_string(),
        description: None,
        model_version: "PetriNet".to_string(),
        properties: None,
        model: model,
    }; 

    let mrp =  ModelRepPn {
        schema: "PetriNet".to_string(),
        r#type: "PetriNet".to_string(),
        properties: properties,
        defs: None,
        additionalproperties: false,
        required: None,
    };

    return mrp;
}
