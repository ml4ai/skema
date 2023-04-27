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
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub r#type: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<StringType>,
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
    items: StateItems,
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
    items: TransitionItems,
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
