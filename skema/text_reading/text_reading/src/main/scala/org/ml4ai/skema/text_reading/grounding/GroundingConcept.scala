package org.ml4ai.skema.text_reading.grounding

/**
  * Ontology concept that serves as grounding element for extractions
  *
  * @param id          Unique identifier of the concept
  * @param name        of the concept
  * @param description of the concept
  * @param synonyms    Optional list of synonyms
  */
case class GroundingConcept(id: String, name: String, description: Option[String], synonyms: Option[Seq[String]], embedding: Option[Array[Float]]) {
  override def equals(arg: Any): Boolean = arg match {
    case other: GroundingConcept => other.id == this.id
    case _ => false
  }

  override def hashCode(): Int = this.id.hashCode
}
