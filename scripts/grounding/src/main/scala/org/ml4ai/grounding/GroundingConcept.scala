package org.ml4ai.grounding

/**
 * Ontology concept that serves as grounding element for extractions
 * @param id Unique identifier of the concept
 * @param name of the concept
 * @param description of the concept
 * @param synonyms Optional list of synonyms
 */
case class GroundingConcept(id:String, name:String, description:Option[String], synonyms:Option[Seq[String]], embedding:Option[Array[Float]]) {
  override def equals( arg:Any): Boolean = arg match {
    case other:GroundingConcept => other.id == this.id
    case _ => false
  }
  override def hashCode(): Int = this.id.hashCode
}

/**
 * Used to return a ranked list of K grounding concepts
 * @param concept instance returned by a grounder implementations
 * @param score of the grounding algorithm given to concept
 */
case class GroundingCandidate(concept:GroundingConcept, score:Float)