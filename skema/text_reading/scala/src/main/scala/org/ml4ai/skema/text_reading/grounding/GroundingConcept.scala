package org.ml4ai.skema.text_reading.grounding

import org.json4s.JValue
import org.json4s.JsonDSL._

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

  def toJValue: JValue = {
    ("id" -> id) ~
    ("name" -> name) ~
    ("description" -> description) ~
    ("synonyms" -> synonyms.map(_.toList)) ~
    ("embedding" -> embedding.map(_.toList))
  }
}
