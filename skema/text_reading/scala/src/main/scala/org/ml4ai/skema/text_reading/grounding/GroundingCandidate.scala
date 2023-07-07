package org.ml4ai.skema.text_reading.grounding

import org.json4s.JValue
import org.json4s.JsonDSL._

/**
  * Used to return a ranked list of K grounding concepts
  *
  * @param concept instance returned by a grounder implementations
  * @param score   of the grounding algorithm given to concept
  */
case class GroundingCandidate(concept: GroundingConcept, score: Float) {

  def toJValue: JValue = {
    ("groundingConcept" -> concept.toJValue) ~
    ("score" -> score)
  }
}
