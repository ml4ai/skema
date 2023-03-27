package org.ml4ai.skema.text_reading.grounding

/**
  * Used to return a ranked list of K grounding concepts
  *
  * @param concept instance returned by a grounder implementations
  * @param score   of the grounding algorithm given to concept
  */
case class GroundingCandidate(concept: GroundingConcept, score: Float)
