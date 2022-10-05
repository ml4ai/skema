package org.ml4ai.grounding

/**
 * Base trait for the all SKEMA grounding implementations
 */
trait Grounder {

  /**
   * Returns an ordered sequence with the top k grounding candidates for the input
   * @param text of the extraction to be grounded
   * @param k number of max candidates to return
   * @return ranked list with the top k candidates
   */
  def groundingCandidates(text:String, k:Int = 1): List[GroundingCandidate]

  /**
   * Returns the top-ranked grounding candidate
   * @param text of the extraction to be grounded
   * @return Some grounding concept if matched, None if didn't match any element of the ontology
   */
  def ground(text:String): Option[GroundingConcept] = groundingCandidates(text) match {
    case GroundingCandidate(topChoice, _)::_ => {
      val x = 0
      Some(topChoice)
    }
    case y => {
      None
    }
  }
}
