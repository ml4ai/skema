package org.ml4ai.grounding

/**
 * Base trait for the all SKEMA grounding implementations
 */
trait Grounder {

  /**
   * Returns an ordered sequence with the top k grounding candidates for the input
   * @param texts of the extractions to be grounded
   * @param k number of max candidates to return
   * @return For each input element, ranked list with the top k candidates
   */
  def groundingCandidates(texts:Seq[String], k:Int): Seq[Seq[GroundingCandidate]]

  /**
    * Shortcut for single case grounding
    */
  def groundingCandidates(text:String, k:Int): Seq[GroundingCandidate] = groundingCandidates(Seq(text), k).head

  /**
   * Returns the top-ranked grounding candidate for each input element
   * @param texts of the extractions to be grounded
   * @return For each input element, some grounding concept if matched, None if didn't match any element of the ontology
   */
  def ground(texts:Seq[String]): Seq[Option[GroundingConcept]] = groundingCandidates(texts, k=1) map {
    case GroundingCandidate(topChoice, _) +: _ => Some(topChoice)
    case _ => None
  }

  /**
    * Shortcut for single case grounding
    */
  def ground(text:String): Option[GroundingConcept] = ground(Seq(text)).head

  /**
    * Shared query normalization logic for grounders
    * @param text original query
    * @return
    */
  protected def normalizeText(text:String, caseFold:Boolean):String = {
    val ret = text.trim
    if(caseFold)
      ret.toLowerCase
    else
      ret
  }

}
