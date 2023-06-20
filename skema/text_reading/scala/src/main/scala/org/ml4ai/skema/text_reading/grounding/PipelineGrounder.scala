package org.ml4ai.skema.text_reading.grounding

import scala.collection.mutable

/**
  * Meta-grounder that allows to build pipeline compositions of grounders
  *
  * @param grounders Ordered sequence of grounders to apply
  */
class PipelineGrounder(grounders:Seq[Grounder]) extends Grounder {
  /**
    * Returns an ordered sequence with the top k grounding candidates for the input
    *
    * @param texts of the extractions to be grounded
    * @param k     number of max candidates to return
    * @return For each input element, ranked list with the top k candidates
    */
  override def groundingCandidates(texts: Seq[String], k: Int): Seq[Seq[GroundingCandidate]] = {
    // Follow a sieve-based approach, to leverage batch grounding
    sieveGrounder(texts.toArray, k, this.grounders)
  }

  private def sieveGrounder(texts:Array[String], k:Int, grounders:Seq[Grounder]):Array[Seq[GroundingCandidate]] = {
    // Apply grounding with the first element of the pipeline, then fall back to the rest if necessary
    if(texts.nonEmpty && grounders.nonEmpty){
      val firstPassGroundings = grounders.head.groundingCandidates(texts, k).toArray
      val secondPassIndices = firstPassGroundings.indices.filter(firstPassGroundings(_).isEmpty).toArray
      val secondPassTexts = secondPassIndices.map(texts)
      val secondPassGroundings = sieveGrounder(secondPassTexts, k, grounders.tail)

      // Insert the secondPassGroundings into the array of firstPassGroundings.
      secondPassIndices.zip(secondPassGroundings).foreach { case (index, grounding) =>
        firstPassGroundings(index) = grounding
      }
      firstPassGroundings
    }
    // Base cases of the recursion
    else if(grounders.isEmpty)
      Array.fill(texts.length)(Seq.empty)
    else
      Array.empty
  }
}
