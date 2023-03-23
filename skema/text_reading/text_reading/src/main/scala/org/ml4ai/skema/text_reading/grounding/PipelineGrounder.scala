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
    sieveGrounding(texts.toArray, k, this.grounders)
  }

  private def sieveGrounding(texts:Array[String], k:Int, grounders:Seq[Grounder]):Array[Seq[GroundingCandidate]] = {
    // Apply grounding with the first element of the pipeline, then fall back to the rest if necessary
    if(texts.nonEmpty && grounders.nonEmpty){
      val grounder = grounders.head
      val initialGroundings = new mutable.Queue[Seq[GroundingCandidate]] ++= grounder.groundingCandidates(texts, k)      // Collect the elements that couldn't be grounded
      val (missingTexts, missingIndices) = ((texts zip initialGroundings).zipWithIndex collect {
        case ((query, groundings), ix) if groundings.isEmpty =>  (query, ix)
      }).unzip
      // Fall back to the next grounders
      val subsequentGroundings = new mutable.Queue[Seq[GroundingCandidate]] ++= sieveGrounding(missingTexts, k, grounders.tail)
      // Merge the results
      val missingIndicesSet = missingIndices.toSet
      (texts.indices map {
        ix =>
          if(missingIndicesSet contains ix) {
            initialGroundings.dequeue() // Necessary to consume the empty element
            subsequentGroundings.dequeue()
          } else
            initialGroundings.dequeue()
      }).toArray
    }
    // Base cases of the recursion
    else if(grounders.isEmpty)
      Array.fill(texts.length)(Seq.empty)
    else
      Array.empty
  }
}
