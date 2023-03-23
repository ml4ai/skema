package org.ml4ai.skema.text_reading.grounding

import org.clulab.dynet.Utils.newSource
import org.clulab.processors.Processor

import scala.collection.mutable
import scala.util.control.Breaks._

/**
  * Entry that can be matched to a text
  * @param lemmas which are the subject of the match
  * @param concept value to match to in case of a hit
  */
case class ManualGroundingEntry(lemmas:Array[String], concept:GroundingConcept)
class ManualGrounder(targetEntries:Iterable[ManualGroundingEntry], processor:Processor) extends Grounder {

  // Cache of previously grounded elements
  private val cache = new mutable.HashMap[String, Seq[GroundingCandidate]]()

  /**
    * Returns an ordered sequence with the top k grounding candidates for the input
    *
    * @param texts of the extractions to be grounded
    * @param k     number of max candidates to return
    * @return For each input element, ranked list with the top k candidates
    */
  override def groundingCandidates(texts: Seq[String], k: Int): Seq[Seq[GroundingCandidate]] = {
    texts map {
      query =>
        val normalizedQuery = Grounder.normalizeText(query, caseFold = true)
        cache.getOrElseUpdate(normalizedQuery, {
          // First, process the query to get the lemmas
          val lemmas = processor.annotate(normalizedQuery).sentences.head.lemmas.getOrElse(Array.empty)
          // TODO, replace this grossly inefficient code with a prefix tree/trie
          var matchingEntry: Option[ManualGroundingEntry] = None
          breakable {
            for (entry <- targetEntries) {
              if (matchingEntry.isEmpty && (lemmas sameElements entry.lemmas)) {
                matchingEntry = Some(entry)
                break
              }
            }
          }

          // Make it a single element sequence
          matchingEntry match {
            case Some(ManualGroundingEntry(_, concept)) => Seq(GroundingCandidate(concept, 1f))
            case None => Seq.empty
          }
        })

    }
  }
}

object ManualGrounder {
  /**
    * Builds a manual grounder from a collection of phrases and grounding ids in a file or resource
    * @param path to the file or resource with the grounding phrases
    * @param processor used to lemmatize the concepts
    * @return instance of ManualGrounder prepared to use
    */
  def fromFileOrResource(path:String, processor: Processor): ManualGrounder = {
    val src = newSource(path)
    val entries =
      src.getLines() map {
        line =>
          val tokens = Grounder.normalizeText(line, caseFold = true).split("\t")
          if(tokens.size >= 2) {
            val (phrase, id, desc) = (tokens.head, tokens(1), tokens.lift(2))
            val lemmas = processor.annotate(phrase).sentences.head.lemmas.get
            Some(ManualGroundingEntry(lemmas, GroundingConcept(id, phrase, desc, None, None)))
          }
          else
            None
      } collect { case Some(e) => e}
    new ManualGrounder(entries.toList, processor)
  }
}
