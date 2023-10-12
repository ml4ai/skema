package org.ml4ai.skema.text_reading.grounding

import org.clulab.processors.Processor

import java.io.{File, FileNotFoundException}
import scala.collection.mutable
import scala.io.Source
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

  def newSource(filename: String): Source = {
    val f = new File(filename)
    if (f.exists()) {
      // This file exists on disk.
      Source.fromFile(filename, "UTF-8")
    } else {
      // The file does not exist on disk.  Let's hope it's in the classpath.
      // This should work for both scala 2.11 and 2.12.
      // The resource will be null if it isn't found, so use an Option!
      val inputStreamOpt = Option(getClass.getResourceAsStream("/" + filename))
      val sourceOpt = inputStreamOpt.map(Source.fromInputStream)
      // This only works for scala 2.12, so we can't cross compile with 2.11.
      // Source.fromResource(filename)
      sourceOpt.getOrElse(throw new FileNotFoundException(s"""Could not find resource "$filename"."""))
    }
  }

  /**
    * Builds a manual grounder from a collection of phrases and grounding ids in a file or resource
    * @param path to the file or resource with the grounding phrases
    * @param processor used to lemmatize the concepts
    * @return instance of ManualGrounder prepared to use
    */
  def fromFileOrResource(path:String, processor: Processor): ManualGrounder = {
    val src = newSource(path)
    val entries =
      src.getLines() flatMap {
        line =>
          val tokens = Grounder.normalizeText(line, caseFold = true).split("\t")
          if(tokens.size >= 2) {
            val (phrase, id, desc) = (tokens.head, tokens(1), tokens.lift(2))
            val lemmas = processor.annotate(phrase).sentences.head.lemmas.get
            Some(ManualGroundingEntry(lemmas, GroundingConcept(id, phrase, desc, None, None)))
          }
          else
            None
      }
    new ManualGrounder(entries.toList, processor)
  }
}
