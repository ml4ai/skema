package org.ml4ai.grounding

import org.clulab.embeddings.{ExplicitWordEmbeddingMap, WordEmbeddingMap}
import ujson.Arr

import java.io.File

class MiraEmbeddingsGrounder(groundingConcepts:IndexedSeq[GroundingConcept], embeddingsModel:WordEmbeddingMap) extends Grounder {

  /**
   * Returns an ordered sequence with the top k grounding candidates for the input
   *
   * @param text of the extraction to be grounded
   * @param k    number of max candidates to return
   * @return ranked list with the top k candidates
   */
  override def groundingCandidates(text: String, k: Int = 1): Seq[GroundingCandidate] = ??? // TODO Sushma, fill in this method
}


object MiraEmbeddingsGrounder{

  /**
   * Parse the MIRA json file into our indexed sequence of GroundingConcepts
   * @param path to the json file with the ontology
   * @return sequence with the in-memory concepts
   */
  def parseMiraJson(path:File): IndexedSeq[GroundingConcept] = {
    ujson.read(path) match {
      case Arr(value) => value.map{
        i =>
          GroundingConcept(
            i("id").str,
            i("name").str,
            i.obj.get("description") match {
              case Some(desc) => Some(desc.str)
              case _ => None
            },
            i.obj.get("synonyms") match {
              case Some(syns) => Some(syns.arr map (_.str))
              case _ => None
            }
          )
      }
    }
  }

  /**
   * Loads a specific word embeddings model from disk
   * @param path to the file containing the embeddings
   * @return a WordEmbeddingMap instance
   */
  def loadWordEmbeddings(path:File): WordEmbeddingMap = ExplicitWordEmbeddingMap(path.getPath, resource = false)


  /**
   * Instantiate the grounder by passing paths to the data files
   * @param miraJsonFile file containing the json file with MIRA concepts
   * @param wordEmbeddingsFile file containing the word embedding model
   */
  def apply(miraJsonFile:File, wordEmbeddingsFile:File) =
    new MiraEmbeddingsGrounder(parseMiraJson(miraJsonFile), loadWordEmbeddings(wordEmbeddingsFile))
}
