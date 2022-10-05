package org.ml4ai.grounding

import org.clulab.embeddings.{ExplicitWordEmbeddingMap, WordEmbeddingMap}
import org.clulab.processors.clu.CluProcessor
import org.clulab.utils.Serializer
import org.ml4ai.grounding.MiraEmbeddingsGrounder.generateEmbedding
import ujson.Arr

import java.io.File
import scala.::

class MiraEmbeddingsGrounder(groundingConcepts:Seq[GroundingConcept], embeddingsModel:WordEmbeddingMap) extends Grounder {

  /**
   * Returns an ordered sequence with the top k grounding candidates for the input
   *
   * @param text of the extraction to be grounded
   * @param k    number of max candidates to return
   * @return ranked list with the top k candidates
   */
  override def groundingCandidates(text: String, k: Int = 1): Seq[GroundingCandidate] = {
    // Generate the embedding of text
    val thisTextEmbedding = generateEmbedding(text, embeddingsModel)
    // Loop over the grounding concepts and get cosine similarity between input embedding and each concept
    val cosineSimilarities =
    for (groundingConcept <- groundingConcepts)  yield {
      // clone groundingConcept's embedding and normalize it
      // normalize text embedding
      // compute dot procuct
      val normalizedConceptEmbedding = groundingConcept.embedding.get.clone()
      WordEmbeddingMap.norm(normalizedConceptEmbedding)
      WordEmbeddingMap.norm(thisTextEmbedding)
      WordEmbeddingMap.dotProduct(normalizedConceptEmbedding, thisTextEmbedding)
    }
    // Choose the top k and return GroundingCandidates
    // val (addSorted, indices) = arr.zipWithIndex.sorted.unzip
    val (sortedCosineSimilarities, sortedIndices) = cosineSimilarities.zipWithIndex.sorted.reverse.unzip
//    sortedCosineSimilarities[sortedIndices[1:k]]

    val topKindices = sortedIndices.take(k)
    val topSimilarities = sortedCosineSimilarities.take(k)
    val topConcepts = topKindices.map(groundingConcepts)
    topConcepts.zip(topSimilarities).map{case (concept, similarity) => GroundingCandidate(concept, similarity)}
  }
}


object MiraEmbeddingsGrounder{

  val processor = new CluProcessor()
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
            },
            embedding = None
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
   * @param ontologyFile file containing the json file with MIRA concepts
   * @param wordEmbeddingsFile file containing the word embedding model
   */
  def apply(ontologyFile:File, wordEmbeddingsFile:File) = {

    val embeddingsModel = loadWordEmbeddings(wordEmbeddingsFile)

    val ontology =
      if(ontologyFile.getName.endsWith(".ser")){
        Serializer.load[Seq[GroundingConcept]](ontologyFile)
      }
      else{
        // If this was not ser, assume it is json
        val ontology = parseMiraJson(ontologyFile)
        val ontologyWithEmbeddings = createOntologyEmbeddings(ontology, embeddingsModel)
        // TODO save serialized file
        val newFileName = ontologyFile.getAbsolutePath + ".ser"
        Serializer.save(ontologyWithEmbeddings, newFileName)
        ontologyWithEmbeddings
      }


    new MiraEmbeddingsGrounder(ontology, embeddingsModel)
  }

  def averageEmbeddings(wordEmbeddings: Array[Array[Float]]): Array[Float] = {
    val size = wordEmbeddings.size
    val embeddingsSum = wordEmbeddings.reduce((a, b) => a.zip(b).map(x => x._1 + x._2))
    embeddingsSum.map(_ / size)
  }

  def generateEmbedding(name: String, embeddings: WordEmbeddingMap): Array[Float] = {
    // Tokenize the string
    val tokens = processor.mkDocument(name).sentences.head.words

    val OOV = Array.fill(embeddings.dim)(0.0f)

    val wordEmbeddings = tokens.map(embeddings.get).map {
      case Some(e) => e.toArray
      case None => OOV
    }

    averageEmbeddings(wordEmbeddings)
  }

  def addEmbeddingToConcept(concept: GroundingConcept, embeddingsModel: WordEmbeddingMap): GroundingConcept = {


    val embedding = generateEmbedding(concept.name, embeddingsModel)

    val descEmbeddings = concept.description match {
      case Some(description) => List(generateEmbedding(description, embeddingsModel))
      case None => Nil
    }

    val synEmbeddings = concept.synonyms match {
      case Some(synonyms) => synonyms.map(s => generateEmbedding(s, embeddingsModel))
      case None => Nil
    }

    val allEmbeddings = (List(embedding) ++ descEmbeddings ++synEmbeddings).toArray

    val avgEmbedding = averageEmbeddings(allEmbeddings)


    GroundingConcept(
      id = concept.id,
      name = concept.name,
      description = concept.description,
      synonyms = concept.synonyms,
      embedding = Some(avgEmbedding)
    )
  }

  // TODO: Pre-process the embedidngs for the ontology offline
  def createOntologyEmbeddings(concepts:Seq[GroundingConcept], embeddingsModel:WordEmbeddingMap):Seq[GroundingConcept] = {
    concepts.map(concept => addEmbeddingToConcept(concept, embeddingsModel))
  }

  // TODO: Deserialize the ontology with the pre-processed embeddings
  def loadPreProcessedOntology(path:File):IndexedSeq[GroundingConcept] = ???
}
