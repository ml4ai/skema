package org.ml4ai.grounding

import org.clulab.embeddings.{ExplicitWordEmbeddingMap, WordEmbeddingMap}
import org.clulab.processors.clu.CluProcessor
import org.clulab.utils.{InputStreamer, Serializer}
import org.ml4ai.grounding.MiraEmbeddingsGrounder.generateNormalizedEmbedding
import org.clulab.embeddings.CompactWordEmbeddingMap
import ujson.Arr

import java.io.File

class MiraEmbeddingsGrounder(groundingConcepts:Seq[GroundingConcept], embeddingsModel:WordEmbeddingMap) extends Grounder {

  /**
   * Returns an ordered sequence with the top k grounding candidates for the input
   *
   * @param text of the extraction to be grounded
   * @param k    number of max candidates to return
   * @return ranked list with the top k candidates
   */
  override def groundingCandidates(text: String, k: Int = 5): Seq[GroundingCandidate] = {
    // Generate the embedding of text
    val queryEmbedding = generateNormalizedEmbedding(text, embeddingsModel)
    // Normalize the query embedding to speed up the cosine similarity computation
    WordEmbeddingMap.norm(queryEmbedding)

    // Loop over the grounding concepts and get cosine similarity between input embedding and each concept
    val cosineSimilarities =
      (for (groundingConcept <- groundingConcepts.par) yield {
        WordEmbeddingMap.dotProduct(groundingConcept.embedding.get, queryEmbedding)
      }).seq
    // Choose the top k and return GroundingCandidates
    // The sorted values are reversed to have it on decreasing size
    val (sortedCosineSimilarities, sortedIndices) = cosineSimilarities.zipWithIndex.sorted.reverse.unzip


    val topKIndices = sortedIndices.take(k)
    val topSimilarities = sortedCosineSimilarities.take(k)
    val topConcepts = topKIndices.map(groundingConcepts)
    (topConcepts zip topSimilarities) map {
      case (concept, similarity) => GroundingCandidate(concept, similarity)
    }
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
              case Some(syns) => Some(syns.arr map {
                case s:ujson.Obj => s.obj.get("value").toString
                case s:ujson.Value => s.toString
              })
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
  def loadWordEmbeddingsFromTextFile(path:File): WordEmbeddingMap = ExplicitWordEmbeddingMap(path.getPath, resource = false)


  /**
   * Loads a word embeddings model from a resource
   * @param resourcePath to the serialized model
   * @return a WordEmbeddingMap instance
   */
  def loadWordEmbeddingsFromResource(resourcePath:String): WordEmbeddingMap = {
    val inputStreamer = new InputStreamer(this)
    val inputStream = inputStreamer.getResourceAsStream(resourcePath)
    val buildType = CompactWordEmbeddingMap.loadSer(inputStream)
    new CompactWordEmbeddingMap(buildType)
  }


  /**
   * Instantiate the grounder by passing paths to the data files
   * @param ontologyFile file containing the json file with MIRA concepts
   * @param wordEmbeddingsFile file containing the word embedding model
   */
  def apply(ontologyFile:File, wordEmbeddingsFile:Option[File] = None): MiraEmbeddingsGrounder = {

    val embeddingsModel = wordEmbeddingsFile match {
      case Some(file) => loadWordEmbeddingsFromTextFile(file)
      case None => loadWordEmbeddingsFromResource("/org/clulab/epimodel/model_streamed_trigram.ser")
    }

    val ontology =
      if(ontologyFile.getName.endsWith(".ser")){
        Serializer.load[Seq[GroundingConcept]](ontologyFile)
      }
      else{
        // If this was not ser, assume it is json
        val ontology = parseMiraJson(ontologyFile)
        val ontologyWithEmbeddings = createOntologyEmbeddings(ontology, embeddingsModel)
        val newFileName = ontologyFile.getAbsolutePath + ".ser"
        Serializer.save(ontologyWithEmbeddings, newFileName)
        ontologyWithEmbeddings
      }


    new MiraEmbeddingsGrounder(ontology, embeddingsModel)
  }

  def averageEmbeddings(wordEmbeddings: Array[Array[Float]]): Array[Float] = {
    val size = wordEmbeddings.length
    val embeddingsSum = wordEmbeddings.reduce((a, b) => a.zip(b).map(x => x._1 + x._2))
    embeddingsSum.map(_ / size)
  }

  def generateNormalizedEmbedding(name: String, embeddings: WordEmbeddingMap): Array[Float] = {
    // Tokenize the string
    val tokens = processor.mkDocument(name.toLowerCase).sentences.head.words

    val OOV = Array.fill(embeddings.dim)(0.0f)

    val wordEmbeddings = tokens.map(embeddings.get).map {
      case Some(e) => e.toArray
      case None => OOV
    }

    averageEmbeddings(wordEmbeddings)
  }

  def addEmbeddingToConcept(concept: GroundingConcept, embeddingsModel: WordEmbeddingMap): GroundingConcept = {


    val embedding = generateNormalizedEmbedding(concept.name.toLowerCase, embeddingsModel)

    val descEmbeddings = concept.description match {
      case Some(description) => List(generateNormalizedEmbedding(description.toLowerCase, embeddingsModel))
      case None => Nil
    }

    val synEmbeddings = concept.synonyms match {
      case Some(synonyms) => synonyms.map(s => generateNormalizedEmbedding(s.toLowerCase, embeddingsModel))
      case None => Nil
    }

    val allEmbeddings = (List(embedding) /*++ descEmbeddings */ ++ synEmbeddings ).toArray

    val avgEmbedding = averageEmbeddings(allEmbeddings)

    // Normalize the averaged embedding to speed up the cosine similarity computation
    WordEmbeddingMap.norm(avgEmbedding)

    GroundingConcept(
      id = concept.id,
      name = concept.name,
      description = concept.description,
      synonyms = concept.synonyms,
      embedding = Some(avgEmbedding)
    )
  }

  def createOntologyEmbeddings(concepts:Seq[GroundingConcept], embeddingsModel:WordEmbeddingMap):Seq[GroundingConcept] = {
    concepts.map(concept => addEmbeddingToConcept(concept, embeddingsModel))
  }
}
