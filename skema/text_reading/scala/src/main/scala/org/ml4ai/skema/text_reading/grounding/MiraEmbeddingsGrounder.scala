package org.ml4ai.skema.text_reading.grounding

import breeze.linalg.DenseVector
import MiraEmbeddingsGrounder.generateNormalizedEmbedding
import org.clulab.processors.Processor
import org.clulab.processors.clu.CluProcessor

import scala.collection.mutable
// to import all packages in linalg
import org.clulab.embeddings.{CompactWordEmbeddingMap, ExplicitWordEmbeddingMap, WordEmbeddingMap}
import org.clulab.utils.{FileUtils, InputStreamer, MED, Serializer}
import ujson.Arr

import java.io.File

class MiraEmbeddingsGrounder(processor: Processor, groundingConcepts:Seq[GroundingConcept], embeddingsModel:WordEmbeddingMap, alpha: Float) extends Grounder {

  /**
   * Returns an ordered sequence with the top k grounding candidates for the input
   *
   * @param texts of the extraction to be grounded
   * @param k    number of max candidates to return
   * @return ranked list with the top k candidates
   */
  override def groundingCandidates(texts: Seq[String], k: Int = 5): Seq[Seq[GroundingCandidate]] = texts map {
    text =>
      // Generate the embedding of text
      val queryEmbedding = generateNormalizedEmbedding(processor, text, embeddingsModel)
      // Normalize the query embedding to speed up the cosine similarity computation
      WordEmbeddingMap.norm(queryEmbedding)

      // Loop over the grounding concepts and get cosine similarity between input embedding and each concept     // Using breeze
      val tempVal = for (groundingConcept <- groundingConcepts) yield {
        WordEmbeddingMap.dotProduct(groundingConcept.embedding.get, queryEmbedding)
      }

      val cosineSimilarities: DenseVector[Float] = DenseVector( tempVal.toArray)

      val similarities =
        if (alpha < 1.0){ // If alpha is 1, then don't even bother computing edit distances
          // Using breeze
          val normalizedEditDistances = DenseVector[Float](groundingConcepts.par.map(concept => getNormalizedDistance(text, concept.name).floatValue()).seq.toArray)
          val alphas = DenseVector.fill(groundingConcepts.length)(alpha)
          val oneMinusAlphas = DenseVector.fill(groundingConcepts.length)(1 - alpha)
          val oneMinusEditDistances = DenseVector.ones[Float](normalizedEditDistances.length) -:- normalizedEditDistances
          (cosineSimilarities *:* alphas) + (oneMinusEditDistances *:* oneMinusAlphas)

        }
        else
          cosineSimilarities


      // Choose the top k and return GroundingCandidates
      // The sorted values are reversed to have it on decreasing size
      val (sortedCosineSimilarities, sortedIndices) = similarities.toArray.zipWithIndex.sorted.reverse.unzip

      val topKIndices = sortedIndices.take(k)
      val topSimilarities = sortedCosineSimilarities.take(k)
      val topConcepts = topKIndices.map(groundingConcepts)

      ((topConcepts zip topSimilarities) map {
        case (concept, similarity) => GroundingCandidate(concept, similarity)
      }).toSeq
  }

  private val editDistancesCache = mutable.HashMap[(String, String), Float]()
  private def getNormalizedDistance(text:String, name:String): Float = {

    def getDistance(source: String, target: String) = MED(source, target, allowSubstitute = false, allowTranspose = false, allowCapitalize = false).getDistance

    editDistancesCache.getOrElseUpdate((text, name), getDistance(text.toLowerCase(), name.toLowerCase())/text.length().max(name.length()))

  }
}


object MiraEmbeddingsGrounder{

  // This is held in reserve and only created if necessary.
  lazy val processor = {
    new CluProcessor()
  }

  /**
   * Parse the MIRA json file into our indexed sequence of GroundingConcepts
   * @param path to the json file with the ontology
   * @return sequence with the in-memory concepts
   */
  def parseMiraJson(json: String): IndexedSeq[GroundingConcept] = {
    ujson.read(json) match {
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
                case s:ujson.Obj => s.obj("value").str
                case s:ujson.Value => s.str
              })
              case _ => None
            },
            embedding = None
          )
      }
      case _ => ??? // Suppress warning and throw exception.
    }
  }


  def filterRelevantTerms(terms:IndexedSeq[GroundingConcept], relevantNamespaces:Set[String]): IndexedSeq[GroundingConcept] = {
    if(relevantNamespaces.nonEmpty){
      terms filter {
        term => {
          val prefix = term.id.split(":").head
          relevantNamespaces contains prefix
        }
      }
    }
    else
      terms
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
    val buildType: CompactWordEmbeddingMap.BuildType = CompactWordEmbeddingMap.loadSer(inputStream)
    //  val wordEmbeddingsMap = new CompactWordEmbeddingMap(buildType)
    val wordEmbeddingsMap = new MemoryMappedWordEmbeddingMap(buildType)

    wordEmbeddingsMap
  }


  /**
   * Instantiate the grounder by passing paths to the data files
   * @param ontologyFile file containing the json file with MIRA concepts
   * @param wordEmbeddingsFile file containing the word embedding model
   */
  def apply(ontologyPath: String, embeddingsModelResourcePath:String, lambda : Float, alpha : Float, relevantNamespaces:Set[String],
      processorOpt: Option[Processor] = None): MiraEmbeddingsGrounder = {
    val processor = processorOpt.getOrElse(this.processor)
    val embeddingsModel = loadWordEmbeddingsFromResource(embeddingsModelResourcePath)

    val ontology =
      if(ontologyPath.endsWith(".ser")){
        // TODO: Loading from a file should only be used for testing.
        Serializer.load[Seq[GroundingConcept]](new File(ontologyPath))
      }
      else {
        // If this was not ser, assume it is json
        val json = FileUtils.getTextFromResource(ontologyPath)
        val ontology = filterRelevantTerms(parseMiraJson(json), relevantNamespaces)

        val ontologyWithEmbeddings = createOntologyEmbeddings(processor, ontology, embeddingsModel, lambda)
//        val newFileName = ontologyPath.getAbsolutePath + ".ser"
//        Serializer.save(ontologyWithEmbeddings, newFileName)
        ontologyWithEmbeddings
      }


    new MiraEmbeddingsGrounder(processor, ontology, embeddingsModel, alpha)
  }

  def averageEmbeddings(wordEmbeddings: Array[Array[Float]]): Array[Float] = {
    val size = wordEmbeddings.length
    val embeddingsSum = wordEmbeddings.reduce((a, b) => a.zip(b).map(x => x._1 + x._2))
    embeddingsSum.map(_ / size)
  }

  def generateNormalizedEmbedding(processor: Processor, name: String, embeddings: WordEmbeddingMap): Array[Float] = {
    // Tokenize the string
    val tokens = processor.mkDocument(name.toLowerCase).sentences.head.words

    val OOV = Array.fill(embeddings.dim)(0.0f)

    val wordEmbeddings = tokens.map(embeddings.get).map {
      case Some(e) => e.toArray
      case None => OOV
    }

    averageEmbeddings(wordEmbeddings)
  }

  def addEmbeddingToConcept(processor: Processor, concept: GroundingConcept, embeddingsModel: WordEmbeddingMap, lambda : Float): GroundingConcept = {

    val embedding = generateNormalizedEmbedding(processor, concept.name.toLowerCase, embeddingsModel)

    val descEmbeddings = concept.description match {
      case Some(description) => List(generateNormalizedEmbedding(processor, description.toLowerCase, embeddingsModel))
      case None => Nil
    }

    val synEmbeddings = concept.synonyms match {
      case Some(synonyms) => synonyms.map(s => generateNormalizedEmbedding(processor, s.toLowerCase, embeddingsModel))
      case None => Nil
    }


    val allEmbeddings = (List(embedding.map(_ * lambda)) ++ descEmbeddings ++ synEmbeddings ).toArray

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

  def createOntologyEmbeddings(processor: Processor, concepts:Seq[GroundingConcept], embeddingsModel:WordEmbeddingMap, lambda : Float):Seq[GroundingConcept] = {
    concepts.map(concept => addEmbeddingToConcept(processor, concept, embeddingsModel, lambda))
  }
}
