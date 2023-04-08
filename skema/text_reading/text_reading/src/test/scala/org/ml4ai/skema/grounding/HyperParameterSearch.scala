package org.ml4ai.skema.grounding

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Gaussian, Uniform}
import com.typesafe.config.ConfigFactory
import org.clulab.utils.FileUtils
import org.ml4ai.skema.text_reading.grounding.MiraEmbeddingsGrounder

import scala.collection.immutable.ListMap
import scala.collection.JavaConverters._

/** *
  * HyperParameterSearch : generic, bruteforce Hyper parameter search using breeze (numpy style) calculations.
  * Follows:
  * For each concept.name
  * Find edit distance between text to be grounded and the concept.name
  * normalized edit distances = Edit distance/length of longest string among (text to be grounded, concept.name)
  * alpha * cosineSimilarities + (1-alpha) (1-normalized edit distances)
  */
object HyperParameterSearch extends App {

  /** *
    * Get accuracy for this lambda, alpha values provided.
    * MiraEmbeddingsGrounder is initialized for lambda, alpha value.
    * @param lambda contribution of the cosine similarity to the grounding score
    * @param alpha contribution of edit distance to grounding score
    * @return accuracy
    */
  def getAccuracyForThisHyperParams(lambda: Float, alpha: Float, path: String): Float = {
    val miraEmbeddingsGrounderGS: MiraEmbeddingsGrounder = {
      val config = ConfigFactory.load().getConfig("Grounding")
      val domainConfig = config.getConfig(config.getString("domain"))
      val ontologyPath = domainConfig.getString("ontologyPath")
      val relevantNamespaces = domainConfig.getStringList("relevantNamespaces").asScala.toSet
      // val embeddingsPath = domainConfig.getString("embeddingsPath")
      MiraEmbeddingsGrounder(ontologyPath, None, lambda, alpha, relevantNamespaces)
    }

    val targets = {
      // Drop the first line that is the header
      FileUtils.getTextFromResource(path = resourcePath).split("\n").drop(1) map { line =>
        val tokens = line.split("\t")
        (tokens(0), tokens(1))
      }
    }


    val predictions = {
      for {(text, groundingId) <- targets
           } yield miraEmbeddingsGrounderGS.ground(text) match {
        case Some(concept) => concept.id == groundingId
        case None => false
      }
    }

    val accuracy = (predictions.count(identity).floatValue() / predictions.length).floatValue()
    accuracy
  }

  // Grid search -bruteforce
  var hyperParamSize = 2

  // Keith's review comment use args.lift for commandline arguments for standalone script
  val resourcePath = args.headOption.getOrElse("/grounding_tests.tsv")
  val isSimulation = args.lift(1).contains("true")
  var lambdas = DenseVector[Float](1f, 10f) //, 100f) //Uniform(10, 1000).samplesVector(hyperParamSize)
  var alphas = DenseVector[Float](0.25f, 0.5f) //, 0.75f) //Gaussian(0.0.toFloat, 1.0.toFloat).samplesVector(hyperParamSize).toDenseVector
  if (isSimulation) {
    lambdas = Uniform(10f, 1000f).samplesVector(hyperParamSize).values.map(_.toFloat).toDenseVector
    alphas = Gaussian(0.0f, 1.0f).samplesVector(hyperParamSize).values.map(_.toFloat).toDenseVector
  }

  val accuracyMap = new scala.collection.mutable.HashMap[(Float, Float), Float]()

  // for each lambda, alpha value chosen just for a base case, loop mover to calculate accuracy
  // Keith's review comment use flatmap instead of for loop over ScalaVectors for lambda & alpha values
  val keyAccuracyPairs = lambdas.toScalaVector.flatMap { lambda =>
    alphas.toScalaVector().map { alpha =>
      (lambda, alpha) -> getAccuracyForThisHyperParams(lambda, alpha, resourcePath)
    }
  }.sortBy(-_._2)

  val sortedAccuracyMap = ListMap(keyAccuracyPairs: _*)

  println("Hyperparameters' scores")
  println(sortedAccuracyMap.map{
    case ((lambda, alpha), accuracy) => s"Lambda: $lambda, Alpha: $alpha\t-\tAccuracy: $accuracy"
  }.mkString("\n"))
}


