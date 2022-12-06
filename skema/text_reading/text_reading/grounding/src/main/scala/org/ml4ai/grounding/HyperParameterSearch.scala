package org.ml4ai.grounding

import breeze.linalg._
import com.typesafe.config.ConfigFactory
import org.clulab.utils.FileUtils
import org.json4s.JsonDSL.boolean2jvalue
import breeze.stats.distributions._

/***
  * HyperParameterSearch : generic, bruteforce Hyper parameter search using breeze (numpy style) calculations.
  * Follows:
  * For each concept.name
  * Find edit distance between text to be grounded and the concept.name
  * normalized edit distances = Edit distance/length of longest string among (text to be grounded, concept.name)
  * alpha * cosineSimilarities + (1-alpha) (1-normalized edit distances)
  */
object HyperParameterSearch extends App {
  // Grid search -bruteforce
  var hyperparam_size: Int = 2;
  val lambdas = DenseVector(1, 10, 100) //Uniform(10, 1000).samplesVector(hyperparam_size)
  val alphas = DenseVector(0.25, 0.5, 0.75) //Gaussian(0.0.toFloat, 1.0.toFloat).samplesVector(hyperparam_size).toDenseVector
  //  val lambdas = DenseVector(1, 3, 5, 7, 10, 50, 75, 100) //Uniform(10, 1000).samplesVector(hyperparam_size)
  //  val alphas = DenseVector(0.25, 0.5, 0.75) //Gaussian(0.0.toFloat, 1.0.toFloat).samplesVector(hyperparam_size).toDenseVector

  val acc_map = new scala.collection.mutable.HashMap[(Float, Float), Float]()

  // for each lambda, alpha value chosen just for a base case, loop mover to calculate accuracy
  for (lambda <- lambdas.toScalaVector()) yield {
    for (alpha <- alphas.toScalaVector()) yield {
      val this_acc = getAccuracyForThisHyperParams(lambda.toFloat, alpha.toFloat)
      acc_map.update((lambda.toFloat, alpha.toFloat), this_acc)
    }
  }

  println(acc_map)

  /***
    * Get accuracy for this lambda, alpha values provided.
    * MiraEmbeddingsGrounder is initialized for lambda, alpha value.
    * @param lambda
    * @param alpha
    * @return accuracy
    */
  def getAccuracyForThisHyperParams(lambda: Float, alpha: Float): Float = {
    val miraEmbeddingsGrounderGS: MiraEmbeddingsGrounder = {
      val config = ConfigFactory.load().getConfig("Grounding")
      val ontologyPath = config.getString("ontologyPath")
      // val embeddingsPath = config.getString("embeddingsPath")
      MiraEmbeddingsGrounder(ontologyPath, None, lambda, alpha)
    }


    val targets = {
      // Drop the first line that is the header
      FileUtils.getTextFromResource("/grounding_tests.tsv").split("\n").drop(1) map { line =>
        val tokens = line.split("\t")
        (tokens(0), tokens(1))
      }
    }

    val predictions =
      for {(text, groundingId) <- targets
           } yield miraEmbeddingsGrounderGS.ground(text) match {
        case Some(concept) => concept.id == groundingId
        case None => false
      }
    val accuracy = (predictions.count(identity).floatValue() / predictions.length).floatValue()
    val acc = accuracy.toFloat
    acc
  }

}

