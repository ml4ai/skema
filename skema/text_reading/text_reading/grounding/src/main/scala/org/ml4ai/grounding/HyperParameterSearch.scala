package org.ml4ai.grounding

import breeze.linalg._
import com.typesafe.config.ConfigFactory
import org.clulab.utils.FileUtils
import org.json4s.JsonDSL.boolean2jvalue
import breeze.stats.distributions._
import ujson.True

import scala.collection.immutable.ListMap

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
    *
    * @param lambda
    * @param alpha
    * @return accuracy
    */
  def getAccuracyForThisHyperParams(lambda: Float, alpha: Float, path: String): Float = {
    val miraEmbeddingsGrounderGS: MiraEmbeddingsGrounder = {
      val config = ConfigFactory.load().getConfig("Grounding")
      val ontologyPath = config.getString("ontologyPath")
      // val embeddingsPath = config.getString("embeddingsPath")
      MiraEmbeddingsGrounder(ontologyPath, None, lambda, alpha)
    }

    val targets = {
      // Drop the first line that is the header
      FileUtils.getTextFromResource(path = path).split("\n").drop(1) map { line =>
        val tokens = line.split("\t")
        (tokens(0), tokens(1))
      }
    }
    print(targets)

    val predictions = {
      for {(text, groundingId) <- targets
           } yield miraEmbeddingsGrounderGS.ground(text) match {
        case Some(concept) => concept.id == groundingId
        case None => false
      }
    }
    print(predictions)
    val accuracy = (predictions.count(identity).floatValue() / predictions.length).floatValue()
    val acc = accuracy.toFloat
    acc
  }

  // Grid search -bruteforce
  var hyperparam_size: Int = 2;

  // Keith's review comment use args.lift for commandline arguments for standalone script
  val resourcePath = args.lift(0).getOrElse("defaultvalue")
  val isSimulation = args.lift(1).map(_ == "true").getOrElse(false)
  var lambdas = DenseVector[Float](1f, 10f) //, 100f) //Uniform(10, 1000).samplesVector(hyperparam_size)
  var alphas = DenseVector[Float](0.25f, 0.5f) //, 0.75f) //Gaussian(0.0.toFloat, 1.0.toFloat).samplesVector(hyperparam_size).toDenseVector
  if (isSimulation) {
    lambdas = Uniform(10f, 1000f).samplesVector(hyperparam_size).values.map(_.toFloat).toDenseVector
    alphas = Gaussian(0.0f, 1.0f).samplesVector(hyperparam_size).values.map(_.toFloat).toDenseVector
  }

  val acc_map = new scala.collection.mutable.HashMap[(Float, Float), Float]()

  // for each lambda, alpha value chosen just for a base case, loop mover to calculate accuracy
  // Keith's review comment use flatmap instead of for loop over ScalaVectors for lambda & alpha values
  val key_accuracy_pairs = lambdas.toScalaVector.flatMap{ lambda =>
    alphas.toScalaVector().map { alpha =>
      (lambda, alpha)  -> getAccuracyForThisHyperParams(lambda, alpha, resourcePath)
    }
  }.sortBy(-_._2)

  val sorted_accuracy_map = ListMap(key_accuracy_pairs:_*)
  println(acc_map)
  print(sorted_accuracy_map)
  print(lambdas)
  (sorted_accuracy_map)
}




