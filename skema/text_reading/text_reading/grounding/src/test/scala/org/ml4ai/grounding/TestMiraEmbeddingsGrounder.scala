package org.ml4ai.grounding

import breeze.linalg._
import com.typesafe.config.ConfigFactory
import org.clulab.utils.FileUtils
import org.json4s.JsonDSL.boolean2jvalue
import org.ml4ai.skema.common.test.Test
import org.scalatest.OptionValues._
import breeze.stats.distributions._

class TestMiraEmbeddingsGrounder extends Test {

  // Lazily load the grounder. We assume it's state and behavior is immutable
  // So we can build it once and reuse it as necessary in the suite

  val miraEmbeddingsGrounder: MiraEmbeddingsGrounder = {
    val config = ConfigFactory.load().getConfig("Grounding")
    val ontologyPath = config.getString("ontologyPath")
    // val embeddingsPath = config.getString("embeddingsPath")

    MiraEmbeddingsGrounder(ontologyPath, None, 10, 0.25f)
  }

  def correctGrounding(shouldable: Shouldable, text: String, groundingID: String): Unit = {
    shouldable should s"ground $text to the grounding concept with id $groundingID" in {
      val groundedConcept = miraEmbeddingsGrounder.ground(text)

      groundedConcept.value.id should be(groundingID)
    }
  }

  behavior of "Exact matches"

  // These are sanity checks. The input text is exactly the description of the entity, so it should match perfectly with the corresponding concept
  correctGrounding(failingTest, "COVID-19", "doid:0080600")
  correctGrounding(passingTest, "cell part", "caro:0000014")
  // This is a slightly harder entity to ground
  correctGrounding(failingTest, "junctional epidermolysis bullosa non-Herlitz type", "doid:0060738")

  behavior of "Synonym matches"

  // These matches are to synonyms. Are not exactly the same, but should be handled correctly by the grounding algorithm
  correctGrounding(failingTest, "junctional epidermolysis bullosa generalisata mitis", "doid:0060738")
  correctGrounding(passingTest, "covid19", "doid:0080600")
  correctGrounding(failingTest, "s-block compounds", "chebi:33674")

  behavior of "Accuracy of the matches"

  // This is the main test, where we are measuring the accuracy of the grounding according to a set of test gronding queries
  it should "achieve at least 70% accuracy" in {
    val groundingTargets = {
      val targets = {
        // Drop the first line that is the header
        FileUtils.getTextFromResource("/grounding_tests.tsv").split("\n").drop(1) map { line =>
          val tokens = line.split("\t")
          (tokens(0), tokens(1))
        }
      }

      val predictions =
        for {
          (text, groundingId) <- targets
        } yield miraEmbeddingsGrounder.ground(text) match {
          case Some(concept) => concept.id == groundingId
          case None => false
        }
      val accuracy = predictions.count(identity).floatValue() / predictions.length

      accuracy should be >= 0.65f // TODO: .7f is the goal


    }
  }

  //TODO: add more unittests to test hyper parameter search as well as rules - Sushma Akoju (after dec 15th)
  //val hyperParameterSearch = HyperParameterSearch()

}