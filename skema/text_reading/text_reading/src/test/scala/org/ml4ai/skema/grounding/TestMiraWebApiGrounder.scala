package org.ml4ai.skema.grounding

import ai.lum.common.ConfigFactory
import com.typesafe.config.Config
import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.{Grounder, GrounderFactory, MiraWebApiGrounder, PipelineGrounder}
import org.scalatest.OptionValues.convertOptionToValuable

import scala.language.reflectiveCalls

class TestMiraWebApiGrounder extends Test {

  val config:Config = ConfigFactory.load().getConfig("Grounding")
  implicit val miraWebApiGrounder: MiraWebApiGrounder = {
    new MiraWebApiGrounder(apiEndPoint = config.getString("apiEndpoint"))
  }

  val pipelineGrounder: PipelineGrounder = {
    new PipelineGrounder(Seq(GrounderFactory.buildManualGrounder(config), miraWebApiGrounder))
  }

  def correctGrounding(shouldable: Shouldable, text: String, groundingID: String)(implicit grounder:Grounder): Unit = {
    shouldable should s"ground $text to the grounding concept with id $groundingID" in {
      val groundedConcept = grounder.ground(text)

      groundedConcept.value.id should be(groundingID)
    }
  }

  behavior of "Exact matches"

  // These are sanity checks. The input text is exactly the description of the entity, so it should match perfectly with the corresponding concept
  correctGrounding(passingTest, "COVID-19", "doid:0080600")
  correctGrounding(failingTest, "cell part", "caro:0000014")
  // This is a slightly harder entity to ground
  correctGrounding(failingTest, "junctional epidermolysis bullosa non-Herlitz type", "doid:0060738")


  behavior of "Pipelined matches"

  correctGrounding(passingTest, "solar flares", "tr:001")(pipelineGrounder)
  correctGrounding(passingTest, "COVID-19", "doid:0080600")(pipelineGrounder)

  it should "correctly match these entries in batch mode" in {
    val groundings = pipelineGrounder.groundingCandidates(Seq("solar flares", "COVID-19", "reproduction ratio", "non groundable"), k = 1)

    groundings.head should not be empty
    groundings.head.head.concept.id should be ("tr:001")

    groundings(1) should not be empty
    groundings(1).head.concept.id should be("doid:0080600")

    groundings(2) should not be empty
    groundings(2).head.concept.id should be("tr:002")

    groundings.last shouldBe empty
  }


}
