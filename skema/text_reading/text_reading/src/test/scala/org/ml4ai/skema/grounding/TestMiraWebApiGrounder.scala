package org.ml4ai.skema.grounding

import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.MiraWebApiGrounder
import org.scalatest.OptionValues.convertOptionToValuable

import scala.language.reflectiveCalls

class TestMiraWebApiGrounder extends Test {

  // TODO: Remove the API endpoint form here and add it to a testing configuration file
  val miraWebApiGrounder: MiraWebApiGrounder = {
    new MiraWebApiGrounder(apiEndPoint = "http://34.230.33.149:8771/api/ground_list")
  }

  def correctGrounding(shouldable: Shouldable, text: String, groundingID: String): Unit = {
    shouldable should s"ground $text to the grounding concept with id $groundingID" in {
      val groundedConcept = miraWebApiGrounder.ground(text)

      groundedConcept.value.id should be(groundingID)
    }
  }

  behavior of "Exact matches"

  // These are sanity checks. The input text is exactly the description of the entity, so it should match perfectly with the corresponding concept
  correctGrounding(failingTest, "COVID-19", "doid:0080600")
  correctGrounding(failingTest, "cell part", "caro:0000014")
  // This is a slightly harder entity to ground
  correctGrounding(failingTest, "junctional epidermolysis bullosa non-Herlitz type", "doid:0060738")

}
