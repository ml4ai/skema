package org.ml4ai.skema.grounding

import org.clulab.serialization.json.stringify
import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.{GroundingCandidate, GroundingConcept}

class TestGroundingCandidate extends Test {

  behavior of "GroundingCandidate"

  def canonicalize(text: String): String = text
      .replaceAllLiterally(" ", "")
      .replaceAllLiterally("\r", "")
      .replaceAllLiterally("\n", "")

  it should "convert to json" in {
    val expectedJson = """
      |{
      |  "groundingConcept": {
      |    "id" : "id",
      |    "name" : "name",
      |    "description" : "description",
      |    "synonyms" : [ "synonym1", "synonym2" ],
      |    "embedding" : [ 1.0, 2.0, 3.0, 4.0 ]
      |  },
      |  "score": 1.5
      |}
    """.stripMargin
    val groundingConcept = GroundingConcept(
      "id",
      "name",
      Some("description"),
      Some(Seq("synonym1", "synonym2")),
      Some(Array(1f, 2f, 3f, 4f))
    )
    val groundingCandidate = GroundingCandidate(groundingConcept, 1.5f)
    val jGroundingCandidate = groundingCandidate.toJValue
    val actualJson = stringify(jGroundingCandidate, pretty = true)

    canonicalize(actualJson) should be (canonicalize(expectedJson))
  }
}
