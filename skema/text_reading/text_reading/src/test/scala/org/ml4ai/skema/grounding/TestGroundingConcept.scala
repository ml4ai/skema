package org.ml4ai.skema.grounding

import org.clulab.serialization.json.stringify
import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.GroundingConcept

class TestGroundingConcept extends Test {

  behavior of "GroundingConcept"

  def canonicalize(text: String): String = text
      .replaceAllLiterally(" ", "")
      .replaceAllLiterally("\r", "")
      .replaceAllLiterally("\n", "")

  it should "convert to json with Some(full) for Options" in {
    val expectedJson = """
      |{
      |  "id" : "id",
      |  "name" : "name",
      |  "description" : "description",
      |  "synonyms" : [ "synonym1", "synonym2" ],
      |  "embedding" : [ 1.0, 2.0, 3.0, 4.0 ]
      |}
    """.stripMargin
    val groundingConcept = GroundingConcept(
      "id",
      "name",
      Some("description"),
      Some(Seq("synonym1", "synonym2")),
      Some(Array(1f, 2f, 3f, 4f))
    )
    val jGroundingConcept = groundingConcept.toJValue
    val actualJson = stringify(jGroundingConcept, pretty = true)

    canonicalize(actualJson) should be (canonicalize(expectedJson))
  }

  it should "convert to json with None for Options" in {
    val expectedJson = """
      |{
      |  "id" : "id",
      |  "name" : "name"
      |}
    """.stripMargin
    val groundingConcept = GroundingConcept(
      "id",
      "name",
      None,
      None,
      None
    )
    val jGroundingConcept = groundingConcept.toJValue
    val actualJson = stringify(jGroundingConcept, pretty = true)

    canonicalize(actualJson) should be (canonicalize(expectedJson))
  }

  it should "convert to json with Some(empty) for Options" in {
    val expectedJson = """
      |{
      |  "id" : "id",
      |  "name" : "name",
      |  "description" : "description",
      |  "synonyms" : [ ],
      |  "embedding" : [ ]
      |}
    """.stripMargin
    val groundingConcept = GroundingConcept(
      "id",
      "name",
      Some("description"),
      Some(Seq.empty[String]),
      Some(Array.empty[Float])
    )
    val jGroundingConcept = groundingConcept.toJValue
    val actualJson = stringify(jGroundingConcept, pretty = true)

    canonicalize(actualJson) should be (canonicalize(expectedJson))
  }
}
