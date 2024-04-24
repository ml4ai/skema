package org.ml4ai.skema.grounding.scala_grounders

import org.ml4ai.skema.test.Test

import java.nio.{Buffer, ByteBuffer, ByteOrder}

import org.ml4ai.skema.text_reading.grounding.Grounder
import org.ml4ai.skema.text_reading.grounding.GroundingCandidate
import org.ml4ai.skema.text_reading.grounding.GroundingConcept

/**
  * 
  * Running command:
  *   sbt "test:runMain org.ml4ai.skema.grounding.scala_grounders.TestAdapter"
  */
class TestAdapter extends Test {

  behavior of "ScalaGroundersAdapter"

  val gcs = Seq(
    GroundingConcept(
      id          = "id1",
      name        = "dog",
      description = Some("this is a cute dog"),
      synonyms    = None,
      embedding   = None
    ), 
    GroundingConcept(
      id          = "id2",
      name        = "cat",
      description = Some("this is a cute cat"),
      synonyms    = None,
      embedding   = None
    ), 
    GroundingConcept(
      id          = "id3",
      name        = "dog cat",
      description = Some("here we have a dog and a cat"),
      synonyms    = None,
      embedding   = None
    ), 
    GroundingConcept(
      id          = "id4",
      name        = "cat",
      description = Some("this is a cute cat"),
      synonyms    = None,
      embedding   = None
    ), 
  )

  val sga = new ScalaGroundersAdapter(gcs)

  val result = sga.groundingCandidates(Seq("dog"), 10).head

  // Check that the first one is a GroundingCandidate with id1
  it should "ground" in {
    result.foreach(println) 
    result.head.concept.id should be ("id1")
  }


}
