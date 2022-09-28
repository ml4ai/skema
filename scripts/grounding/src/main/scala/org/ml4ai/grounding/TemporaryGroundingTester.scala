package org.ml4ai.grounding

import com.typesafe.config.ConfigFactory

import java.io.File

class TemporaryGroundingTester()  {
  val config = ConfigFactory.load().getConfig("Grounding")

  // Add the bulk of your code here.
  val grounder = MiraEmbeddingsGrounder(new File(config.getString("ontologyPath")),
    new File(config.getString("embeddingsPath")))

  // Do some tinkering here
  val grounding = grounder.ground("covid 19")
  println(grounding)
}

object TemporaryGroundingTester extends App{

  def apply(): TemporaryGroundingTester = new TemporaryGroundingTester()

  TemporaryGroundingTester()
}
