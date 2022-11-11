package org.ml4ai.skema.test

import com.typesafe.config.Config
import org.clulab.odin.Mention
import org.clulab.processors.Document
import org.clulab.serialization.json.JSONSerializer
import org.json4s.jackson.JsonMethods._
import org.ml4ai.skema.text_reading.OdinEngine
import org.ml4ai.skema.text_reading.apps.ExtractAndAlign

object TestUtils {

  // From Processors -- I couldn't import it for some reason
  def jsonStringToDocument(jsonstr: String): Document = JSONSerializer.toDocument(parse(jsonstr))

  val successful = Seq()

  protected var mostRecentOdinEngine: Option[OdinEngine] = None
  protected var mostRecentConfig: Option[Config] = None
  lazy val extractorAligner = ExtractAndAlign

  // This is the standard way to extract mentions for testing
  def extractMentions(ieSystem: OdinEngine, text: String): Seq[Mention] = {
    ieSystem.extractFromText(text, true, None)
  }

  def newOdinSystem(config: Config): OdinEngine = this.synchronized {
    val readingSystem =
      if (mostRecentOdinEngine.isEmpty) OdinEngine.fromConfig(config)
      else if (mostRecentConfig.get == config) mostRecentOdinEngine.get
      else OdinEngine.fromConfig(config)

    mostRecentOdinEngine = Some(readingSystem)
    mostRecentConfig = Some(config)
    readingSystem
  }
}
