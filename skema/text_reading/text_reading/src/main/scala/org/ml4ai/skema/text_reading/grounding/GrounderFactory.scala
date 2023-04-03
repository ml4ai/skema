package org.ml4ai.skema.text_reading.grounding

import com.typesafe.config.Config
import org.clulab.processors.fastnlp.FastNLPProcessor

object GrounderFactory {
  def getInstance(config: Config, chosenEngine: Option[String] = None): Grounder = {
    val engine = chosenEngine getOrElse config.getString("engine")
    val prependManualGrounder = config.getBoolean("forceManualGroundings")
    lazy val manualGrounder = buildManualGrounder(config)
    // Grounding parameters
    engine.toLowerCase match {
        case "miraembeddings" =>
          val ontologyFilePath = config.getString("ontologyPath")
          val lambda = config.getInt("lambda")
          val alpha = config.getDouble("alpha").toFloat
          val grounder = MiraEmbeddingsGrounder(ontologyFilePath, None, lambda, alpha)
          if(prependManualGrounder)
            new PipelineGrounder(Seq(manualGrounder, grounder))
          else
            grounder
        case "mirawebapi" =>
          val endpoint = config.getString("apiEndpoint")
          val grounder = new MiraWebApiGrounder(endpoint)
          if (prependManualGrounder)
            new PipelineGrounder(Seq(manualGrounder, grounder))
          else
            grounder
      case "manual" => manualGrounder
        case other =>
          throw new RuntimeException(s"$other - is not implemented as a grounding engine")
      }
  }

  def buildManualGrounder(config:Config): ManualGrounder = {
    val manualGroundingsResourcePath = config.getString("manualGroundings")
    val processor = new FastNLPProcessor(withChunks = false, internStrings = false)
    ManualGrounder.fromFileOrResource(manualGroundingsResourcePath, processor)
  }
}
