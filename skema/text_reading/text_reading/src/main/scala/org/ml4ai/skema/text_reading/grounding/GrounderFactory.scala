package org.ml4ai.skema.text_reading.grounding

import com.typesafe.config.Config
import org.clulab.processors.fastnlp.FastNLPProcessor

object GrounderFactory {
  def getInstance(config:Config):Grounder = {
    // Grounding parameters
    config.getString("engine").toLowerCase match {
      case "miraembeddings" =>
        val ontologyFilePath = config.getString("ontologyPath")
        val lambda = config.getInt("lambda")
        val alpha = config.getDouble("alpha").toFloat
        MiraEmbeddingsGrounder(ontologyFilePath, None, lambda, alpha)
      case "mirawebapi" =>
        val endpoint = config.getString("apiEndpoint")
        new MiraWebApiGrounder(endpoint)
      case "manual" =>
        val manualGroundingsResourcePath = config.getString("manualGroundings")
        val processor = new FastNLPProcessor(withChunks = false, internStrings = false)
        ManualGrounder.fromFileOrResource(manualGroundingsResourcePath, processor)
      case other =>
        throw new RuntimeException(s"$other - is not implemented as a grounding engine")
    }
  }
}
