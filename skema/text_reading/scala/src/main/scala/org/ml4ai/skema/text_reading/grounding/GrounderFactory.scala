package org.ml4ai.skema.text_reading.grounding

import com.typesafe.config.Config
import org.clulab.processors.Processor
import org.clulab.processors.fastnlp.FastNLPProcessor

import scala.collection.JavaConverters._

object GrounderFactory {

  def getInstance(config: Config, processorOpt: Option[Processor] = None, chosenEngine: Option[String] = None): Grounder = {
    val engine = chosenEngine getOrElse config.getString("engine")
    val prependManualGrounder = config.getBoolean("forceManualGroundings")
    val domain = config.getString("domain")
    val domainConfig = config.getConfig(domain)
    lazy val manualGrounder = buildManualGrounder(domainConfig)
    // Grounding parameters
    engine.toLowerCase match {
        case "miraembeddings" =>
          val ontologyFilePath = domainConfig.getString("ontologyPath")
          val embeddingsModelPath = domainConfig.getString("embeddingsModelPath")
          val lambda = domainConfig.getInt("lambda")
          val alpha = domainConfig.getDouble("alpha").toFloat
          val namespaces = domainConfig.getStringList("relevantNamespaces").asScala.toSet
          val grounder = MiraEmbeddingsGrounder(ontologyFilePath, embeddingsModelPath, lambda, alpha, namespaces, processorOpt)
          if (prependManualGrounder)
            new PipelineGrounder(Seq(manualGrounder, grounder))
          else
            grounder
        case "mirawebapi" =>
          val endpoint = domainConfig.getString("apiEndpoint")
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
