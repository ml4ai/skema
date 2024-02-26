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
    lazy val manualGrounder = buildManualGrounder(domainConfig, processorOpt)
    // Grounding parameters
    engine.toLowerCase match {
        case "miraembeddings" =>
          val ontologyFilePath = domainConfig.getString("ontologyPath")
          val embeddingsModelPath = {
            if(domainConfig.hasPath("embeddingsModelPath"))
              domainConfig.getString("embeddingsModelPath")
            else
              "/org/clulab/glove/glove.840B.300d.10f.kryo"
          }


          val lambda = if(domainConfig.hasPath("lambda"))
            domainConfig.getInt("lambda")
          else
            10.0f
          val alpha = if(domainConfig.hasPath("alpha"))
              domainConfig.getDouble("alpha").toFloat
          else
            1.0f
          val namespaces = {
            if(domainConfig.hasPath("relevantNamespaces"))
              domainConfig.getStringList("relevantNamespaces").asScala.toSet
            else
              Set[String]()
          }
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

  def buildManualGrounder(config:Config, processorOpt: Option[Processor] = None): ManualGrounder = {
    processorOpt.foreach { processor =>
      // This check is to ensure former behavior.
      assert(processor.isInstanceOf[FastNLPProcessor])
    }
    // Although not chunking can save time, it will also waste space, so use an existing processor if possible.
    val processor = processorOpt.getOrElse(new FastNLPProcessor(withChunks = false, internStrings = false))

    if(config.hasPath("manualGroundings")) {
      val manualGroundingsResourcePath = config.getString("manualGroundings")

      ManualGrounder.fromFileOrResource(manualGroundingsResourcePath, processor)
    }
    else
      new ManualGrounder(targetEntries = Nil, processor)
  }
}
