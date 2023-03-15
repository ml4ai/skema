package org.ml4ai.skema.text_reading

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.clulab.processors.Document
import org.clulab.utils.Logging
import org.ml4ai.grounding.{GroundingCandidate, MiraEmbeddingsGrounder, MiraWebApiGrounder}
import org.ml4ai.skema.text_reading.attachments.GroundingAttachment

import scala.util.matching.Regex

class TextReadingPipeline extends Logging {
  logger.info("Initializing the OdinEngine ...")

  // Read the configuration from the files
  val generalConfig: Config = ConfigFactory.load()
  val readerType: String = generalConfig[String]("ReaderType")
  val defaultConfig: Config = generalConfig[Config](readerType)
  val config: Config = defaultConfig.withValue("preprocessorType", ConfigValueFactory.fromAnyRef("PassThrough"))
  val groundingConfig: Config = generalConfig.getConfig("Grounding")

  // Grounding parameters
  private val ontologyFilePath = groundingConfig.getString("ontologyPath")
  private val groundingAssignmentThreshold = groundingConfig.getDouble("assignmentThreshold")
//  private val grounder = MiraEmbeddingsGrounder(ontologyFilePath, None, lambda = 10, alpha = 1.0f) // TODO: Fix this @Enrique
  private val grounder = new MiraWebApiGrounder("http://34.230.33.149:8771/api/ground_list")

  val numberPattern: Regex = """^\.?(\d)+([.,]?\d*)*$""".r

  // Odin Engine instantiation
  private val odinEngine = OdinEngine.fromConfig(config)
  // Initialize the odin engine
  odinEngine.annotate("x = 10")

  /**
    * Assigns grounding elements to a mention
    *
    * @param m mention to ground
    * @return instance of the mention with grounding attachment if applicable
    */
  def groundMention(m: Mention): Mention = m match {
    case tbm: TextBoundMention =>
      val isGrounded =
        tbm.attachments.exists(!_.isInstanceOf[GroundingAttachment])

      if (!isGrounded) {
        // Don't ground unless it is not a number
        numberPattern.findFirstIn(tbm.text.trim) match {
          case None =>
            val topGroundingCandidates = grounder.groundingCandidates(tbm.text, k=5).filter {
              case GroundingCandidate(_, score) => score >= groundingAssignmentThreshold
            }

            if (topGroundingCandidates.nonEmpty)
              tbm.withAttachment(new GroundingAttachment(topGroundingCandidates))
            else
              tbm
          case Some(_) => tbm // If it is a number, then don't ground
        }

      }
      else
        tbm
    case e: EventMention =>
      val groundedArguments =
        e.arguments.mapValues(_.map(groundMention))


      e.copy(arguments = groundedArguments)
    // This is duplicated while we fix the Mention trait to define the abstract method copy
    case e: RelationMention =>
      val groundedArguments =
        e.arguments.mapValues(_.map(groundMention))

      e.copy(arguments = groundedArguments)
    case m => m
  }

  /**
    * Runs the mention extraction engine on the  parameter
    *
    * @param text     to annotate
    * @param fileName to identify the provenance of the document being annotated
    * @return Annotated doc object and the mentions extracted by odin
    */
  def extractMentions(text: String, fileName: Option[String]): (Document, Seq[Mention]) = {
    // Extract mentions and apply grounding
    val ExtractionResults(doc, mentions) = odinEngine.extractFromText(text, keepText = true, fileName)
    // Run grounding
    val groundedMentions = mentions.par.map {
      // Only ground arguments of events and relations, to save time
      case e@(_: EventMention | _: RelationMention) => groundMention(e)
      case m => m
    }.seq

    (doc, groundedMentions)
  }
}
