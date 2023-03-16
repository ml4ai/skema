package org.ml4ai.skema.text_reading

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.clulab.processors.Document
import org.clulab.utils.Logging
import org.ml4ai.skema.text_reading.attachments.GroundingAttachment
import org.ml4ai.skema.text_reading.grounding.{GroundingCandidate, MiraWebApiGrounder}

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

  def groundMentions(mentions: Seq[Mention]):Seq[Mention] = {
    // Helper function to fetch all the arguments of mentions that have to be grounded
    def fetchGroundableArguments(ms: Seq[Mention]):Seq[TextBoundMention] = ms flatMap {
      case tbm: TextBoundMention =>
        val isGrounded =
          tbm.attachments.exists(!_.isInstanceOf[GroundingAttachment])

        if (!isGrounded) {
          // Don't ground unless it is not a number
          numberPattern.findFirstIn(tbm.text.trim) match {
            case None => List(tbm)
            case Some(_) => Nil // If it is a number, then don't ground
          }

        }
        else
          Nil
      case e @ (_:EventMention | _:RelationMention) => e.arguments.values.flatMap(fetchGroundableArguments)
      case _ => Nil

    }

    // Get the mentions to ground from the events and relations
    val candidatesToGround = fetchGroundableArguments(mentions).distinct

    // Do batch grounding
    val mentionsGroundings = grounder.groundingCandidates(candidatesToGround.map(_.text), k=5).map{
      gs => gs.filter{
        // Filter by grounding threshold
        case GroundingCandidate(_, score) => score >= groundingAssignmentThreshold
      }
    }

    val groundedTextBoundMentions =
      (candidatesToGround zip mentionsGroundings).map{
        case (original, groundings) =>
          if (groundings.nonEmpty)
            original ->original.withAttachment(new GroundingAttachment(groundings))
          else
            original -> original
      }.toMap

    // Helper function to recursively replace the mentions with their grounded arguments, regardless of the nesting level
    def rebuildArguments(m:Mention,
                         groundedMentions:Map[TextBoundMention, Mention]):Mention =  m match {
      case tb:TextBoundMention => groundedTextBoundMentions.getOrElse(tb, tb)
      case e:EventMention =>
        e.copy(arguments = e.arguments mapValues {
          _.map{
            e => rebuildArguments(e, groundedMentions)
          }
        })
      case r: RelationMention =>
        r.copy(arguments = r.arguments mapValues {
          _.map {
            e => rebuildArguments(e, groundedMentions)
          }
        })
    }

    // Now, rebuild the events, replacing the arguments with the grounded mentions
    mentions map (m => rebuildArguments(m, groundedTextBoundMentions))
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

    // Choose which mentions to ground
    val (textBound, nonTextBound) = mentions.partition(m => if(m.isInstanceOf[TextBoundMention]) true else false)

    // Ground them
    val groundedMentions = groundMentions(nonTextBound)

    (doc, groundedMentions ++ textBound)
  }
}
