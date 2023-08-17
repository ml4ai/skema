package org.ml4ai.skema.text_reading

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.clulab.processors.{Document, Processor}
import org.clulab.utils.Logging
import org.ml4ai.skema.text_reading.attachments.GroundingAttachment
import org.ml4ai.skema.text_reading.grounding.{Grounder, GrounderFactory, GroundingCandidate}
import org.ml4ai.skema.text_reading.mentions.CrossSentenceEventMention

import scala.language.implicitConversions
import scala.util.matching.Regex

class TextReadingPipeline(processorOpt: Option[Processor] = None, odinEngineOpt: Option[OdinEngine] = None, grounderOpt: Option[Grounder] = None) extends Logging {
  logger.info("Initializing the OdinEngine ...")

  val odinEngine = odinEngineOpt.getOrElse(TextReadingPipeline.newOdinEngine())
  val grounder = grounderOpt.getOrElse(TextReadingPipeline.newGrounder(processorOpt))
  val groundingAssignmentThreshold = {
    val config: Config = ConfigFactory.load()
    val groundingConfig: Config = config.getConfig("Grounding")

    groundingConfig.getDouble("assignmentThreshold")
  }
  private val numberPattern: Regex = """^\.?(\d)+([.,]?\d*)*$""".r

  def isGroundable(m:TextBoundMention): Boolean = {
    val isGrounded =
      m.attachments.exists(!_.isInstanceOf[GroundingAttachment])

    if (!isGrounded) {
      // Don't ground unless it is not a number
      numberPattern.findFirstIn(m.text.trim) match {
        case None => true
        case Some(_) => false// If it is a number, then don't ground
      }
    }
    else
      false
  }

  def fetchNestedArguments(ms: Seq[Mention]): Seq[TextBoundMention] = ms flatMap {
    case tbm: TextBoundMention => List(tbm)
    case e:Mention => e.arguments.values.flatMap(fetchNestedArguments)
  }

  def groundMentions(mentions: Seq[Mention]):Seq[Mention] = {
    // Helper function to fetch all the arguments of mentions that have to be grounded


    // Get the mentions to ground from the events and relations
    val candidatesToGround = fetchNestedArguments(mentions).filter(isGroundable).distinct

    // Do batch grounding
    val mentionsGroundings = grounder.groundingCandidates(candidatesToGround.map(_.text), k=5).map{
      gs => gs.filter(_.score >= groundingAssignmentThreshold)
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
      case cs: CrossSentenceEventMention =>
        cs.copy(arguments = cs.arguments mapValues {
          _.map {
            e => rebuildArguments(e, groundedMentions)
          }
        })

    }

    // Now, rebuild the events, replacing the arguments with the grounded mentions
    mentions map (m => rebuildArguments(m, groundedTextBoundMentions))
  }

  private case class MentionKey(doc:Document, sent:Int, start:Int, end:Int)

  private object MentionKey {
    def apply(m: Mention): MentionKey = MentionKey(m.document, m.sentence, m.start, m.end)
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
    val (textBound, nonTextBound) = mentions.partition(_.isInstanceOf[TextBoundMention])

    // Ground them
    val groundedNotTextBound = groundMentions(nonTextBound)

    // Add them together
    val groundedTextBound = fetchNestedArguments(groundedNotTextBound)
    val groundedCache:Set[MentionKey] = groundedTextBound.map(MentionKey(_)).toSet
    val ungroundedTextBound = textBound.filterNot(m => groundedCache.contains(MentionKey(m))) // Try to avoid duplicates

    // Return them!
    (doc, groundedNotTextBound ++ ungroundedTextBound  ++ groundedTextBound)
  }
}

object TextReadingPipeline {

  def newOdinEngine(): OdinEngine = {
    val config: Config = ConfigFactory.load()
    val readerType: String = config[String]("ReaderType")
    val readerConfig: Config = config[Config](readerType)
    val odinConfig: Config = readerConfig.withValue("preprocessorType", ConfigValueFactory.fromAnyRef("PassThrough"))
    val odinEngine = OdinEngine.fromConfig(odinConfig)

    odinEngine.annotate("x = 10")
    odinEngine
  }

  def newGrounder(processorOpt: Option[Processor] = None, chosenEngineOpt: Option[String] = None): Grounder = {
    val config: Config = ConfigFactory.load()
    val groundingConfig: Config = config.getConfig("Grounding")

    GrounderFactory.getInstance(groundingConfig, processorOpt, chosenEngineOpt)
  }
}
