package org.ml4ai.skema.text_reading

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.clulab.processors.Document
import org.clulab.utils.Logging
import org.ml4ai.skema.text_reading.attachments.GroundingAttachment
import org.ml4ai.skema.text_reading.grounding.{GrounderFactory, GroundingCandidate}
import org.ml4ai.skema.text_reading.mentions.CrossSentenceEventMention
import org.ml4ai.skema.text_reading.serializer.SkemaJSONSerializer

import java.io.File
import scala.language.implicitConversions
import scala.util.matching.Regex

class TextReadingPipeline extends Logging {
  logger.info("Initializing the OdinEngine ...")

  // Read the configuration from the files
  val generalConfig: Config = ConfigFactory.load()
  val readerType: String = generalConfig[String]("ReaderType")
  val defaultConfig: Config = generalConfig[Config](readerType)
  val config: Config = defaultConfig.withValue("preprocessorType", ConfigValueFactory.fromAnyRef("PassThrough"))

  private val groundingConfig: Config = generalConfig.getConfig("Grounding")
  private val groundingAssignmentThreshold = groundingConfig.getDouble("assignmentThreshold")
  private val grounder = GrounderFactory.getInstance(groundingConfig)
  private val numberPattern: Regex = """^\.?(\d)+([.,]?\d*)*$""".r

  // Odin Engine instantiation
  private val odinEngine = OdinEngine.fromConfig(config)
  // Initialize the odin engine
  odinEngine.annotate("x = 10")

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
    val mentionsGroundings = grounder.groundingCandidates(candidatesToGround.map(_.text), k=15).map{
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

  def generateGroundingOutputFilename(file:File): File = {
    val dir = file.getAbsoluteFile.getParent
    val oldName = file.getName.split("\\.").dropRight(1).mkString(".")
    val newName = s"grounding_$oldName.tsv"
    new File(dir, newName)
  }
  def serializeExtractionsForGrounding(extractions:Seq[Mention]): Seq[String] = {
    def getArgs(m:Mention):Option[Map[String, Seq[Mention]]] = m match {
      case e:EventMention =>
        Some(e.arguments)
      case r:RelationMention =>
        Some(r.arguments)
      case _ => None
    }
    // Just keep events and relations relevant to us
    val relevant = extractions.filter{
      case _:TextBoundMention => false
      case _ => true
    }
    // Header
    val header = Seq("Text", "Role", "Context", "Candidate ID", "Candidate Text", "Score", "Annotation").mkString("\t")
    // Compute the annotation chunks for each argument of each event
    val chunks =
      relevant.flatMap{
        event => {
          val context = event.sentenceObj.getSentenceText // Updated this to contain the whole sentence instead of just the event
          getArgs(event) match {
            case Some(args) =>
              args.flatMap {
                case (role, ms) =>
                  ms flatMap {
                    m =>
                      val text = m.text
                      val candidates =
                        m.attachments
                          .toSeq
                          .filter(_.isInstanceOf[GroundingAttachment])
                          .flatMap(_.asInstanceOf[GroundingAttachment].candidates map {
                              c => Seq(c.concept.id, c.concept.name, c.score.toString)
                            }
                          )

                      val x =
                      candidates.zipWithIndex.map {
                        case (c, ix) =>
                          c match {
                            case Seq(id, name, score) =>
                              if (ix == 0)
                                Seq(text, role, context, id, name, score, "").mkString("\t")
                              else
                                Seq("", "", "", id, name, score, "").mkString("\t")
                            case _ => ""
                          }
                      }

                    x
                  }
              }
            case None => Seq.empty[String]
          }
        }
      }
    header::chunks.toList
  }

  /**
    * Serializes the extractions into a string
    * @param extractions
    * @return JSON string that represents the extractions serialized
    */
  def serializeExtractions(extractions:Seq[Mention]):String = ujson.write(SkemaJSONSerializer.serializeMentions(extractions))
}
