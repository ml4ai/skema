package org.ml4ai.skema.text_reading

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.clulab.pdf2txt.Pdf2txt
import org.clulab.pdf2txt.common.pdf.TextConverter
import org.clulab.pdf2txt.languageModel.GigawordLanguageModel
import org.clulab.pdf2txt.preprocessor._
import org.clulab.utils.Logging
import org.ml4ai.grounding.{GroundingCandidate, MiraEmbeddingsGrounder}
import org.ml4ai.skema.text_reading.attachments.{GroundingAttachment, MentionLocationAttachment}
import org.ml4ai.skema.text_reading.data.CosmosJsonDataLoader
import org.ml4ai.skema.text_reading.serializer.AutomatesJSONSerializer
import scala.util.matching.Regex

import scala.collection.mutable.ArrayBuffer


class CosmosTextReadingPipeline extends Logging {

  /**
    * Assigns grounding elements to a mention
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
            val topGroundingCandidates = grounder.groundingCandidates(tbm.text).filter {
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
  private val grounder = MiraEmbeddingsGrounder(ontologyFilePath, None, lambda = 10, alpha = 1.0f) // TODO: Fix this @Enrique

  // Odin Engine instantiation
  private val odinEngine = OdinEngine.fromConfig(config)

  // PDF converted to fix pdf tokenization artifacts
  private val pdfConverter = new TextConverter()
  private val languageModel = GigawordLanguageModel()
  private val preprocessors = Array(
    new LinePreprocessor(),
    new ParagraphPreprocessor(),
    new UnicodePreprocessor(),
    new CasePreprocessor(CasePreprocessor.defaultCutoff),
    new NumberPreprocessor(NumberPreprocessor.Hyperparameters()),
    new LigaturePreprocessor(languageModel),
    new LineBreakPreprocessor(languageModel),
    new WordBreakByHyphenPreprocessor(),
    // new WordBreakBySpacePreprocessor(languageModel) // This is by default NeverLanguageModel.
  )
  val pdf2txt = new Pdf2txt(pdfConverter, preprocessors)
  ////

  val numberPattern: Regex = """^\.?(\d)+([.,]?\d*)*$""".r

  // cosmos stores information about each block on each pdf page
  // for each block, we load the text (content) and the location of the text (page_num and block order/index on the page)
  val loader = new CosmosJsonDataLoader

  private val jsonSeparator  = "<::>"

  /**
    * Runs the textReadingPipeline over a cosmos file
    * @param jsonPath Path to the json file to annotate
    * @return Mentions extracted by the TR textReadingPipeline
    */
  def extractMentions(jsonPath: String): Seq[Mention] = {

    // TODO: Make this interpretable
    val textsAndLocations = loader.loadFile(jsonPath)
    val textsAndFilenames = textsAndLocations.map(_.split(jsonSeparator).slice(0, 2).mkString(jsonSeparator))
    val locations = textsAndLocations.map(_.split(jsonSeparator).takeRight(2).mkString(jsonSeparator)) //location = pageNum::blockIdx

    logger.info("started extracting")
    // extract mentions form each text block
    val mentions = for (tf <- textsAndFilenames) yield {
      val Array(rawText, filename) = tf.split(jsonSeparator)
      // Extract mentions and apply grounding

      val text = pdf2txt.process(rawText, maxLoops = 1)

      // Extract mentions and apply grounding
      odinEngine.extractFromText(text, keepText = true, Some(filename)).par.map {
        // Only ground arguments of events and relations, to save time
        case e@(_: EventMention | _: RelationMention) => groundMention(e)
        case m => m
      }.seq

    }

    // store location information from cosmos as an attachment for each mention
    val menWInd = mentions.zipWithIndex
    val mentionsWithLocations = new ArrayBuffer[Mention]()

    for (tuple <- menWInd) {
      // get page and block index for each block; cosmos location information will be the same for all the mentions within one block
      val menInTextBlocks = tuple._1
      val id = tuple._2
      val location = locations(id).split(jsonSeparator).map(loc => loc.split(",").map(_.toInt)) //(_.toDouble.toInt)
      val pageNum = location.head
      val blockIdx = location.last

      for (m <- menInTextBlocks) {
        val filename = m.document.id.getOrElse("unknown_file")
        val newMen = m.withAttachment(new MentionLocationAttachment(filename, pageNum, blockIdx, "MentionLocation"))
        mentionsWithLocations.append(newMen)
      }
    }

    mentionsWithLocations
  }

  /**
    * Extracts the mentions and serializes them into a json string
    * @param jsonPath Path to the json file to annotate
    * @return string with the json representation of the extractions and the document annotations
    */
  def serializeToJson(jsonPath: String): String = ujson.write(AutomatesJSONSerializer.serializeMentions(this.extractMentions(jsonPath)))
}
