package org.ml4ai.skema.text_reading.apps

import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import org.clulab.odin.{Mention, TextBoundMention}
import org.clulab.odin.serialization.json.JSONSerializer
import org.ml4ai.skema.text_reading.data.CosmosJsonDataLoader

import scala.io.Source
import org.clulab.pdf2txt.Pdf2txt
import org.clulab.pdf2txt.common.pdf.TextConverter
import org.clulab.pdf2txt.languageModel.GigawordLanguageModel
import org.clulab.pdf2txt.preprocessor.{CasePreprocessor, LigaturePreprocessor, LineBreakPreprocessor, LinePreprocessor, NumberPreprocessor, ParagraphPreprocessor, UnicodePreprocessor, WordBreakByHyphenPreprocessor, WordBreakBySpacePreprocessor}
import org.ml4ai.grounding.{GroundingCandidate, MiraEmbeddingsGrounder}
import org.ml4ai.skema.text_reading.OdinEngine
import org.ml4ai.skema.text_reading.alignment.AlignmentHandler
import org.ml4ai.skema.text_reading.attachments.{GroundingAttachment, MentionLocationAttachment}
import org.ml4ai.skema.text_reading.scienceparse.ScienceParseClient
import org.ml4ai.skema.text_reading.serializer.AutomatesJSONSerializer
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

object CosmosJsonPipeline extends App {
  val inputPath = args.head


  val generalConfig: Config = ConfigFactory.load()
  val readerType: String = generalConfig.getString("ReaderType")
  val defaultConfig: Config = generalConfig.getConfig(readerType)
  val config: Config = defaultConfig.withValue("preprocessorType", ConfigValueFactory.fromAnyRef("PassThrough"))
  val groundingConfig = generalConfig.getConfig("Grounding")
  val ieSystem = OdinEngine.fromConfig(config)
  var proc = ieSystem.proc
  val serializer = JSONSerializer
  lazy val scienceParse = new ScienceParseClient(domain = "localhost", port = "8080")
  lazy val commentReader = OdinEngine.fromConfigSection("CommentEngine")
  lazy val alignmentHandler = new AlignmentHandler(ConfigFactory.load().getConfig("alignment"))
  protected lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  private val ontologyFilePath = groundingConfig.getString("ontologyPath")
  private val groundingAssignmentThreshold = groundingConfig.getDouble("assignmentThreshold")
  private val grounder = MiraEmbeddingsGrounder(ontologyFilePath, None, 10, 0.0f) // TODO: Fix this @Enrique
  val pdfConverter = new TextConverter()
  val languageModel = GigawordLanguageModel()
  val preprocessors = Array(
    new LinePreprocessor(),
    new ParagraphPreprocessor(),
    new UnicodePreprocessor(),
    new CasePreprocessor(CasePreprocessor.defaultCutoff),
    new NumberPreprocessor(NumberPreprocessor.Hyperparameters()),
    new LigaturePreprocessor(languageModel),
    new LineBreakPreprocessor(languageModel),
    new WordBreakByHyphenPreprocessor(),
    new WordBreakBySpacePreprocessor(languageModel) // This is by default NeverLanguageModel.
  )
  val pdf2txt = new Pdf2txt(pdfConverter, preprocessors)

  // cosmos stores information about each block on each pdf page
  // for each block, we load the text (content) and the location of the text (page_num and block order/index on the page)
  val loader = new CosmosJsonDataLoader
  val textsAndLocations = loader.loadFile(inputPath)
  val textsAndFilenames = textsAndLocations.map(_.split("<::>").slice(0, 2).mkString("<::>"))
  val locations = textsAndLocations.map(_.split("<::>").takeRight(2).mkString("<::>")) //location = pageNum::blockIdx

  println("started extracting")
  // extract mentions form each text block
  val mentions = for (tf <- textsAndFilenames) yield {
    val Array(rawText, filename) = tf.split("<::>")
    // Extract mentions and apply grounding

    val text = pdf2txt.process(rawText, maxLoops = 1)

    ieSystem.extractFromText(text, keepText = true, Some(filename)).par.map {
      case tbm: TextBoundMention => {
        val topGroundingCandidates = grounder.groundingCandidates(tbm.text).filter {
          case GroundingCandidate(_, score) => score >= groundingAssignmentThreshold
        }


        if (topGroundingCandidates.nonEmpty)
          tbm.withAttachment(new GroundingAttachment(topGroundingCandidates))
        else
          tbm
      }
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
    val location = locations(id).split("<::>").map(loc => loc.split(",").map(_.toInt)) //(_.toDouble.toInt)
    val pageNum = location.head
    val blockIdx = location.last

    for (m <- menInTextBlocks) {
      val filename = m.document.id.getOrElse("unknown_file")
      val newMen = m.withAttachment(new MentionLocationAttachment(filename, pageNum, blockIdx, "MentionLocation"))
      mentionsWithLocations.append(newMen)
    }
  }


  val exportedData = ujson.write(AutomatesJSONSerializer.serializeMentions(mentionsWithLocations))

//  println(exportedData)
}
