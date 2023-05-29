package org.ml4ai.skema.text_reading

import org.clulab.odin.Mention
import org.clulab.pdf2txt.Pdf2txt
import org.clulab.pdf2txt.common.pdf.TextConverter
import org.clulab.pdf2txt.languageModel.GigawordLanguageModel
import org.clulab.pdf2txt.preprocessor._
import org.ml4ai.skema.text_reading.attachments.MentionLocationAttachment
import org.ml4ai.skema.text_reading.data.CosmosJsonDataLoader
import org.ml4ai.skema.text_reading.scenario_context.{ContextEngine, CosmosOrderer, SentenceIndexOrderer}
import org.ml4ai.skema.text_reading.serializer.SkemaJSONSerializer

import scala.collection.mutable.ArrayBuffer



class CosmosTextReadingPipeline(contextWindowSize:Int) extends TextReadingPipeline {

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



  // cosmos stores information about each block on each pdf page
  // for each block, we load the text (content) and the location of the text (page_num and block order/index on the page)
  val loader = new CosmosJsonDataLoader

  private val jsonSeparator  = "<::>"

  /**
    * Runs the textReadingPipeline over a cosmos file
    * @param jsonPath Path to the json file to annotate
    * @return Mentions extracted by the TR textReadingPipeline
    */
  def extractMentionsFromCosmosJson(jsonPath: String): Seq[Mention] = {

    //TODO: Make this interpretable
    val textsAndLocations = loader.loadFile(jsonPath)
    val textsAndFilenames = textsAndLocations.map(_.split(jsonSeparator).slice(0, 2).mkString(jsonSeparator))
    val locations = textsAndLocations.map(_.split(jsonSeparator).takeRight(2).mkString(jsonSeparator)) //location = pageNum::blockIdx

    logger.info(s"Started annotation of $jsonPath")
    // extract mentions form each text block
    val mentions = for (tf <- textsAndFilenames) yield {
      val Array(rawText, filename) = tf.split(jsonSeparator)
      // Extract mentions and apply grounding

      val text = pdf2txt.process(rawText, maxLoops = 1)
      this.extractMentions(text, Some(filename))._2

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

    // Resolve scenario context
    val cosmosOrderer = new CosmosOrderer(mentionsWithLocations)
    val scenarioContextEngine = new ContextEngine(windowSize = contextWindowSize, mentionsWithLocations, cosmosOrderer)
    val mentionsWithScenarioContext = mentionsWithLocations map scenarioContextEngine.resolveContext
    logger.info(s"Finished annotation of $jsonPath")
    mentionsWithScenarioContext
  }

  /**
    * Extracts the mentions and serializes them into a json string
    * @param jsonPath Path to the json file to annotate
    * @return string with the json representation of the extractions and the document annotations
    */
  def extractMentionsFromJsonAndSerialize(jsonPath: String): String = this.serializeExtractions(this.extractMentionsFromCosmosJson(jsonPath))
}
