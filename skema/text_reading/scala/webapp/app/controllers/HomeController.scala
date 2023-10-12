package controllers

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory}
import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.clulab.processors.{Document, Sentence}
import org.clulab.serialization.json.stringify
import org.ml4ai.skema.text_reading.TextReadingPipeline
import org.json4s.{JArray, JValue}
import org.ml4ai.skema.text_reading.{CosmosTextReadingPipeline, TextReadingPipelineWithContext}
import org.ml4ai.skema.text_reading.serializer.SkemaJSONSerializer
import org.ml4ai.skema.text_reading.utils.DisplayUtils
import org.slf4j.{Logger, LoggerFactory}
import play.api.libs.json._
import play.api.mvc._

import javax.inject._
import scala.io.Source

/*

Models:
  * MiraEmbeddingsGrounder has an embeddings file, e.g., epidemiology_embeddings_model.ser, but it is memory mapped.
  * CluProcessor has a model, usually glove-840b-300d-10f-kryo, with vectors.  The 10f indicates frequency truncation.
  * CosmosTextReadingPipeline as a model in the PdfConverter, gigaword.ser, but only words and counts are used.

Processors:
  * The OdinEngine has a FastNLPProcessor.
  * The MiraEmbeddingsGrounder has a CluProcessor.
  * DocumentByWord of Pdf2Text has a CluProcessor.

*/

@Singleton
class HomeController @Inject()(cc: ControllerComponents) extends AbstractController(cc) {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
  logger.info("Initializing the OdinEngine ...")

  // Make one of each of these now and share it with the pipelines.
  // This will have a FastNLPProcessor.
  val odinEngineOpt = Some(TextReadingPipeline.newOdinEngine())
  val fastNlpProcessorOpt = Some(odinEngineOpt.get.proc)

  // Use this one instead of the lazy val in MireEmbeddingsGrounder or newGrounder.
  val miraEmbeddingsGrounder = TextReadingPipeline.newGrounder(fastNlpProcessorOpt, chosenEngineOpt = Some("miraembeddings"))

  val processorOpt = None//Some(DocumentByWord.processor)
  // TODO Add the window parameter to the configuration file.
  val cosmosPipeline = new CosmosTextReadingPipeline(contextWindowSize = 3, processorOpt, odinEngineOpt, Some(miraEmbeddingsGrounder))
  // TODO Add the window parameter to the configuration file.
  val plainTextPipeline = new TextReadingPipelineWithContext(processorOpt, odinEngineOpt, Some(miraEmbeddingsGrounder))

  logger.info("Completed Initialization ...")


  // config
  val config = ConfigFactory.load()
  val appVersion: String = config[String]("skema.version")

  // -----------------------------------------------------------------
  // Home page
  // -----------------------------------------------------------------

  def index() = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.index())
  }

  /**
   * Returns the App version using the APP_VERSION ENV variable.
   */
  def version() = Action { Ok(appVersion) }

  def parseSentence(sent: String, showEverything: Boolean) = Action {
    val (doc, eidosMentions) = processPlayText(sent)
    logger.info(s"Sentence returned from processPlayText : ${doc.sentences.head.getSentenceText}")
    val json = mkJson(sent, doc, eidosMentions, showEverything) // we only handle a single sentence

    Ok(json)
  }

  // -----------------------------------------------------------------
  // API functions
  // -----------------------------------------------------------------

  def cosmosJsonToMentions: Action[AnyContent] = Action { request =>
    val json = request.body.asJson.get.toString()
    val ujsonArray = ujson.read(json)
    val ujsonValues = ujsonArray.arr.par.map { ujsonValue =>
      cosmosPipeline.extractMentionsFromJsonAndSerialize(ujsonValue)
    }.seq
    val ujsonResult = ujson.Arr.from(ujsonValues)
    val playJsonResult = ujsonToPlayJson(ujsonResult)

    Ok(playJsonResult)
  }

  def groundStringsToMira(k: Int): Action[AnyContent] = Action { request =>
    val text = request.body.asText.get
    val texts = Source.fromString(text).getLines.map(_.trim).filter(_.nonEmpty).toVector
    val groundingCandidates = miraEmbeddingsGrounder.groundingCandidates(texts, k)
    val jGroundingCandidates = groundingCandidates.map(_.map(_.toJValue).toList).toList
    val json4sResult = JArray(jGroundingCandidates.map(JArray(_)))
    val playJsonResult = json4sToPlayJson(json4sResult)

    Ok(playJsonResult)
  }

  def textFileToMentions: Action[AnyContent] = Action { request =>
    val json = request.body.asJson.get.toString()
    val ujsonArray = ujson.read(json)
    val ujsonValues = ujsonArray.arr.par.map { ujsonValue =>
      val mentions = plainTextPipeline.extractMentionsWithContext(ujsonValue.toString(), contextWindowSize = 3)

      SkemaJSONSerializer.serializeMentions(mentions)
    }.seq
    val ujsonResult = ujson.Arr.from(ujsonValues)
    val playJsonResult = ujsonToPlayJson(ujsonResult)

    Ok(playJsonResult)
  }

  // -----------------------------------------------------------------
  // Swagger pages
  // -----------------------------------------------------------------

  def openAPI(version: String) = Action {
    Ok(views.html.api(version))
  }

  // -----------------------------------------------------------------
  // Backend methods that do stuff :)
  // -----------------------------------------------------------------

  def json4sToPlayJson(jValue: JValue): JsValue = {
    val json = stringify(jValue, pretty = true)
    val playJson = Json.parse(json)

    playJson
  }

  def ujsonToPlayJson(value: ujson.Value): JsValue = {
    val json = ujson.write(value)
    val playJson = Json.parse(json)

    playJson
  }

  def processPlayText(text: String): (Document, Vector[Mention]) = {
    logger.info(s"Processing sentence : ${text}")

    val (doc, mentions) = cosmosPipeline.extractMentions(text, None)
    val sorted = mentions.sortBy(m => (m.sentence, m.getClass.getSimpleName)).toVector

    logger.info(s"Done extracting the mentions ... ")
    logger.info(s"They are : ${mentions.map(m => m.text).mkString(",\t")}")

    // return the sentence and all the mentions extracted ... TODO: fix it to process all the sentences in the doc
    (doc, sorted)
  }

  protected def mkParseObj(sentence: Sentence, sb: StringBuilder): Unit = {
    def getTdAt(option: Option[Array[String]], n: Int): String = {
      val text = if (option.isEmpty) ""
      else option.get(n)

      "<td>" + xml.Utility.escape(text) + "</td>"
    }

    sentence.words.indices.foreach { i =>
      sb
        .append("<tr>")
        .append("<td>" + xml.Utility.escape(sentence.words(i)) + "</td>")
        .append(getTdAt(sentence.tags, i))
        .append(getTdAt(sentence.lemmas, i))
        .append(getTdAt(sentence.entities, i))
        .append(getTdAt(sentence.norms, i))
        .append(getTdAt(sentence.chunks, i))
        .append("</tr>")
    }
  }

  protected def mkParseObj(doc: Document): String = {
    val header =
      """
        |  <tr>
        |    <th>Word</th>
        |    <th>Tag</th>
        |    <th>Lemma</th>
        |    <th>Entity</th>
        |    <th>Norm</th>
        |    <th>Chunk</th>
        |  </tr>
      """.stripMargin
    val sb = new StringBuilder(header)

    doc.sentences.foreach(mkParseObj(_, sb))
    sb.toString
  }

  def mkJson(text: String, doc: Document, mentions: Vector[Mention], showEverything: Boolean): JsValue = {
    logger.info("Found mentions (in mkJson):")
    mentions.foreach(DisplayUtils.displayMention)

    val sent = doc.sentences.head
    val syntaxJsonObj = Json.obj(
      "text" -> text,
      "entities" -> mkJsonFromTokens(doc),
      "relations" -> mkJsonFromDependencies(doc)
    )
    val eidosJsonObj = mkJsonForEidos(text, sent, mentions, showEverything)
    val groundedAdjObj = mkGroundedObj(mentions)
    val parseObj = mkParseObj(doc)

    // These print the html and it's a mess to look at...
    // println(s"Grounded Gradable Adj: ")
    // println(s"$groundedAdjObj")

    Json.obj(
      "syntax" -> syntaxJsonObj,
      "eidosMentions" -> eidosJsonObj,
      "groundedAdj" -> groundedAdjObj,
      "parse" -> parseObj
    )
  }

  def mkGroundedObj(mentions: Vector[Mention]): String = {
    var objectToReturn = ""

    //return events and relations first---those tend to be the ones we are most interested in and having them come first will help avoid scrolling through the entities first
    // collect relation mentions for display
    val relations = mentions.flatMap {
      case m: RelationMention => Some(m)
      case _ => None
    }

    val events = mentions.filter(_ matches "Event")
    if (events.nonEmpty) {
      objectToReturn += s"<h2>Found Events:</h2>"
      for (event <- events) {
        objectToReturn += s"${DisplayUtils.webAppMention(event)}"
      }
    }

    // Entities
    val entities = mentions.filter(_ matches "Entity")
    if (entities.nonEmpty){
      objectToReturn += "<h2>Found Entities:</h2>"
      for (entity <- entities) {
        objectToReturn += s"${DisplayUtils.webAppMention(entity)}"
      }
    }

    objectToReturn += "<br>"
    objectToReturn
  }

  def mkJsonForEidos(sentenceText: String, sent: Sentence, mentions: Vector[Mention], showEverything: Boolean): Json.JsValueWrapper = {
    val topLevelTBM = mentions.flatMap {
      case m: TextBoundMention => Some(m)
      case _ => None
    }
    // collect event mentions for display
    val events = mentions.flatMap {
      case m: EventMention => Some(m)
      case _ => None
    }
    // collect relation mentions for display
    val relations = mentions.flatMap {
      case m: RelationMention => Some(m)
      case _ => None
    }
    // collect triggers for event mentions
    val triggers = events.flatMap { e =>
      val argTriggers = for {
        a <- e.arguments.values
        if a.isInstanceOf[EventMention]
      } yield a.asInstanceOf[EventMention].trigger
      e.trigger +: argTriggers.toSeq
    }
    // collect event arguments as text bound mentions
    val entities = for {
      e <- events ++ relations
      a <- e.arguments.values.flatten
    } yield a match {
      case m: TextBoundMention => m
      case m: RelationMention => new TextBoundMention(m.labels, m.tokenInterval, m.sentence, m.document, m.keep, m.foundBy)
      case m: EventMention => m.trigger
    }
    // generate id for each textbound mention
    val tbMentionToId = (entities ++ triggers ++ topLevelTBM)
      .distinct
      .zipWithIndex
      .map { case (m, i) => (m, i + 1) }
      .toMap
    // return brat output
    Json.obj(
      "text" -> sentenceText,
      "entities" -> mkJsonFromEntities(entities ++ topLevelTBM, tbMentionToId),
      "triggers" -> mkJsonFromEntities(triggers, tbMentionToId),
      "events" -> mkJsonFromEventMentions(events, tbMentionToId),
      "relations" -> (if (showEverything) mkJsonFromRelationMentions(relations, tbMentionToId) else Array[String]())
    )
  }

  def mkJsonFromEntities(mentions: Vector[TextBoundMention], tbmToId: Map[TextBoundMention, Int]): Json.JsValueWrapper = {
    val entities = mentions.map(m => mkJsonFromTextBoundMention(m, tbmToId(m)))
    Json.arr(entities: _*)
  }

  def mkJsonFromTextBoundMention(m: TextBoundMention, i: Int): Json.JsValueWrapper = {
    Json.arr(
      s"T$i",
      HomeController.statefulRepresentation(m).label,
      Json.arr(Json.arr(m.startOffset, m.endOffset))
    )
  }

  def mkJsonFromEventMentions(ee: Seq[EventMention], tbmToId: Map[TextBoundMention, Int]): Json.JsValueWrapper = {
    var i = 0
    val jsonEvents = for (e <- ee) yield {
      i += 1
      mkJsonFromEventMention(e, i, tbmToId)
    }
    Json.arr(jsonEvents: _*)
  }

  def mkJsonFromEventMention(ev: EventMention, i: Int, tbmToId: Map[TextBoundMention, Int]): Json.JsValueWrapper = {
    Json.arr(
      s"E$i",
      s"T${tbmToId(ev.trigger)}",
      Json.arr(mkArgMentions(ev, tbmToId): _*)
    )
  }

  def mkJsonFromRelationMentions(rr: Seq[RelationMention], tbmToId: Map[TextBoundMention, Int]): Json.JsValueWrapper = {
    var i = 0
    val jsonRelations = for (r <- rr) yield {
      i += 1
      mkJsonFromRelationMention(r, i, tbmToId)
    }
    Json.arr(jsonRelations: _*)
  }

  def getArg(r: RelationMention, name: String): TextBoundMention = r.arguments(name).head match {
    case m: TextBoundMention => m
    case m: EventMention => m.trigger
    case r: RelationMention => prioritizedArg(r)//smushIntoTextBound(r) //fixme - this is likely not the right solution...!
  }

  def prioritizedArg(r: RelationMention): TextBoundMention = {
    val priorityArgs = Seq("pitch", "beat", "value")
    val prioritized = r.arguments.filter(a => priorityArgs.contains(a._1)).values.flatten.headOption
    prioritized.getOrElse(r.arguments.values.flatten.head).asInstanceOf[TextBoundMention] //fixme
  }

  def mkJsonFromRelationMention(r: RelationMention, i: Int, tbmToId: Map[TextBoundMention, Int]): Json.JsValueWrapper = {
    val relationArgNames = r.arguments.keys.toSeq
    val head = relationArgNames.head
    val last = relationArgNames.last

    // fixme: this is a temp solution to avoid error caused by the assertion below, but the visualization does not look right
    if (relationArgNames.length > 2) {
      logger.warn("More than three args, webapp will need to be updated to handle!")
    }
//    assert(relationArgNames.length < 3, "More than three args, webapp will need to be updated to handle!")
    Json.arr(
      s"R$i",
      r.label,
      // arguments are hardcoded to ensure the direction (controller -> controlled)
      Json.arr(
        Json.arr(head, "T" + tbmToId(getArg(r, head))),
        Json.arr(last, "T" + tbmToId(getArg(r, last)))
      )
    )
  }

  def mkArgMentions(ev: EventMention, tbmToId: Map[TextBoundMention, Int]): Seq[Json.JsValueWrapper] = {
    val args = for {
      argRole <- ev.arguments.keys
      m <- ev.arguments(argRole)
    } yield {
      val arg = m match {
        case m: TextBoundMention => m
        case m: RelationMention => new TextBoundMention(m.labels, m.tokenInterval, m.sentence, m.document, m.keep, m.foundBy)
        case m: EventMention => m.trigger
      }
      mkArgMention(argRole, s"T${tbmToId.getOrElse(arg, "X")}")
    }
    args.toSeq
  }

  def mkArgMention(argRole: String, id: String): Json.JsValueWrapper = {
    Json.arr(argRole, id)
  }

  def mkJsonFromTokens(doc: Document): Json.JsValueWrapper = {
    var offset = 0

    val tokens = doc.sentences.flatMap { sent =>
      val tokens = sent.words.indices.map(i => mkJsonFromToken(sent, offset, i))
      offset += sent.words.size
      tokens
    }
    Json.arr(tokens: _*)
  }

  def mkJsonFromToken(sent: Sentence, offset: Int, i: Int): Json.JsValueWrapper = {
    Json.arr(
      s"T${offset + i + 1}", // token id (starts at one, not zero)
      sent.tags.get(i), // lets assume that tags are always available
      Json.arr(Json.arr(sent.startOffsets(i), sent.endOffsets(i)))
    )
  }

  def mkJsonFromDependencies(doc: Document): Json.JsValueWrapper = {
    var offset = 1

    val rels = doc.sentences.flatMap { sent =>
      var relId = 0
      val deps = sent.universalEnhancedDependencies.get // lets assume that dependencies are always available
      val rels = for {
        governor <- deps.outgoingEdges.indices
        (dependent, label) <- deps.outgoingEdges(governor)
      } yield {
        val json = mkJsonFromDependency(offset + relId, offset + governor, offset + dependent, label)
        relId += 1
        json
      }
      offset += sent.words.size
      rels
    }
    Json.arr(rels: _*)
  }

  def mkJsonFromDependency(relId: Int, governor: Int, dependent: Int, label: String): Json.JsValueWrapper = {
    Json.arr(
      s"R$relId",
      label,
      Json.arr(
        Json.arr("governor", s"T$governor"),
        Json.arr("dependent", s"T$dependent")
      )
    )
  }
}

object HomeController {

  // fixme: ordering/precedence...
  def statefulRepresentation(mention: Mention): Mention = {
    val stateSuffix = "" // Where is this supposed to come from?

    // If you found something, append the affix to top label and add to the Seq of labels
    if (stateSuffix.nonEmpty) {
      val modifiedLabels = (mention.label + stateSuffix) +: mention.labels
      val out = mention match {
        case tb: TextBoundMention => tb.copy(labels = modifiedLabels)
        case rm: RelationMention => rm.copy(labels = modifiedLabels)
        case em: EventMention => em.copy(labels = modifiedLabels)
      }

      out
    }
    else
      mention
  }
}
