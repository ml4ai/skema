package org.ml4ai.skema.text_reading.alignment

import java.io.File
import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory}
import ai.lum.common.FileUtils._
import org.ml4ai.skema.text_reading.apps.ExtractAndAlign.{COMMENT_TO_GLOBAL_VAR, EQN_TO_GLOBAL_VAR, GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT, GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_IDENTIFIER, GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT, GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER, GLOBAL_VAR_TO_UNIT_VIA_CONCEPT, GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER, SRC_TO_COMMENT, allLinkTypes}
import org.clulab.embeddings.SanitizedWordEmbeddingMap
import org.clulab.utils.Sourcer
import org.ml4ai.skema.test.TestAlignment
import org.ml4ai.skema.text_reading.apps.{AlignmentArguments, ExtractAndAlign}
import org.ml4ai.skema.text_reading.utils.AlignmentJsonUtils
import ujson.Value

/* Tests the alignment payload created based on the toy double-epidemic-and-chime files in /test/resources;
  should changes need to be made to the toy document, the latex template is stored under /test/resources/toy_document_tex;
  if changes are made, the new pdf will need to be processed with science parse (or cosmos) and the payload will need to be recreated
  (can use align_experiment.py for that); the sample grfn is only there so that align_experiment is runnable---copy the testing source variables from the testing payload and paste those into the newly created payload;
  Currently, there are three types of tests:
  1) the test that checks that all supported links are made (the toy document contains all of them); the toy document will need to be updated if new link types are added;
  2) a set of tests (one for every identifier that occurs in the toy doc) that check if the identifier got correct alignments for every link type (template at the bottom of the file; to ignore negative tests, add a line like this to direct or indirect desired maps: <link_type> -> ("", failingNegative)); if there are two possible values, can add them in desired as a "::"-separated string; interval values are "||"-separated;
  3) tests for comment_to_gvar tests (the only type of indirect link we have now); there are few because the alignment for them is very basic and whatever can go wrong will be obvious from these two tests.

 */
class TestAlign extends TestAlignment {

  // This is a failingTest.  It requires Python and external files.
  val enabled = false
  val config: Config = ConfigFactory.load("test.conf")

  // Try to load this huge thing just once.
  val w2vPath: String = config[String]("alignment.w2vPath")
  lazy val w2v =
      if (w2vPath == ExtractAndAlign.w2vPath)
        ExtractAndAlign.w2v
      else {
        val vectors = Sourcer.sourceFromResource(w2vPath)
        val w2v = new SanitizedWordEmbeddingMap(vectors, None, false)

        w2v
      }

  val relevantArgs: List[String] = config[List[String]]("alignment.relevantArgs")
  lazy val alignmentHandler = new AlignmentHandler(w2v, relevantArgs.toSet)
  // get general configs
  val serializerName: String = config[String]("apps.serializerName")
  val numAlignments: Int = config[Int]("apps.numAlignments")
  val numAlignmentsSrcToComment: Int = config[Int]("apps.numAlignmentsSrcToComment")
  val scoreThreshold: Int = config[Int]("apps.scoreThreshold")
  val groundToSVO: Boolean = config[Boolean]("apps.groundToSVO")
  val groundToWiki: Boolean = false //config[Boolean]("apps.groundToWiki")
  val maxSVOgroundingsPerVar: Int = config[Int]("apps.maxSVOgroundingsPerVar")
  val appendToGrFN: Boolean = config[Boolean]("apps.appendToGrFN")

  // alignment-specific configs

  val debug: Boolean = config[Boolean]("alignment.debug")
  val inputDir = new File(getClass.getResource("/").getFile)
  val payLoadFileName: String = config[String]("alignment.unitTestPayload")
  val payloadFile = new File(inputDir, payLoadFileName)
  val payloadPath: String = payloadFile.getAbsolutePath
  val payloadJson: ujson.Value = ujson.read(payloadFile.readString())
  val jsonObj: ujson.Value = payloadJson.obj

  val argsForGrounding: AlignmentArguments = AlignmentJsonUtils.getArgsForAlignment(payloadPath, jsonObj, groundToSVO, groundToWiki, serializerName)

  lazy val groundings: ujson.Value = ExtractAndAlign.groundMentions(
    payloadJson,
    argsForGrounding.identifierNames,
    argsForGrounding.identifierShortNames,
    argsForGrounding.descriptionMentions,
    argsForGrounding.parameterSettingMentions,
    argsForGrounding.intervalParameterSettingMentions,
    argsForGrounding.unitMentions,
    argsForGrounding.commentDescriptionMentions,
    argsForGrounding.equationChunksAndSource,
    argsForGrounding.svoGroundings,
    argsForGrounding.wikigroundings,
    groundToSVO,
    groundToWiki,
    saveWikiGroundings = false,
    maxSVOgroundingsPerVar,
    alignmentHandler,
    Some(numAlignments),
    Some(numAlignmentsSrcToComment),
    scoreThreshold,
    appendToGrFN,
    debug
  )

  lazy val links = groundings.obj("links").arr
  lazy val extractedLinkTypes = links.map(_.obj("link_type").str).distinct

  failingTest should "have all the link types" in {
    val allLinksTypesFlat = allLinkTypes.obj.filter(_._1 != "disabled").obj.flatMap(obj => obj._2.obj.keySet).toSeq
    val overlap = extractedLinkTypes.intersect(allLinksTypesFlat)
    overlap.length == extractedLinkTypes.length  shouldBe true
    overlap.length == allLinksTypesFlat.length shouldBe true
  }

  override def runAllAlignTests(variable: String, directLinks: Map[String, Seq[Value]], indirectLinks: Map[String, Seq[(String, Double)]],
      directDesired: Map[String, Tuple2[String, String]], indirectDesired: Map[String, Tuple2[String, String]]): Unit = {
    if (enabled)
      super.runAllAlignTests(variable, directLinks, indirectLinks, directDesired, indirectDesired)
  }

  if (enabled)
  {
    val idfr = "R0" // basic reproduction number
    behavior of idfr

    val directDesired = Map(
      GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("microbes per year::mm", passingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("m2/year", passingTestString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("2.71", passingTestString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("100", passingTestString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_IDENTIFIER ->("1", passingTestString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT ->("0.5||1", passingTestString),
      EQN_TO_GLOBAL_VAR -> ("R_0", passingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("Rb", passingTestString)
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("Rb",passingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)
  {
    val idfr = "c" // number of people exposed
    behavior of idfr

    val directDesired = Map(
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      EQN_TO_GLOBAL_VAR -> ("", failingNegativeString),
      COMMENT_TO_GLOBAL_VAR -> ("", failingNegativeString)
    )

    val indirectDesired = Map(//Map.empty[String, (String, String)]
      SRC_TO_COMMENT -> ("", failingNegativeString)
    )
    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)
  {
    val idfr = "β" //fixme: maybe if there is very little text for variable, make aligner depend more on the variable?
    behavior of idfr

    val directDesired = Map(
      EQN_TO_GLOBAL_VAR -> ("beta", passingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("beta", failingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString)
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("beta",failingTestString)
    )
    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)
  }

  if(enabled)
  {
    val idfr = "γ"
    behavior of idfr

    val directDesired = Map(
      EQN_TO_GLOBAL_VAR -> ("gamma", passingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("gamma", failingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString)
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("gamma", failingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)
  }

  if (enabled)
  {
    val idfr = "A" // virus
    behavior of idfr

    val directDesired = Map(//Map.empty[String, (String, String)]
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      COMMENT_TO_GLOBAL_VAR -> ("", failingNegativeString)


    )

    val indirectDesired = Map(//Map.empty[String, (String, String)]
      SRC_TO_COMMENT -> ("", failingNegativeString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)
  {
    val idfr = "a" // removal rate of infectives
    behavior of idfr

    val directDesired = Map(
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("2/3::3", passingTestString),
      EQN_TO_GLOBAL_VAR -> ("a", passingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("a", passingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("a",passingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)
  {
    val idfr = "r" // infection rate
    behavior of idfr

    val directDesired = Map(
      GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("germs per second", passingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("germs per second", passingTestString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("65::9.788 × 10-8", passingTestString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("0.5", passingTestString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_IDENTIFIER ->("negative", failingTestString), // need processing for word param settings
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT ->("0.2||5.6", passingTestString),
      EQN_TO_GLOBAL_VAR -> ("r", passingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("inc_inf", failingTestString) // got misaligned to 'removal rate of infectives'
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("inc_inf",failingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }


  if (enabled)
  {

    val idfr = "I" // infected
    behavior of idfr

    val directDesired = Map(
      GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("individuals", failingTestString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("10", passingTestString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_IDENTIFIER ->("0", failingTestString),// in text it's "positive", need something for word to param setting conversion + need to test intervals better, e.g., the first one in tuple is lower bound and second is upper OR it does not matter for align - quality of extractions is in param setting testing; here, the important part is being linked properly
      EQN_TO_GLOBAL_VAR -> ("I", failingTestString), // I is not in eq but there's I(0), I(t), and I_P; what to align? Or nothing?
      COMMENT_TO_GLOBAL_VAR -> ("I", failingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT ->("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT ->("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT ->("", failingNegativeString)
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("I",failingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)
  {
    val idfr = "R"
    behavior of idfr

    val directDesired = Map(
      EQN_TO_GLOBAL_VAR -> ("R", failingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("R", failingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("", failingNegativeString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("", failingNegativeString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString),
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("R", failingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)
  {
    val idfr = "τ"
    behavior of idfr

    val directDesired = Map(
      GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("mm", passingTestString),
      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("mm::millimeters", passingTestString), // both values possible
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("450", passingTestString),
      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("32", passingTestString),
      EQN_TO_GLOBAL_VAR -> ("tau", passingTestString),
      COMMENT_TO_GLOBAL_VAR -> ("t_a", passingTestString)
    )

    val indirectDesired = Map(
      SRC_TO_COMMENT -> ("t_a",passingTestString)
    )

    val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
    runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

  }

  if (enabled)

    {
      val idfr = "S"
      behavior of idfr

      val directDesired = Map(
        GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("people", failingTestString),
        GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("6.8 millions", failingTestString),
        GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("4.5 million", failingTestString), //fixme: how did 6.8 get attached to S?
        EQN_TO_GLOBAL_VAR -> ("S", passingTestString),
        COMMENT_TO_GLOBAL_VAR -> ("S", failingTestString),
        GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("", failingNegativeString),
        GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT -> ("", failingNegativeString)
      )

      val indirectDesired = Map(
        SRC_TO_COMMENT -> ("S", failingTestString)
      )

      val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
      runAllAlignTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)

    }

  /* INDIRECT LINK TESTS
  for now, the only indirect link is source to comment
   */
  lazy val src_comment_links = links.filter(_.obj("link_type").str == SRC_TO_COMMENT)

  failingTest should "have a src to comment element for source variable a" in {
    src_comment_links.exists(l => l.obj("element_1").str.split("::").last == "a" & l.obj("element_2").str.split("::").last == "a" && l.obj("score").num == 1) shouldBe true
  }

  failingTest should "have a src to comment element for source variable gamma" in {
    src_comment_links.exists(l => l.obj("element_1").str.split("::").last == "gamma" & l.obj("element_2").str.split("::").last == "gamma" && l.obj("score").num == 1) shouldBe true
  }



  //  // template
  //  {
  //    val idfr = "E"
  //    behavior of idfr
  //
  //    val directDesired = Map(
  //      GLOBAL_VAR_TO_UNIT_VIA_IDENTIFIER -> ("E", passingTest),
  //      GLOBAL_VAR_TO_UNIT_VIA_CONCEPT -> ("E", passingTest),
  //      GLOBAL_VAR_TO_PARAM_SETTING_VIA_IDENTIFIER -> ("E", passingTest),
  //      GLOBAL_VAR_TO_PARAM_SETTING_VIA_CONCEPT -> ("E", passingTest),
  //      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_IDENTIFIER ->("E", passingTest),
  //      GLOBAL_VAR_TO_INT_PARAM_SETTING_VIA_CONCEPT ->("E", passingTest),
  //      EQN_TO_GLOBAL_VAR -> ("E", passingTest),
  //      COMMENT_TO_GLOBAL_VAR -> ("E", passingTest)
  //    )
  //    //
  //    val indirectDesired = Map(
  //      SRC_TO_COMMENT -> ("E",passingTest)
  //    )
  //    //
  //  val (directLinks, indirLinks) = getLinksForGvar(idfr, links)
  //  runAllTests(idfr, directLinks, indirLinks, directDesired, indirectDesired)
  //    //
  //  }

}
