package org.ml4ai.skema.test

import org.ml4ai.skema.text_reading.apps.ExtractAndAlign.{allLinkTypes, whereIsGlobalVar, whereIsNotGlobalVar}
import org.ml4ai.skema.text_reading.apps.AlignmentBaseline
import ujson.Value

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

class TestAlignment extends Test {
  val passingTestString = "passing"
  val failingTestString = "failing"
  val failingNegativeString = "failingNegative"
  /*** TEST TYPES*/
  // DIRECT LINK TEST

  def runFailingTest(whichTest: String): Unit = {
    ignore should whichTest in {
      1 shouldEqual 1
    }
  }
  def topDirectLinkTest(idf: String, desired: String, directLinks: Map[String, Seq[Value]],
                        linkType: String, status: String): Unit = {

    val threshold = allLinkTypes("direct").obj(linkType).num
    if (status == "passing") {
      it should f"have a correct $linkType link for global var ${idf}" in {
        val topScoredLink = directLinks(linkType).sortBy(_.obj("score").num).reverse.head
        // which element in this link type we want to check
        val whichLink = whereIsNotGlobalVar(linkType)

        // element 1 of this link (eq gl var) should be E
        desired.split("::") should contain (topScoredLink(whichLink).str.split("::").last)
        topScoredLink("score").num >= threshold shouldBe true
      }
    } else {
      val failingMessage = if (status=="failingNegative") {
        f"have NO $linkType link for global var ${idf}"
      } else {
        f"have a correct $linkType link for global var ${idf}"
      }
      runFailingTest(failingMessage)
    }

  }

  def negativeDirectLinkTest(idf: String, directLinks: Map[String, Seq[Value]],
                             linkType: String): Unit = {
    it should s"have NO ${linkType} link for global var $idf" in {
      val threshold = allLinkTypes("direct").obj(linkType).num
      val condition1 = Try(directLinks.keys.toList.contains(linkType) should be(false)).isSuccess
      val condition2 = Try(directLinks
      (linkType).sortBy(_.obj("score").num).reverse.head("score").num > threshold shouldBe false).isSuccess
      assert(condition1 || condition2)
    }
  }


  // INDIRECT LINK TESTS

  def topIndirectLinkTest(idf: String, desired: String, inDirectLinks: Map[String,
    Seq[(String, Double)]],
                          linkType: String, status: String): Unit = {
    // todo: use threshold here now that we are saving it
    val threshold = allLinkTypes("indirect").obj(linkType).num
    if (status == "passing") {
      it should f"have a correct $linkType link for global var ${idf}" in {
        // these are already sorted
        val topScoredLink = inDirectLinks(linkType)
//          for (l <- topScoredLink) println(">>>" + l)
        topScoredLink.head._1.split("::").last shouldEqual desired
        topScoredLink.head._2 > threshold shouldBe true
      }
    } else {
      runFailingTest(f"have a correct $linkType link for global var ${idf}")
    }
  }


  def negativeIndirectLinkTest(idf: String, indirectLinks: Map[String, Seq[(String, Double)]],
                               linkType: String): Unit = {
    it should s"have NO ${linkType} link for global var $idf" in {

      val threshold = allLinkTypes("indirect").obj(linkType).num
      val condition1 = Try(indirectLinks.keys.toList.contains(linkType) should be(false)).isSuccess
      val condition2 = Try(indirectLinks
      (linkType).head._2 > threshold shouldBe false).isSuccess
      assert(condition1 || condition2)
    }
  }


  // Alignment testing utils
  def getLinksWithIdentifierStr(identifierName: String, allLinks: Seq[Value], inclId: Boolean): Seq[Value] = {

    // when searching all links, can't include element uid, but when we search off of intermediate node (e.g., comment identifier when searching for gvar to src code alignment for testing, have to use uid
    val toReturn = if (inclId) {
      allLinks.filter(l => l.obj("element_1").str == identifierName || l.obj("element_2").str == identifierName)
    } else {
      allLinks.filter(l => l.obj("element_1").str.split("::").last == identifierName || l.obj("element_2").str.split("::").last == identifierName)

    }
    toReturn
  }

  // return indirect links of a given type as a list of strings per each intermediate node
  def findIndirectLinks(allDirectVarLinks: Seq[Value], allLinks: Seq[Value], linkTypeToBuildOffOf: String,
      indirectLinkType: String, nIndirectLinks: Int): Map[String, Seq[(String, Double)]] = {
    val indirectLinkEndNodes = new ArrayBuffer[(String,Double)]()
    val allIndirectLinks = new ArrayBuffer[Value]()
    // we have links for some var, e.g., I(t)
    // through one of the existing links, we can get to another type of node
    val topNDirectLinkOfTargetTypeSorted = allDirectVarLinks.filter(_.obj("link_type").str==linkTypeToBuildOffOf).sortBy(_.obj("score").num).reverse.slice(0, nIndirectLinks)
    // keep for debugging
//      for (tdl <- topNDirectLinkOfTargetTypeSorted) {
//        println("dir link: " + tdl)
//      }
    val sortedIntermNodeNames = new ArrayBuffer[String]()


    for (dl <- topNDirectLinkOfTargetTypeSorted) {
      // get intermediate node of indirect link - for comment_to_gvar link, it's element_1
      val intermNodeJustName = linkTypeToBuildOffOf match {
        case "comment_to_gvar" => dl("element_1").str
        case _ => ???
      }
      sortedIntermNodeNames.append(intermNodeJustName)


      val indirectLinksForIntermNode = getLinksWithIdentifierStr(intermNodeJustName, allLinks, true).filter(_.obj
      ("link_type").str == indirectLinkType).sortBy(_.obj("score").num).reverse
      for (il <- indirectLinksForIntermNode) {

        allIndirectLinks.append(il)
//          println("indir links per interm node: " + il)
      }

    }


    // return only the ones of the given type
    val groupedByElement2 = allIndirectLinks.groupBy(_.obj("element_2").str)//.map(gr => (gr._1,gr._2.sortBy(_.obj("score").num)))
//      for (g <- groupedByElement2) {
//        println("G: " + g._1)
//        for (i <- g._2) {
//          println("=>" + i)
//        }
//      }

    val maxLinksPerIntermNode = groupedByElement2.maxBy(_._2.length)._2.length

    for (i <- 0 until maxLinksPerIntermNode) {
      for (j <- 0 until sortedIntermNodeNames.length) {
        val intermNodeName = sortedIntermNodeNames(j)

        val endNode = groupedByElement2(intermNodeName).map(l => (l.obj("element_1").str, l.obj("score").num))
        if (endNode.length > i) {
          indirectLinkEndNodes.append(endNode(i))
        }

      }
    }

    Map(indirectLinkType -> indirectLinkEndNodes)
  }

  def printGroupedLinksSorted(links: Map[String, Seq[Value]]): Unit = {
    for (gr <- links) {
      for (i <- gr._2.sortBy(_.obj("score").num).reverse) {
        println(i)
      }
      println("----------")
    }
  }

  def printIndirectLinks(indirectLinks: Map[String, Seq[String]]): Unit = {
    for (l <- indirectLinks) {
      println("link type: " + l._1)
      println("linked nodes: " + l._2.mkString(" :: "))
    }
  }

  def getLinksForGvar(idfr: String, allLinks: Seq[Value]): (Map[String, Seq[Value]], Map[String, Seq[(String, Double)]]) = {
    val maybeGreek = AlignmentBaseline.replaceGreekWithWord(idfr, AlignmentBaseline.greek2wordDict.toMap).replace("\\\\", "")
    // links are returned as maps from link types to a) full links for direct links and b) sequences of elements linked indirectly to the global var
    // sorted by score based on the two edges in the indirect link
    // all links that contain the target text global var string
    // and only the links that contain global vars
    val allDirectLinksForIdfr = if (idfr == maybeGreek) {
      // this means this is not a greek letter, so proceed as ususal
      getLinksWithIdentifierStr(idfr, allLinks, false).filter(link => whereIsGlobalVar.contains(link.obj("link_type").str))
    } else {
      getLinksWithIdentifierStr(idfr, allLinks, false).filter(link => whereIsGlobalVar.contains(link.obj("link_type").str)) ++    getLinksWithIdentifierStr(maybeGreek, allLinks, false).filter(link => whereIsGlobalVar.contains(link.obj("link_type").str))
    }

//      for (l <- allDirectLinksForIdfr) println("link: " + l)

    // filtering out the links with no idfr with the correct idf; can't have the link uid in the test itself bc those
    // are randomly generated on every run
    // this has to be one of the links with global variable

    val idfrWithIdFromMultipleLinks = new ArrayBuffer[String]()
    for (l <- allDirectLinksForIdfr) {
      val linkType = l("link_type").str
      val whichElement = whereIsGlobalVar(linkType)
      val fullIdfrUid = l(whichElement).str
      if (fullIdfrUid.split("::").last == idfr) {
        idfrWithIdFromMultipleLinks.append(fullIdfrUid)
      }

    }

    val fullIdfrUid = idfrWithIdFromMultipleLinks.head
//      println("full idfr uid: " + fullIdfrUid)
    val onlyWithCorrectIdfr = allDirectLinksForIdfr.filter(link => link(whereIsGlobalVar(link("link_type").str)).str ==
      fullIdfrUid)
    // group those by link type - in testing, will just be checking the rankings of links of each type
    val directLinksGroupedByLinkType = onlyWithCorrectIdfr.groupBy(_.obj("link_type").str)

    // keep for debug
//      for (l <- directLinksGroupedByLinkType) {
//        println(s"---${l._1}---")
//        for (i <- l._2.sortBy(el => el("score").num).reverse) println(">> " + i)
//      }

    // get indirect links; currently, it's only source to comment links aligned through comment (comment var is the intermediate node)
    val indirectLinks = if (directLinksGroupedByLinkType.contains("comment_to_gvar")) {
      findIndirectLinks(onlyWithCorrectIdfr, allLinks, "comment_to_gvar", "source_to_comment", 3)

    } else null


//      for (linkGr <- indirectLinks) {
//        println(s"===${linkGr._1}===")
//        for (link <- linkGr._2) {
//          println(link)
//        }
//      }

    (directLinksGroupedByLinkType, indirectLinks)
  }


  def runAllAlignTests(variable: String, directLinks: Map[String, Seq[Value]], indirectLinks: Map[String, Seq[(String, Double)]],
      directDesired: Map[String, Tuple2[String, String]], indirectDesired: Map[String, Tuple2[String, String]]): Unit = {
    for (dl <- directDesired) {
      val desired = dl._2._1
      val linkType = dl._1
      val status = dl._2._2
      topDirectLinkTest(variable, desired, directLinks, linkType, status)
    }

//      for (dl <- indirectLinks) println("indir: " + dl._1 + " " + dl._2)
    for (dl <- indirectDesired) {
      val desired = dl._2._1
      val linkType = dl._1
      val status = dl._2._2
      topIndirectLinkTest(variable, desired, indirectLinks, linkType, status)
    }

    for (dlType <- allLinkTypes("direct").obj.keys) {
      if (!directDesired.contains(dlType)) {
        negativeDirectLinkTest(variable, directLinks, dlType)
      }
    }

    for (dlType <- allLinkTypes("indirect").obj.keys) {
      if (!indirectDesired.contains(dlType)) {
        negativeIndirectLinkTest(variable, indirectLinks, dlType)
      }
    }
  }
}
