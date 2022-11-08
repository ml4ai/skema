package org.ml4ai.skema.test

import ai.lum.common.ConfigUtils._
import com.typesafe.config.{Config, ConfigFactory}
import org.clulab.aske.automates.OdinEngine
import org.clulab.aske.automates.OdinEngine._
import org.clulab.aske.automates.attachments.AutomatesAttachment
import org.clulab.aske.automates.utils.MentionUtils
import org.clulab.odin.Mention
import org.ml4ai.skema.common.test.Test

class ExtractionTest(val ieSystem: OdinEngine) extends Test {
  def this(config: Config = ConfigFactory.load("test")) = this(TestUtils.newOdinSystem(config[Config]("TextEngine")))

  def extractMentions(text: String): Seq[Mention] = TestUtils.extractMentions(ieSystem, text)

  // Event Specific

  def testDescriptionEvent(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testBinaryEvent(mentions, DESCRIPTION_LABEL, VARIABLE_ARG, DESCRIPTION_ARG, desired)
  }

  def testFunctionEvent(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testBinaryEvent(mentions, FUNCTION_LABEL, FUNCTION_OUTPUT_ARG, FUNCTION_INPUT_ARG, desired)
  }

  def testModelDescrsEvent(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testBinaryEvent(mentions, MODEL_DESCRIPTION_LABEL, MODEL_NAME_ARG, MODEL_DESCRIPTION_ARG, desired)
  }

  def testModelLimitEvent(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testBinaryEvent(mentions, MODEL_LIMITATION_LABEL, MODEL_NAME_ARG, MODEL_DESCRIPTION_ARG, desired)
  }

  def testUnitEvent(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testBinaryEvent(mentions, UNIT_LABEL, VARIABLE_ARG, UNIT_ARG, desired)
  }

  def testParameterSettingEvent(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testBinaryEvent(mentions, PARAMETER_SETTING_LABEL, VARIABLE_ARG, VALUE_ARG, desired)
  }

  def testParameterSettingEventInterval(mentions: Seq[Mention], desired: Seq[(String, Seq[String])]): Unit = {
    testThreeArgEvent(mentions, INTERVAL_PARAMETER_SETTING_LABEL, VARIABLE_ARG, VALUE_LEAST_ARG, VALUE_MOST_ARG, desired)
  }

  // General Purpose

  def withinOneSentenceTest(mentions: Seq[Mention]): Unit = {
    // makes sure all the args of a mentioned are contained within one sentence (note: not same for cross-sentence mentions)
    for (m <- mentions) {
      for (arg <- m.arguments) {
        arg._2.head.sentence shouldEqual m.sentence
      }
    }
  }

  def testTextBoundMention(mentions: Seq[Mention], eventType: String, desired: Seq[String]): Unit = {
    val found = mentions.filter(_ matches eventType).map(_.text)
    found.length should be(desired.size)

    desired.foreach(d => found should contain(d))
  }

  def testIfHasAttachmentType(mentions: Seq[Mention], attachmentType: String): Unit = {
    for (f <- mentions) {
      TestUtils.extractorAligner.returnAttachmentOfAGivenTypeOption(f.attachments, attachmentType).isDefined shouldBe true
    }
  }

  def testIfHasAttachments(mentions: Seq[Mention]): Unit = {
    mentions.foreach(f => testIfHasAttachment(f))
  }

  def testIfHasAttachment(mention: Mention): Unit = {
    mention.attachments.nonEmpty shouldBe true
  }

  //used for parameter setting tests where the setting is an interval
  def testThreeArgEvent(mentions: Seq[Mention], eventType: String, arg1Role: String, arg2Role: String, arg3Role: String, desired: Seq[(String, Seq[String])]): Unit = {

    val found = mentions.filter(_ matches eventType)
    found.length should be(desired.size)

    // note: assumes there's only one of each variable
    val grouped = found.groupBy(_.arguments(arg1Role).head.text)
    // we assume only one variable (arg1) arg!
    for {
      (desiredVar, desiredParameters) <- desired
      correspondingMentions = grouped.getOrElse(desiredVar, Seq())
    } testThreeArgEventString(correspondingMentions, arg1Role, desiredVar, arg2Role, desiredParameters.head, arg3Role, desiredParameters.last)

  }

  def testBinaryEvent(mentions: Seq[Mention], eventType: String, arg1Role: String, arg2Role: String, desired: Seq[(String, Seq[String])]): Unit = {
    val found = mentions.filter(_ matches eventType)
    found.length should be(desired.size)



    val grouped = found.groupBy(_.arguments(arg1Role).head.text) // we assume only one variable (arg1) arg!
    for {
      (desiredVar, desiredDescrs) <- desired
      correspondingMentions = grouped.getOrElse(desiredVar, Seq())
    } testBinaryEventStrings(correspondingMentions, arg1Role, desiredVar, arg2Role, desiredDescrs)
  }


  def testBinaryEventStrings(ms: Seq[Mention], arg1Role: String, arg1String: String, arg2Role: String, arg2Strings: Seq[String]) = {
    val identifierDescriptionPairs = for {
      m <- ms
      a1 <- m.arguments.getOrElse(arg1Role, Seq()).map(MentionUtils.getMentionText(_))
      a2 <- m.arguments.getOrElse(arg2Role, Seq()).map(MentionUtils.getMentionText(_))
    } yield (a1, a2)

    arg2Strings.foreach(arg2String => identifierDescriptionPairs should contain ((arg1String, arg2String)))
  }

  def testUnaryEvent(mentions: Seq[Mention], eventType: String, arg1Role: String, desired: Seq[String]): Unit = {
    val found = mentions.filter(_ matches eventType)
    found.length should be(desired.size)
    val grouped = found.groupBy(_.arguments(arg1Role).head.text) // we assume only one variable (arg1) arg!
    // when desired matches the text of the input arg, corresponding mentions are returned and the test passes
    // when the text does not match, there is no key in grouped for that so the returned seq is empty, and we get a failing test//
    for {
      desiredFragment <- desired
      correspondingMentions = grouped.getOrElse(desiredFragment, Seq())
    } testUnaryEventStrings(correspondingMentions, arg1Role, eventType, desired)
  }

  def testUnaryEventStrings(ms: Seq[Mention], arg1Role: String, eventType: String, arg1Strings: Seq[String]) = {
    val functionFragment = for {
      m <- ms
      a1 <- m.arguments.getOrElse(arg1Role, Seq()).map(MentionUtils.getMentionText(_))
    } yield a1
    arg1Strings.foreach(arg1String => functionFragment should contain (arg1String))
  }

  //used for parameter setting tests where the setting is an interval
  def testThreeArgEventString(ms: Seq[Mention], arg1Role: String, arg1String: String, arg2Role: String, arg2String: String, arg3Role: String, arg3String: String): Unit = {

    // assumes there is one of each arg
    val varMinMaxSettings =  for {
      m <- ms
      a1 = if (m.arguments.contains(arg1Role)) m.arguments.get(arg1Role).head.map(_.text).head else ""
      a2 = if (m.arguments.contains(arg2Role)) m.arguments.get(arg2Role).head.map(_.text).head else ""
      a3 = if (m.arguments.contains(arg3Role)) m.arguments.get(arg3Role).head.map(_.text).head else ""
    } yield (a1, a2, a3)

    varMinMaxSettings should contain ((arg1String, arg2String, arg3String))
  }

  def mentionHasArguments(m: Mention, argName: String, argValues: Seq[String]): Unit = {
    // Check that the desired number of that argument were found
    val selectedArgs = m.arguments.getOrElse(argName, Seq())
    selectedArgs should have length(argValues.length)

    // Check that each of the arg values is found
    val argStrings = selectedArgs.map(_.text)
    argValues.foreach(argStrings should contain (_))
  }


  def getAttachmentJsonsFromArgs(mentions: Seq[Mention]): Seq[ujson.Value] = {
    val allAttachmentsInEvent = for {
      m <- mentions
      arg <- m.arguments
      a <- arg._2
      if a.attachments.nonEmpty
      att <- a.attachments
    } yield att.asInstanceOf[AutomatesAttachment].toUJson
    allAttachmentsInEvent
  }

}
