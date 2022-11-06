package org.ml4ai.grounding.common.utils

import org.scalactic.source.Position
import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.Tag

class TesterTag extends Tag("TesterTag")

class Test extends FlatSpec with Matchers {
  val passingTest = it
  val failingTest = ignore
  val brokenSyntaxTest = ignore
  val toDiscuss = ignore

  type Inable = { def in(testFun: => Any)(implicit pos: Position): Unit }
  type Shouldable = { def should(string: String): Inable }

  object Nobody extends TesterTag
  object Somebody extends TesterTag
  object Andrew extends TesterTag
  object Becky extends TesterTag
  object Masha extends TesterTag
  object Interval extends TesterTag
  object DiscussWithModelers extends TesterTag // i.e., Clay and Adarsh
}
