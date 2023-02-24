package org.ml4ai.skema.text_reading.entities

import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class TestRegexBuilder extends FlatSpec with Matchers {

  behavior of "RegexBuilder"

  it should "find the very strings that were added" in {
    val strings = Set(
      "abcde",
      "abcdeabcde",
      "abdce"
    )
    val regex = RegexBuilder.build(strings)

    strings.foreach { string =>
      regex.findFirstIn(string) should be(Some(string))
    }
  }

  class StringGenerator(maxLength: Int) {
    val random = new Random(0)

    def generate: String = {
      val length = random.nextInt(maxLength) + 1
      val letters = 0.until(length).map { _ =>
        ('a' + random.nextInt(26)).toChar
      }
      val word = letters.mkString

      word
    }
  }

  it should "match random strings" in {
    val stringGenerator = new StringGenerator(5)

    val strings = 1.to(100).map { _ =>
      stringGenerator.generate
    }.toSet
    val regex = RegexBuilder.build(strings)
    val text = strings.mkString(" ")

    strings.foreach { string =>
      regex.findFirstIn(string) should be(Some(string))
    }
    regex.findFirstIn(text) should be('defined)

    1.to(1000).foreach { _ =>
      val string = stringGenerator.generate
      val expected = strings.exists(string.contains)

      regex.findFirstIn(string).isDefined should be (expected)
    }
  }
}
