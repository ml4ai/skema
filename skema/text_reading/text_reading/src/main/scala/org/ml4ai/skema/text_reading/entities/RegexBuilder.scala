package org.ml4ai.skema.text_reading.entities

import scala.util.matching.Regex

object RegexBuilder {

  def build(strings: Set[String]): Regex = {
    // These will be matched eagerly.  Sort them so the longer words come before their prefixes.
    val string = strings.toSeq.sorted(Ordering[String].reverse).map(Regex.quote).mkString("|")

    string.r
  }
}
