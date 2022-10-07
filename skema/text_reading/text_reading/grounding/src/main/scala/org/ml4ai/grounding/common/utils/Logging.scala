package org.ml4ai.grounding.common.utils

import org.slf4j.Logger
import org.slf4j.LoggerFactory

trait Logging {
  lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)
}
