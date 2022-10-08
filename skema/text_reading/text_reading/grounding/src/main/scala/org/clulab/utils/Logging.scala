package org.clulab.utils

import org.slf4j.{Logger, LoggerFactory}

trait Logging {
  lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)
}
