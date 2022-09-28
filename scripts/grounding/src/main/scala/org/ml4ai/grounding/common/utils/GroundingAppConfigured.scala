package org.ml4ai.grounding.common.utils

import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory

trait GroundingAppConfigured extends Configured {
  // This line doesn't work if there is a leading / in the resource name.  I tried.
  lazy val config = ConfigFactory.parseResourcesAnySyntax("GroundingApp")

  override def getConf: Config = config
}
