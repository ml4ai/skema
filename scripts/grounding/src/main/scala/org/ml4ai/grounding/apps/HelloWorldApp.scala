package org.ml4ai.grounding.apps

import org.ml4ai.grounding.GroundingApp
import org.ml4ai.grounding.common.utils.GroundingAppApp

object HelloWorldApp extends GroundingAppApp {
  val appMessage = args.lift(0).getOrElse(getArgString("apps.HelloWorldApp.message", Some("App message not found!")))
  logger.info(appMessage)

  val groundingApp = GroundingApp()
  val classMessage = groundingApp.getArgString("GroundingApp.message", Some("Class message not found!"))
  logger.info(classMessage)
}
