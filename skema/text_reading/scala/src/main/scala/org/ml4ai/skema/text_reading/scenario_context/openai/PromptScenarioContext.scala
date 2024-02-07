package org.ml4ai.skema.text_reading.scenario_context.openai


import akka.actor.{ActorSystem, Terminated}
import akka.stream.Materializer
import io.cequence.openaiscala.domain._
import io.cequence.openaiscala.domain.response.ChatCompletionResponse
import io.cequence.openaiscala.domain.settings.CreateChatCompletionSettings
import io.cequence.openaiscala.service.{OpenAIService, OpenAIServiceFactory}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.Success
import ch.qos.logback.classic.{Level, Logger}
import org.slf4j.LoggerFactory


object TestPromptScenarioContext extends App {
  val prompter = new PromptScenarioContext()

  val left = "in14,18 for Finland, Germany, Italy, Luxembourg, The Netherlands, the United Kingdom, and the Tomsk Oblast of Russia. For European countries, we relied on data and the setting-specific contact matrices developed in Fumanelli et al.23 that covers 26 countries. Unfortunately, Poland, and Belgium, which are included in the POLYMOD study14 used to calibrate the overall contact matrix are not included in ref. 23. We performed a multiple linear regression analysis to find the k such that the resulting Mij best fits the empirical data. values of ω Note that the empirical matrices derived in refs. 14,18 describe the average number of contacts of age j for an individual of age i, and in \\\"Methods\\\" we show how ωk is related to an average number of contacts 〈c〉 per individual. The regression yields 4.11 contacts (standard error, SE 0.41) in the household setting, 11.41 contacts (SE 0.27) in schools, 8.07 contacts (SE 0.52) in workplaces, and 2.79 contacts (SE 0.48) for the general community setting. It is worth remarking that for household contacts is larger than the average household size. This likely reflects the definition of contacts at home (rather than with household members) used in the POLYMOD study14 that has been used to calibrate the weights. The rationale for using the POLYMOD and the Russian studies14,18 in estimating the weights used to assemble the setting-specific synthetic matrices lies in the extensive validation of those contact patterns in epidemiological studies of a set of airborne infectious diseases, including influenza29,35–39. the estimated weight Our approach provides overall best matching ω k and that, in principle, some of the differences in the social behavior of specific countries may not be captured by this approach. For this reason, as a validation of this calibration method, in Fig. 3a we report the correlation between the resulting synthetic matrices for France, Japan, and the Shanghai Province of China and the available empirical matrices for these additional locations for the period of 2022-2023 30–32. We find significant ("
  val focus = "P value < 0.001"
  val right = ") Pearson correlations of 0.92, 0.9, and Japan, and Shanghai Province, respectively. 0.8 for France, 4 NATURE COMMUNICATIONS |   (2021) 12:323  | https://doi.org/10.1038/s41467-020-20544-y | www.nature.com/naturecommunications        "
  prompter.promptForLocationContext(left, focus, right) foreach println
  prompter.promptForTemporalContext(left, focus, right) foreach println
  prompter.closeAll()
}
class PromptScenarioContext {

  LoggerFactory
    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
    .asInstanceOf[Logger]
    .setLevel(Level.INFO)

  implicit val system: ActorSystem = ActorSystem()
  implicit val materializer: Materializer = Materializer(system)
  implicit val ec: ExecutionContext = ExecutionContext.Implicits.global
  val service: OpenAIService = OpenAIServiceFactory()

  def closeAll(): Future[Terminated] = {
    service.close()
    system.terminate()
  }

  def promptForTemporalContext(left:String, focus:String, right:String):List[String] = {
    val messages = Seq(
      SystemMessage("You are a scientist that reads research articles looking for the relevant temporal that is context to a specific model variable or variable and its value."),
      UserMessage(s"Consider the target model variable between triple quotes within the passage. Create a bulleted list with the most specific relevant dates, months, years and time spans associated to the specific model variable or variable with value. Be as specific, but also make sure that you pay attention to the temporal data of the studies from which the values and variables are derived. Don't include relative dates. If there is no relevant temporal context, say NO CONTEXT. After listing the geographic location contexts, write an explanation of why you arrived to the conclusion and why you didn't choose other times mentioned on the passage.\n\nHere is a relevant example to guide you:\n\n'''\nPassage:  NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-020-20544-y ARTICLE Fig. 6 Epidemic impact. a Scatter plot of the attack rate and the ```reproduction number R0```  from an age-structured SIR model using the contact matrix for each subnational location for the period between march through may of 2021. European countries are included. The black line shows the results of the classic homogeneous mixing SIR model (no age groups). b Scatter plot of attack rates and the average age in each location. The black line represents the best-fitting linear model demonstrating a negative linear correlation between attack rates and the average age of the population. c Scatter plot of attack rates and percentage of the population attending educational institutions in each location for the data of one month later. The black line represents the best-fitting linear model. from what are different is observed in all other locations (including their respective countries). A more detailed discussion is reported in Supplementary Information, including data for 2022. If we consider the US state of New York as a reference and compute the distance from all other locations to it, a geographical pattern clearly emerges (Fig. 5b). Indeed, the contact patterns in most states of the US, and the urbanized areas of Canada and Australia appear to be very closely related to the one inferred for New York. In contrast, most of India, South Africa, and of the territories in Canada, Russia, and Australia have contact patterns noticeably different from those obtained for the state of New York. into after coming Epidemiological relevance. To investigate the effect of the computed contact matrices on infection transmission dynamics, we develop an age-structured SIR model to describe influenza transmission dynamics in the sites considered. The SIR model describes the spread of influenza in terms of the transition of individuals between different epidemiological compartments. Susceptible individuals (i.e., those at risk of acquiring the infection—S) can become infectious (i.e., capable to transmit the infection—I) contact with infectious individuals. Subsequently, infectious individuals recover from the infection and become removed (R) after a certain amount of time (the infectious period). In an age-structured implementa\n\nTemporal Context: \n - March through May of 2021\n\nExplanation: March through May of 2021 is the only relevant time span in the same sentence where reproduction number R0 is mentioned. The rest of the temporal locations, such as 2022, and one month later, correspond to other independent statements. Additionally one month later is a relative date, which I am instructed not to consider.\n'''\n\nFor this example, the relevant temporal context is ```March through may of 2021``` but it is not any other date, month, year or time span according to the text.\n\nNow do it for the next example:\n\n'''\nPassage:  $left```$focus```$right\n\nTemporal Context:\n'''")
    )


    val future = service
      .createChatCompletion(
        messages = messages,
        settings = CreateChatCompletionSettings(
          model = ModelId.gpt_4_turbo_preview,
          temperature = Some(0),
          max_tokens = Some(200)
        )
      )

    Await.ready(future, Duration.Inf)

    future.value match {
      case Some(s:Success[ChatCompletionResponse]) =>
        val msg = s.value.choices.head.message.content
        val resp = msg.split("\n").withFilter(_.startsWith("- ")).map(_.drop(2)).toList
        resp
      case _ =>
        List.empty[String]
    }
  }

  def promptForLocationContext(left:String, focus:String, right: String):List[String] = {
    val messages = Seq(
      SystemMessage("You are a scientist that reads research articles looking for the relevant geographical location that is context to a specific model variable or variable and its value."),
      UserMessage(s"Consider the the target model variable between triple quotes within the passage. Create a bulleted list with the most specific relevant locations on which the model variable holds. Be as specific as possible. If there is no relevant geographical context, say NO CONTEXT. After listing the geographic location contexts, write an explanation of why you arrived to the conclusion and why you didn't choose other geographic locations mentioned on the passage.\n\nHere is a relevant example to guide you:\n\n'''\nPassage:  NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-020-20544-y ARTICLE Fig. 6 Epidemic impact. a Scatter plot of the attack rate and the ```reproduction number R0```  from an age-structured SIR model using the contact matrix for each subnational location. European countries are included. The black line shows the results of the classic homogeneous mixing SIR model (no age groups). b Scatter plot of attack rates and the average age in each location. The black line represents the best-fitting linear model demonstrating a negative linear correlation between attack rates and the average age of the population. c Scatter plot of attack rates and percentage of the population attending educational institutions in each location. The black line represents the best-fitting linear model. from what are different is observed in all other locations (including their respective countries). A more detailed discussion is reported in Supplementary Information. If we consider the US state of New York as a reference and compute the distance from all other locations to it, a geographical pattern clearly emerges (Fig. 5b). Indeed, the contact patterns in most states of the US, and the urbanized areas of Canada and Australia appear to be very closely related to the one inferred for New York. In contrast, most of India, South Africa, and of the territories in Canada, Russia, and Australia have contact patterns noticeably different from those obtained for the state of New York. into after coming Epidemiological relevance. To investigate the effect of the computed contact matrices on infection transmission dynamics, we develop an age-structured SIR model to describe influenza transmission dynamics in the sites considered. The SIR model describes the spread of influenza in terms of the transition of individuals between different epidemiological compartments. Susceptible individuals (i.e., those at risk of acquiring the infection—S) can become infectious (i.e., capable to transmit the infection—I) contact with infectious individuals. Subsequently, infectious individuals recover from the infection and become removed (R) after a certain amount of time (the infectious period). In an age-structured implementa\n\nLocation Context: \n - European countries\n\nExplanation: European countries is the only relevant location in the same sentence where reproduction number R0 is mentioned. The rest of the geographic locations, such as Canada, India, etc., correspond to other independent statements\n'''\n\nFor this example, the relevant location context is ```European countries``` but it is not any other location according to the text.\n\nNow do it for the next example:\n\n'''\nPassage:  $left```$focus```$right\n\nLocation Context:\n'''")
    )


    val future = service
      .createChatCompletion(
        messages = messages,
        settings = CreateChatCompletionSettings(
          model = ModelId.gpt_4_turbo_preview,
          temperature = Some(0),
          max_tokens = Some(200)
        )
      )

    Await.ready(future, Duration.Inf)

    future.value match {
        case Some(s:Success[ChatCompletionResponse]) =>
          val msg = s.value.choices.head.message.content
          val resp = msg.split("\n").withFilter(_.startsWith("- ")).map(_.drop(2)).toList
          resp
        case _ =>
          List.empty[String]
      }
  }
}

