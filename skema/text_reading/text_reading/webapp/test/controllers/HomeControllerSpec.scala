package controllers

import org.scalatestplus.play._
import org.scalatestplus.play.guice._
import play.api.mvc.{Action, AnyContent}
import play.api.test._
import play.api.test.Helpers._

/**
 * Add your spec here.
 * You can mock out a whole application including requests, plugins etc.
 *
 * For more information, see https://www.playframework.com/documentation/latest/ScalaTestingWithScalaTest
 */
class HomeControllerSpec extends PlaySpec with GuiceOneAppPerTest with Injecting {
  val expectedString = "SKEMA TextReading Odin Visualizer"

  "HomeController GET" should {

    "render the index page from a new instance of controller" in {
      val controller = new HomeController(stubControllerComponents())
      val home = controller.index().apply(FakeRequest(GET, "/"))

      status(home) mustBe OK
      contentType(home) mustBe Some("text/html")
      val actualString = contentAsString(home)
      actualString must include (expectedString)
    }

    "render the index page from the application" in {
      val controller = inject[HomeController]
      val home = controller.index().apply(FakeRequest(GET, "/"))

      status(home) mustBe OK
      contentType(home) mustBe Some("text/html")
      val actualString = contentAsString(home)
      actualString must include (expectedString)
    }

    "render the index page from the router" in {
      val request = FakeRequest(GET, "/")
      val home = route(app, request).get

      status(home) mustBe OK
      contentType(home) mustBe Some("text/html")
      val actualString = contentAsString(home)
      actualString must include (expectedString)
    }
  }


  def find(haystack: String, text: String): Int = {
    val needle = "\"" + text + "\":"

    @annotation.tailrec
    def loop(index: Int, count: Int): Int = {
      val newIndex = haystack.indexOf(needle, index)

      if (newIndex < 0) count
      else loop(newIndex + 1, count + 1)
    }

    loop(0, 0)
  }

  "HomeController POST groundStringsToMira" should {

    "render the groundingCandidates from the router" in {
      val request = FakeRequest(POST, "/groundStringsToMira?k=6")
          .withTextBody("COVID-19\ncell part")
      val controller = new HomeController(Helpers.stubControllerComponents())
      val action: Action[AnyContent] = controller.groundStringsToMira(k = 6)
      val futureResult = action.apply(request)
      val json = contentAsString(futureResult)

      find(json, "groundingConcept") mustBe (12)
      find(json, "score") mustBe (12)
      find(json, "id") mustBe (12)
    }
  }
}