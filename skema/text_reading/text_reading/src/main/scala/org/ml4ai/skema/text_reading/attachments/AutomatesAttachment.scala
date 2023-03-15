package org.ml4ai.skema.text_reading.attachments

import org.clulab.odin.{Attachment, Mention}
import org.ml4ai.skema.text_reading.grounding.GroundingCandidate
import org.ml4ai.skema.text_reading.quantities.Interval
import play.api.libs.json.{JsValue, Json}
import ujson.Value

abstract class AutomatesAttachment extends Attachment with Serializable {

    // Support for JSON serialization
  def toJson: JsValue

  def toUJson: ujson.Value

}

class GroundingAttachment(candidates:Seq[GroundingCandidate]) extends AutomatesAttachment {

  override def toJson: JsValue = Json.arr(candidates map {
    c => Json.obj(
      "name" -> c.concept.name,
      "id" -> c.concept.id,
      "score" -> c.score
    )
  })

  override def toUJson: Value = ujson.Arr(candidates map {
    c =>
      ujson.Obj(
        "name" -> c.concept.name,
        "id" -> c.concept.id,
        "score" -> c.score
      )
  })
}

class LocationContextAttachment(locations:Seq[Mention]) extends AutomatesAttachment {
  override def toJson: JsValue = Json.arr(locations map {
    l => Json.obj(
      "scenarioLocation" -> l.text
    )
  })

  override def toUJson: Value = ujson.Obj(
    "scenarioLocation" -> (locations map (_.text)).distinct
  )
}

class TimeContextAttachment(times:Seq[Mention]) extends AutomatesAttachment {
  override def toJson: JsValue = Json.arr(times map {
    l => Json.obj(
      "scenarioTime" -> l.text
    )
  })


  override def toUJson: Value = ujson.Obj(
    "scenarioTime" -> (times map (_.text)).distinct
  )
}

class MentionLocationAttachment(val filename: String, val  pageNum: Seq[Int], val blockIdx: Seq[Int], attType: String) extends AutomatesAttachment {

  override def toJson: JsValue =  Json.obj(
    "filename" -> filename,
    "pageNum" -> pageNum,
    "blockIdx" -> blockIdx,
    "attType" -> attType)

  // use 'asInstanceOf' + this method to retrieve the information from the attachment

  def toUJson: ujson.Value = ujson.Obj(
    "filename" -> filename,
    "pageNum" -> pageNum,
    "blockIdx" -> blockIdx,
    "attType" -> attType) //"MentionLocation"
}

class DiscontinuousCharOffsetAttachment(charOffsets: Seq[(Int, Int)], attType: String) extends AutomatesAttachment {

  override def toJson: JsValue = ???

  def toUJson: ujson.Value = ujson.Obj(
    "charOffsets" -> offsetsToUJson(charOffsets),
    "attType" -> attType) //"DiscontinuousCharOffset"

  def offsetsToUJson(charOffsets: Seq[(Int, Int)]): ujson.Value = {
    val json = charOffsets.map(seq => ujson.Arr(seq._1, seq._2))
    json
  }

}


class ParamSetAttachment(attachedTo: String, attType: String) extends AutomatesAttachment {

  override def toJson: JsValue = ???

  def toUJson: ujson.Value = {
    val toReturn = ujson.Obj()

    toReturn("attachedTo") = attachedTo
    toReturn("attType") = attType //"ParamSetAtt"
    toReturn
  }

}

class ParamSettingIntAttachment(inclusiveLower: Option[Boolean], inclusiveUpper: Option[Boolean], attachedTo: String, attType: String) extends AutomatesAttachment {

  override def toJson: JsValue = ???

  def toUJson: ujson.Value = {
    val toReturn = ujson.Obj()

    if (inclusiveLower.isDefined) {
      toReturn("inclusiveLower") = inclusiveLower.get
    } else {
      toReturn("inclusiveLower") = ujson.Null
    }

    if (inclusiveUpper.isDefined) {
      toReturn("inclusiveUpper") = inclusiveUpper.get
    } else {
      toReturn("inclusiveUpper") = ujson.Null
    }

    toReturn("attachedTo") = attachedTo
    toReturn("attType") = attType //"ParamSettingIntervalAtt"
    toReturn
  }

}

class UnitAttachment(attachedTo: String, attType: String) extends AutomatesAttachment {

  override def toJson: JsValue = ???

  def toUJson: ujson.Value = {
    val toReturn = ujson.Obj()

    toReturn("attachedTo") = attachedTo
    toReturn("attType") = attType //"UnitAtt"
    toReturn
  }

}

class ContextAttachment(attType: String, context: ujson.Value, foundBy: String) extends AutomatesAttachment {

  override def toJson: JsValue = ???

  def toUJson: ujson.Value = ujson.Obj (
      "contexts" -> contextsToJsonObj(context),
      "attType" -> attType,
      "foundBy" -> foundBy
    )

  def contextsToJsonObj(contexts: ujson.Value): ujson.Value = {
//    val contextsToJsonObj = contexts.map(seq => ujson.Arr(seq))
    val contextsToJsonObj = ujson.Value(contexts)
    contextsToJsonObj
  }
}

class FunctionAttachment(attType: String, trigger: String, foundBy: String) extends AutomatesAttachment {

  override def toJson: JsValue = ???

  def toUJson: ujson.Value = {
    val toReturn = ujson.Obj()
    toReturn("attType") = attType //"FunctionAtt"
    toReturn("trigger") = trigger
    toReturn("foundBy") = foundBy
    toReturn
  }

}