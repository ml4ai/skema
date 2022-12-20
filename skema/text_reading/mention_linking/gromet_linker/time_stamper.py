import datetime

class TimeStamper():
	def stamp(self) -> str:
		pass

class DebugTimeStamper(TimeStamper):
	def stamp(self) -> str:
		return "2022-01-01 00:00:01.0"

class NowTimeStamper(TimeStamper):
	def stamp(self) -> str:
		return str(datetime.datetime.now())

time_stamper = DebugTimeStamper()
