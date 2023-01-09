"""Objects that hold timeseries objects defined over multiple locations."""
import datetime
###CTM from dataclasses import dataclass, field, fields, replace
###CTM_START
from dataclasses import fields, replace
###CTM_END
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .._typing import ArrayLike, PathLike
###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM from .adm_mapping import AdminLevel, AdminLevelMapping
###CTM_START
from ..numerical_libs import xp
from .adm_mapping import AdminLevelMapping
###CTM_END


# TODO now that this holds the adm_mapping, the adm_ids column can probably be replaced...
###CTM @dataclass(frozen=True)
class SpatialStratifiedTimeseries(object):
    """Class representing a generic timeseries that is defined over admin regions."""

    ###CTM_START
    # adm_level: int
    # adm_ids: ArrayLike
    # dates: ArrayLike  # TODO cupy cant handle this...
    # adm_mapping: AdminLevelMapping  # = field(init=False)
    ###CTM_END
    ###CTM_START
    def __init__(self, adm_level: int, adm_ids: ArrayLike, dates: ArrayLike, adm_mapping: AdminLevelMapping):
        self.adm_level = adm_level
        self.adm_ids = adm_ids
        self.dates = dates
        self.adm_mapping = adm_mapping
    ###CTM_END

    def __post_init__(self):
        """Perform some simple shape validation on after initing."""
        valid_shape = self.dates.shape + self.adm_ids.shape
        ###CTM_START
        # for f in fields(self):
        #     if "data_field" in f.metadata:
        #         field_shape = getattr(self, f.name).shape
        #         if field_shape != valid_shape:
        #             logger.error("Invalid timeseries shape {}; expected {}.", field_shape, valid_shape)
        #             raise ValueError
        ###CTM_END

    def __repr__(self) -> str:
        """Only print summary of the object in interactive sessions."""
        names = [f.name for f in fields(self) if f.name not in ["adm_ids", "dates"]]
        return (
            f"{names} for {self.adm_ids.shape[0]} adm{self.adm_level} regions from {self.start_date} to {self.end_date}"
        )

    ###CTM @property
    def start_date(self) -> datetime.date:
        """The first date of valid data."""
        return self.dates[0]

    ###CTM @property
    def end_date(self) -> datetime.date:
        """The last date of valid data."""
        return self.dates[-1]

    ###CTM @property
    def n_days(self) -> int:
        """Total number of days of data."""
        return len(self.dates)

    ###CTM @property
    def n_loc(self) -> int:
        """Total number of locations for which we have timeseries at the base admin level."""
        return len(self.adm_ids)

    ###CTM @staticmethod
    ###CTM @sync_numerical_libs
    def _generic_from_csv(
        filename: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate: Optional[datetime.date] = None,
        force_enddate_dow: Optional[int] = None,
        adm_col: str = "adm2",
        date_col: str = "date",
        column_names: Dict[str, str] = None,
    ):
        """Return a dict containing args to a subclass's constructor"""
        df = pd.read_csv(
            filename,
            index_col=[adm_col, date_col],
            engine="c",
            parse_dates=["date"],
        ).sort_index()

        dates = df.index.unique(level=date_col).values
        dates = dates.astype("datetime64[s]").astype(datetime.date)
        date_mask = _mask_date_range(dates, n_days, valid_date_range, force_enddate, force_enddate_dow)

        ret = {
            "dates": dates[date_mask],
            "adm_ids": xp.array(df.index.unique(level=adm_col).values),
        }

        for fcolumn, out_name in column_names.items():
            var_full_hist = xp.array(df[fcolumn].unstack().fillna(0.0).values).T
            ret[out_name] = var_full_hist[date_mask]

        return ret

    def to_dict(self, level=None):
        """Return the timeseries as a dict containing the data it's indices."""
        # get data to requested adm level
        obj = self.sum_adm_level(level) if level is not None else self

        ret = {f.name: getattr(obj, f.name) for f in fields(obj) if f.name not in ("adm_level", "adm_mapping")}

        # reshape index columns and get the right name for the adm id column
        ret_shp = ret["dates"].shape + ret["adm_ids"].shape
        ret["date"] = np.broadcast_to(ret.pop("dates")[..., None], ret_shp)
        adm_col_name = f"adm{obj.adm_level}"
        ret[adm_col_name] = np.broadcast_to(ret.pop("adm_ids")[None, ...], ret_shp)

        # Flatten arrays
        ret = {k: np.ravel(xp.to_cpu(v)) for k, v in ret.items()}

        return ret

    def to_dataframe(self, level=None):
        """Return the timeseries as a pandas.DataFrame."""
        data_dict = self.to_dict(level)
        df = pd.DataFrame(data_dict)
        adm_col = df.columns[df.columns.str.match("adm[0-9]")].item()
        return df.set_index([adm_col, "date"]).sort_index()

    def to_csv(self, filename, level=None):
        """Save the timeseries data to a csv."""
        # TODO log
        df = self.to_dataframe(level)
        df.to_csv(filename, index=True)

    def replace(self, **changes):
        """Replace data columns."""
        return replace(self, **changes)

    def sum_adm_level(self, level: Union[int, str]):
        """Sum the values to a different admin level then they are defined on."""
        if type(level) == str:

            # Check string begins with 'adm'
            if level[:3] == "adm":
                level = int(level.split("adm")[-1])
            else:
                logger.error("String admin aggregation level must begin with adm")
                raise ValueError

        # TODO masking, weighting?
        if level > self.adm_level:
            logger.error("Requested sum to finer adm level than the data.")
            raise ValueError

        if level == self.adm_level:
            return self

        # out_id_map = self.adm_mapping.levels[level].idx
        out_id_map = self.adm_mapping.levels[level].idx[self.adm_mapping.levels[self.adm_level].level_idx]
        new_ids = self.adm_mapping.levels[level].ids

        new_data = {"adm_level": level, "adm_ids": new_ids, "dates": self.dates, "adm_mapping": self.adm_mapping}
        for f in fields(self):
            if "summable" in f.metadata:
                orig_ts = getattr(self, f.name)
                new_ts = xp.zeros(new_ids.shape + self.dates.shape, dtype=orig_ts.dtype)
                xp.scatter_add(new_ts, out_id_map, orig_ts.T)
                new_data[f.name] = new_ts.T

        return self.__class__(**new_data)

    def validate_isfinite(self):
        """Check that there are no invalid values in the timeseries (nan, inf, etc)."""
        for f in fields(self):
            if "data_field" in f.metadata:
                col_name = f.name
                data = getattr(self, f.name)
                finite = xp.isfinite(data)
                if xp.any(~finite):
                    locs = xp.argwhere(~finite)
                    logger.error(
                        f"Nonfinite values found in column {col_name} of {self.__class__.__qualname__} at {locs}!",
                    )
                    raise RuntimeError


def _mask_date_range(
    dates: np.ndarray,
    n_days: Optional[int] = None,
    valid_date_range: Tuple[Optional[datetime.date], Optional[datetime.date]] = (None, None),
    force_enddate: Optional[datetime.date] = None,
    force_enddate_dow: Optional[int] = None,
):
    """Calculate the bool mask for dates to include from the historical data."""
    valid_date_range = list(valid_date_range)

    if valid_date_range[0] is None:
        valid_date_range[0] = dates[0]
    if valid_date_range[1] is None:
        valid_date_range[1] = dates[-1]

    # Set the end of the date range to the last valid date that is the requested day of the week
    if force_enddate_dow is not None:
        end_date_dow = valid_date_range[1].weekday()
        days_after_forced_dow = (end_date_dow - force_enddate_dow + 7) % 7
        valid_date_range[1] = dates[-(days_after_forced_dow + 1)]

    if force_enddate is not None:
        if force_enddate_dow is not None:
            if force_enddate.weekday() != force_enddate_dow:
                logger.error("Start date not consistant with required day of week")
                raise RuntimeError
        valid_date_range[1] = force_enddate

    # only grab the requested amount of history
    if n_days is not None:
        valid_date_range[0] = valid_date_range[-1] - datetime.timedelta(days=n_days - 1)

    # Mask out dates not in request range
    date_mask = (dates >= valid_date_range[0]) & (dates <= valid_date_range[1])
    return date_mask


###CTM @dataclass(frozen=True, repr=False)
class CSSEData(SpatialStratifiedTimeseries):
    """Dataclass that holds the CSSE case/death data."""

    ###CTM_START
    # cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    # cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    # incident_cases: ArrayLike = field(default=None, metadata={"data_field": True, "summable": True})
    # incident_deaths: ArrayLike = field(default=None, metadata={"data_field": True, "summable": True})
    ###CTM_END
    ###CTM_START
    def __init__(self,
                 adm_level: int, adm_ids: ArrayLike, dates: ArrayLike, adm_mapping: AdminLevelMapping,
                 cumulative_cases: ArrayLike, cumulative_deaths: ArrayLike, incident_cases: ArrayLike, incident_deaths: ArrayLike):
        super().__init__(adm_level, adm_ids, dates, adm_mapping)
        self.cumulative_cases = cumulative_cases
        self.cumulative_deaths = cumulative_deaths
        self.incident_cases = incident_cases
        self.incident_deaths = incident_deaths
    ###CTM_END

    def __post_init__(self):
        """Fill in incidence data if only inited with cumulatives."""
        if self.incident_cases is None:
            object.__setattr__(self, "incident_cases", xp.gradient(self.cumulative_cases, axis=0, edge_order=2))
        if self.incident_deaths is None:
            object.__setattr__(self, "incident_deaths", xp.gradient(self.cumulative_deaths, axis=0, edge_order=2))

        super().__post_init__()

    ###CTM @staticmethod
    def from_csv(
        file: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate: Optional[datetime.date] = None,
        force_enddate_dow: Optional[int] = None,
        adm_mapping: Optional[AdminLevelMapping] = None,
    ):
        """Read CSSE data from a CSV."""
        logger.info("Reading historical CSSE data from {}", file)
        adm_level = "adm2"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate,
            force_enddate_dow,
            adm_level,
            column_names={"cumulative_reported_cases": "cumulative_cases", "cumulative_deaths": "cumulative_deaths"},
        )

        var_dict["adm_mapping"] = adm_mapping
        return CSSEData(2, **var_dict)


###CTM @dataclass(frozen=True, repr=False)
class HHSData(SpatialStratifiedTimeseries):
    """Dataclass that holds the HHS hospitalization data."""

    ###CTM_START
    # current_hospitalizations: ArrayLike = field(metadata={"data_field": True, "summable": True})
    # incident_hospitalizations: ArrayLike = field(metadata={"data_field": True, "summable": True})
    ###CTM_END
    ###CTM_START
    def __init__(self,
                 adm_level: int, adm_ids: ArrayLike, dates: ArrayLike, adm_mapping: AdminLevelMapping,
                 current_hospitalizations: ArrayLike, incident_hospitalizations: ArrayLike):
        super().__init__(adm_level, adm_ids, dates, adm_mapping)
        self.current_hospitalizations = current_hospitalizations
        self.incident_hospitalizations = incident_hospitalizations
    ###CTM_END

    # TODO we probably need to store a AdminLevelMapping in each timeseries b/c the hhs adm_ids dont line up with the csse ones after we aggregate them to adm1...
    ###CTM @staticmethod
    def from_csv(
        file: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate: Optional[datetime.date] = None,
        force_enddate_dow: Optional[int] = None,
        adm_mapping: Optional[AdminLevelMapping] = None,
    ):
        """Read HHS data from a CSV."""
        logger.info("Reading historical HHS hospitalization data from {}", file)
        adm_level = "adm1"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate,
            force_enddate_dow,
            adm_col=adm_level,
            column_names={
                "incident_hospitalizations": "incident_hospitalizations",
                "current_hospitalizations": "current_hospitalizations",
            },
        )

        var_dict["adm_mapping"] = adm_mapping
        return HHSData(1, **var_dict)


# TODO redundant w/ BuckyFittedCaseData?!
###CTM @dataclass(frozen=True, repr=False)
class BuckyFittedData(SpatialStratifiedTimeseries):
    """Dataclass for fitted case/death time series data."""

    ###CTM_START
    # cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    # cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    # incident_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    # incident_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    ###CTM_END
    ###CTM_START
    def __init__(self,
                 adm_level: int, adm_ids: ArrayLike, dates: ArrayLike, adm_mapping: AdminLevelMapping,
                 cumulative_cases: ArrayLike, cumulative_deaths: ArrayLike, incident_cases: ArrayLike, incident_deaths: ArrayLike):
        super().__init__(adm_level, adm_ids, dates, adm_mapping)
        self.cumulative_cases = cumulative_cases
        self.cumulative_deaths = cumulative_deaths
        self.incident_cases = incident_cases
        self.incident_deaths = incident_deaths
    ###CTM_END


###CTM_START
# @dataclass(frozen=True, repr=False)
# class BuckyFittedCaseData(SpatialStratifiedTimeseries):
#     """Dataclass that holds the fitted CSSE case/death data, including incidence rates."""
#
#     cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
#     cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
#     incident_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
#     incident_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
#
#     @staticmethod
#     def from_csv(
#         file: PathLike,
#         n_days: Optional[int] = None,
#         valid_date_range=(None, None),
#         force_enddate: Optional[datetime.date] = None,
#         force_enddate_dow: Optional[int] = None,
#         adm_mapping: Optional[AdminLevelMapping] = None,
#     ):
#         """Read fitted CSSE data from CSV."""
#         logger.info("Reading fitted CSSE data from {}", file)
#         adm_level = "adm2"
#         var_dict = SpatialStratifiedTimeseries._generic_from_csv(
#             file,
#             n_days,
#             valid_date_range,
#             force_enddate,
#             force_enddate_dow,
#             adm_level,
#             column_names={
#                 "cumulative_cases": "cumulative_cases",
#                 "cumulative_deaths": "cumulative_deaths",
#                 "incident_cases": "incident_cases",
#                 "incident_deaths": "incident_deaths",
#             },
#         )
#
#         var_dict["adm_mapping"] = adm_mapping
#         return BuckyFittedCaseData(2, **var_dict)
###CTM_END
