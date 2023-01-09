"""Monte carlo output handler."""
from pathlib import Path

import fastparquet
import numpy as np
import pandas as pd
from loguru import logger

###CTM from ..numerical_libs import sync_numerical_libs, xp
###CTM_START
from ..numerical_libs import xp
###CTM_END
from ..util.async_thread import AsyncQueueThread


###CTM @sync_numerical_libs
def init_write_thread(**kwargs):  # pylint: disable=unused-argument
    """Init write thread w/ a nonblocking stream."""
    stream = xp.cuda.Stream(non_blocking=True) if xp.is_cupy else None
    pinned_mem = {}
    return {"stream": stream, "pinned_mem": pinned_mem}


###CTM @sync_numerical_libs
def write_parquet_dataset(df_data, data_dir, stream, pinned_mem):
    """Write a dataframe of MC output to parquet."""
    for k, v in df_data.items():
        if k not in pinned_mem:
            pinned_mem[k] = xp.empty_like_pinned(v)

        xp.to_cpu(v, stream=stream, out=pinned_mem[k])

    if stream is not None:
        stream.synchronize()

    df = pd.DataFrame({k: np.array(v) for k, v in pinned_mem.items()})

    for date, date_group in df.groupby("date"):
        date_partition = data_dir / f"date={date}"
        date_partition.mkdir(exist_ok=True)
        rid = df["rid"].iloc[0]
        fname = str(date_partition / f"rid_{rid}.parquet")
        fastparquet.write(fname, date_group.drop(columns=["date"]), stats=False, write_index=False)


def merge_parquet_dataset(data_dir, **kwargs):
    """Merge parquet datasets w/ fastparquet."""
    logger.info("Merging parquet files")

    for partition in data_dir.glob("date=*"):
        fastparquet.writer.merge([str(f) for f in partition.glob("*.parquet")])


class BuckyOutputWriter:
    """Class to manage the writing of raw output files and all the threading that comes with it."""

    ###CTM @sync_numerical_libs
    def __init__(self, output_base_dir, run_id, data_format="parquet"):
        """Init the writer globals."""
        self.output_dir = Path(output_base_dir) / str(run_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        if data_format == "parquet":
            self.write_thread = AsyncQueueThread(
                write_parquet_dataset,
                pre_func=init_write_thread,
                post_func=merge_parquet_dataset,
                data_dir=data_dir,
            )

    ###CTM @sync_numerical_libs
    def write_metadata(self, adm_mapping, str_dates, fitted_timeseries=None):
        """Write metadata to output dir."""
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        # write out adm level mappings
        # TODO should just copy from data?
        adm_map_file = metadata_dir / "adm_mapping.csv"
        adm_mapping.to_csv(adm_map_file)

        # write out dates
        date_file = metadata_dir / "dates.csv"
        np.savetxt(date_file, str_dates, header="date", comments="", delimiter=",", fmt="%s")

        if fitted_timeseries is not None:
            for name, ts in fitted_timeseries.items():
                ts_file = metadata_dir / (name + ".csv")
                ts.to_csv(ts_file)

    def write_mc_data(self, data_dict):
        """Write the data from one MC to the output dir."""
        # flatten the shape
        for c in data_dict:
            data_dict[c] = data_dict[c].ravel()

        # push the data off to the write thread
        self.write_thread.put(data_dict)

    def write_params(self, seed, params):
        """TODO WIP Write MC parameters per iteration."""
        # TODO
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        params_base_dir = metadata_dir / "params"

        # could also flatten the dict and savez it?

        # TODO rewrite as a recursive func to handle deeper nesting
        for k, v in params.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    param_dir = params_base_dir / k / k2
                    param_dir.mkdir(parents=True, exist_ok=True)
                    f = param_dir / (str(seed) + ".npy")
                    # print(f)
                    xp.save(f, v2)
            else:
                param_dir = params_base_dir / k
                param_dir.mkdir(parents=True, exist_ok=True)
                f = param_dir / (str(seed) + ".npy")
                # print(f)
                xp.save(f, v)

        # from IPython import embed
        # embed()
        raise NotImplementedError

    def write_historical_data(self):
        """TODO Write historical data used."""
        # TODO
        raise NotImplementedError

    def write_par_files(self):
        """TODO Copy parameter specs."""
        # TODO
        raise NotImplementedError

    def close(self):
        """Cleanup and join write thread."""
        self.write_thread.close()
