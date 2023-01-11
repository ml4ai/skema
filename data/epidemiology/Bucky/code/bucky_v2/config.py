"""Global configuration handler for Bucky, also include prior parameters"""

from glob import glob
from importlib import resources
from pathlib import Path, PosixPath

from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
from ruamel.yaml.scalarfloat import ScalarFloat

yaml = YAML()


class YamlPath(PosixPath):
    """Class to wrap path-like objects fromt he yaml files."""

    yaml_tag = "!path"

    ###CTM @classmethod
    def to_yaml(cls, representer, node):
        """to_yaml."""
        # print(f"{''.join(node.parts)}")
        return representer.represent_scalar(cls.yaml_tag, f"{'/'.join(node.parts)}")

    ###CTM @classmethod
    def from_yaml(cls, constructor, node):
        """from_yaml."""
        return cls(node.value)


yaml.register_class(YamlPath)
from loguru import logger

###CTM from .numerical_libs import sync_numerical_libs, xp
###CTM_START
from .numerical_libs import xp
###CTM_END

from .util import distributions
from .util.age_interp import age_bin_interp
from .util.nested_dict import NestedDict


def locate_base_config():
    """Locate the base_config package that shipped with bucky (it's likely in site-packages)."""
    ###CTM_START
    # with resources.path("bucky", "__init__.py") as pkg_init_path:
    #     return pkg_init_path.parent / "base_config"
    ###CTM_END
    ###CTM_START
    pkg_init_path = resources.path("bucky", "__init__.py")
    return pkg_init_path.parent / "base_config"
    ###CTM_END


def locate_current_config():
    """Find the config file/directory to use."""
    potential_locations = [
        Path.cwd(),
        Path.home(),
    ]

    for p in potential_locations:
        cfg_dir = p / "bucky.conf.d"
        if cfg_dir.exists() and cfg_dir.is_dir():
            logger.info("Using bucky config directory at {}", str(cfg_dir))
            return cfg_dir

        cfg_one_file = p / "bucky.yml"
        if cfg_one_file.exists():
            logger.info("Using bucky config file at {}", str(cfg_one_file))
            return cfg_one_file

    base_cfg = locate_base_config()
    logger.warning("Local Bucky config not found, using defaults at {}", str(base_cfg))
    return base_cfg


class BuckyConfig(NestedDict):
    """Bucky configuration."""

    def load_cfg(self, par_path):
        """Load the bucky config from disk."""
        base_cfg = locate_base_config()

        self._load_one_cfg(base_cfg)
        if par_path != base_cfg:
            self._load_one_cfg(par_path)

        self._cast_floats()
        return self

    def _load_one_cfg(self, par_path):
        """Read in the YAML cfg file(s)."""
        logger.info("Loading bucky config from {}", par_path)
        par = Path(par_path)

        ###CTM_START
        # try:
        #     if par.is_dir():
        #         for f_str in sorted(glob(str(par / "**"), recursive=True)):
        #             f = Path(f_str)
        #             if f.is_file():
        #                 if f.suffix not in {".yml", ".yaml"}:
        #                     logger.warning("Ignoring non YAML file {}", f)
        #                     continue
        #
        #                 logger.debug("Loading config file {}", f)
        #                 self.update(yaml.load(f.read_text(encoding="utf-8")))  # nosec
        #     else:
        #         self.update(yaml.load(par.read_text(encoding="utf-8")))  # nosec
        # except FileNotFoundError:
        #     logger.exception("Config not found!")
        ###CTM_END
        ###CTM_START
        if par.is_dir():
            for f_str in sorted(glob(str(par / "**"), recursive=True)):
                f = Path(f_str)
                if f.is_file():
                    if f.suffix not in {".yml", ".yaml"}:
                        logger.warning("Ignoring non YAML file {}", f)
                        continue

                    logger.debug("Loading config file {}", f)
                    self.update(yaml.load(f.read_text(encoding="utf-8")))  # nosec
        else:
            self.update(yaml.load(par.read_text(encoding="utf-8")))  # nosec
        ###CTM_END

        return self

    ###CTM @sync_numerical_libs
    def _to_arrays(self, copy=False):
        """Cast all terminal sub-lists into xp.arrays."""
        # wip
        def _cast_to_array(v):
            """Cast a single non-str iterable to an array."""
            return v if isinstance(v, str) else xp.array(v)

        ret = self.apply(_cast_to_array, copy=copy, apply_to_lists=True)
        return ret

    ###CTM @sync_numerical_libs
    def _to_lists(self, copy=False):
        """Cast all terminal sub-arrays into lists."""
        # wip
        def _cast_to_list(v):
            """Cast an xp.array to a list (an move to cpu mem)."""
            return xp.to_cpu(xp.squeeze(v)).tolist() if isinstance(v, xp.ndarray) else v

        ret = self.apply(_cast_to_list, copy=copy)
        return ret

    def _cast_floats(self, copy=False):
        """Cast all yaml float objects to python floats."""

        def _cast_float(v):
            """Cast a yaml float to a python float."""
            return float(v) if isinstance(v, ScalarFloat) else v

        ret = self.apply(_cast_float, copy=copy)
        return ret

    def to_yaml(self, *args, **kwargs):
        """Dump the object to yaml."""
        stream = StringIO()

        yaml.dump(self._to_lists(copy=True).to_dict(), stream, *args, **kwargs)
        return stream.getvalue()

    ###CTM @sync_numerical_libs
    def interp_age_bins(self):
        """Interpolate any age stratified params to the model specified age bins."""

        def _interp_values_one(d):
            """Interp one array to the model's age bins."""
            d["value"] = age_bin_interp(self["model.structure.age_bins"], d.pop("age_bins"), d["value"])
            return d

        def _interp_dists_one(d):
            """Interp one distribution to the model's age bins."""
            bins = d.pop("age_bins")
            if "loc" in d["distribution"]:
                d["distribution.loc"] = age_bin_interp(self["model.structure.age_bins"], bins, d["distribution.loc"])
            if "scale" in d["distribution"]:
                d["distribution.scale"] = age_bin_interp(
                    self["model.structure.age_bins"],
                    bins,
                    d["distribution.scale"],
                )
            return d

        self._to_arrays()
        ret = self.apply(_interp_values_one, contains_filter=["age_bins", "value"])
        ret = ret.apply(_interp_dists_one, contains_filter=["age_bins", "distribution"])
        return ret

    def promote_sampled_values(self):
        """Promote sampled distributions up in the hierarchy so they are more easily referenced."""

        def _promote_values(d):
            """Promote one value if it's size 1."""
            return d["value"] if len(d) == 1 else d

        ret = self.apply(_promote_values, contains_filter="value")
        return ret

    ###CTM @sync_numerical_libs
    def _set_default_variances(self, copy=False):
        """Set gaussian variance to the default for params that don't do so explictly."""

        def _set_reroll_var(d):
            """Set variance for one param."""
            if d["distribution.func"] == "truncnorm" and "scale" not in d["distribution"]:
                d["distribution.scale"] = xp.abs(
                    xp.array(self["model.monte_carlo.default_gaussian_variance"]) * xp.array(d["distribution.loc"]),
                )
            return d

        ret = self.apply(_set_reroll_var, copy=copy, contains_filter="distribution")
        return ret

    # TODO move to own class like distributionalConfig?
    ###CTM @sync_numerical_libs
    def sample_distributions(self):
        """Draw a sample from each distributional parameter and drop it inline (in a returned copy of self)"""

        # TODO add something like 'register_distribtions' so we dont have to iterate the tree to find them?
        def _sample_distribution(d):
            """Draw a sample from one distribution."""
            dist = d.pop("distribution")._to_arrays()
            func = dist.pop("func")

            if hasattr(distributions, func):
                base_func = getattr(distributions, func)
            elif hasattr(xp.random, func):  # noqa: SIM106
                base_func = getattr(xp.random, func)
            else:
                raise ValueError(f"Distribution {func} does not exist!")

            d["value"] = base_func(**dist)
            return d

        # self._to_arrays()
        ret = self._set_default_variances(copy=True)
        ret = ret.interp_age_bins()
        ret = ret.apply(_sample_distribution, contains_filter="distribution")
        ret = ret.interp_age_bins()
        ret = ret.promote_sampled_values()
        ret = ret._cast_floats()
        return ret


base_cfg = BuckyConfig()
cfg = BuckyConfig()

"""
def load_base_cfg(path):
    base_cfg.load_cfg(path)


def roll_cfg_distributions():
    cfg = base_cfg.sample_distributions()
"""

if __name__ == "__main__":
    file = "par2/"
    cfg = BuckyConfig().load_cfg(file)
    # print(cfg)

    samp = cfg.sample_distributions()
    # print(samp)
