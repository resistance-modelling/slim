import json
import datetime as dt
import pandas as pd


def to_dt(string_date):
    """Convert from string date to datetime date

    :param string_date: Date as string timestamp
    :type string_date: str
    :return: Date as Datetime
    :rtype: [type]
    """
    dt_format = '%Y-%m-%d %H:%M:%S'
    return dt.datetime.strptime(string_date, dt_format)


class Config:
    """Simulation configuration and parameters"""

    def __init__(self, environment_file, param_file, logger):
        """Read the configuration from files

        :param environment_file: Path to the environment JSON file
        :type environment_file: string
        :param param_file: path to the simulator parameters JSON file
        :param logger: Logger to be used
        :type logger: logging.Logger
        """

        # set logger
        self.logger = logger

        # read and set the params
        with open(environment_file) as f:
            data = json.load(f)

        self.params = RuntimeConfig(param_file)

        # time and dates
        self.start_date = to_dt(data["start_date"]["value"])
        self.end_date = to_dt(data["end_date"]["value"])
        self.tau = data["tau"]["value"]

        # general parameters
        self.ext_pressure = data["ext_pressure"]["value"]
        self.lice_pop_modifier = data["lice_pop_modifier"]["value"]

        # farms
        self.farms = [FarmConfig(farm_data["value"], self.logger)
                      for farm_data in data["farms"]]
        self.nfarms = len(self.farms)

        # reservoir
        reservoir_data = data["reservoir"]["value"]
        # starting sealice population in reservoir = number of cages * modifier
        # TODO: confirm this is intended behaviour
        # TODO: is external pressure and modifier the same?
        total_cages = sum([farm.n_cages for farm in self.farms])
        self.reservoir_num_lice = self.lice_pop_modifier * total_cages
        self.reservoir_num_fish = reservoir_data["num_fish"]["value"]
        self.reservoir_enfish_res = reservoir_data["enfish_res"]["value"]
        self.reservoir_ewt_res = reservoir_data["ewt_res"]["value"]

    def __getattr__(self, name):
        # obscure marshalling trick.
        params = self.__getattribute__("params")
        if name in dir(params):
            return params.__getattribute__(name)
        return self.__getattribute__(name)


class RuntimeConfig:
    """Simulation parameters and constants"""

    def __init__(self, hyperparam_file):
        with open(hyperparam_file) as f:
            data = json.load(f)

        self.f_meanEMB = data["f_meanEMB"]["value"]
        self.f_sigEMB = data["f_sigEMB"]["value"]
        self.env_meanEMB = data["env_meanEMB"]["value"]
        self.env_sigEMB = data["env_sigEMB"]["value"]
        self.EMBmort = data["EMBmort"]["value"]
        self.delay_EMB = data["delay_EMB"]["value"]
        self.delta_EMB = data["delta_EMB"]["value"]


class FarmConfig:
    """Config for individual farm"""

    def __init__(self, data, logger):
        """Create farm configuration

        :param data: Dictionary with farm data
        :type data: dict
        :param logger: Logger to be used
        :type logger: logging.Logger
        """

        # set logger
        self.logger = logger

        # set params
        self.num_fish = data["num_fish"]["value"]
        self.n_cages = data["ncages"]["value"]
        self.farm_location = data["location"]["value"]
        self.farm_start = to_dt(data["start_date"]["value"])
        self.cages_start = [to_dt(date)
                            for date in data["cages_start_dates"]["value"]]

        # generate treatment dates from ranges
        self.treatment_dates = []
        for range_data in data["treatment_dates"]["value"]:
            from_date = to_dt(range_data["from"])
            to_date = to_dt(range_data["to"])
            self.treatment_dates.extend(
                pd.date_range(from_date, to_date).to_pydatetime().tolist())
