import datetime

from pettingzoo.test import api_test
from slim.types.policies import NO_ACTION


class TestSimulatorEnv:
    def test_api(self, sim_env):
        # A suit of API tests
        api_test(sim_env)

    def test_step_increase_day(self, sim_env_unwrapped):
        sim_env_unwrapped.reset()
        cur_day = sim_env_unwrapped.cur_day
        sim_env_unwrapped.step(NO_ACTION)
        sim_env_unwrapped.step(NO_ACTION)
        new_day = sim_env_unwrapped.cur_day
        assert new_day == cur_day + datetime.timedelta(days=1)

        # 30-day test
        for day in range(30):
            sim_env_unwrapped.step(NO_ACTION)
            sim_env_unwrapped.step(NO_ACTION)

        new_day = sim_env_unwrapped.cur_day
        assert new_day == cur_day + datetime.timedelta(days=31)
