from pettingzoo.test import api_test


class TestSimulatorEnv:
    def test_api(self, sim_env):
        api_test(sim_env)
