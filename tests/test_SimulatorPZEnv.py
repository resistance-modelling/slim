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

    def test_gym_space_farm(self, first_farm, first_cage, cur_day):
        space = first_farm.get_gym_space()
        assert not space["current_treatments"][0]
        assert space["allowed_treatments"] == 7

        first_farm.apply_action(cur_day, 0)
        first_farm.apply_action(cur_day + datetime.timedelta(days=1), 1)
        first_cage.get_lice_treatment_mortality(cur_day + datetime.timedelta(days=2))

        assert first_cage.is_treated()
        assert first_cage.current_treatments == [0, 1]
        space = first_farm.get_gym_space()
        assert space["current_treatments"][0]
        assert space["current_treatments"][1]
        assert space["allowed_treatments"] == 5

        first_farm.fallow()
        space = first_farm.get_gym_space()
        assert space["current_treatments"][-1]
