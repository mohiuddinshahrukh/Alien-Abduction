import logging
import os
import random
from copy import deepcopy

import numpy as np
from clemcore.clemgame import GameInstanceGenerator

try:
    from functions import FUNCTION_REGISTRY
    from protocol import (
        ACTIVE_IO_MODE,
        ACTIVE_MEMBERSHIP_MODE,
        PASSIVE_IO_MODE,
        PASSIVE_IO_ONESHOT_MODE,
        PASSIVE_MEMBERSHIP_MODE,
        PASSIVE_MEMBERSHIP_ONESHOT_MODE,
    )
    from utils import (
        create_passive_io_examples,
        create_passive_membership_examples,
        create_static_test_cases,
    )
except ModuleNotFoundError:
    from function_detective.functions import FUNCTION_REGISTRY
    from function_detective.protocol import (
        ACTIVE_IO_MODE,
        ACTIVE_MEMBERSHIP_MODE,
        PASSIVE_IO_MODE,
        PASSIVE_IO_ONESHOT_MODE,
        PASSIVE_MEMBERSHIP_MODE,
        PASSIVE_MEMBERSHIP_ONESHOT_MODE,
    )
    from function_detective.utils import (
        create_passive_io_examples,
        create_passive_membership_examples,
        create_static_test_cases,
    )

logger = logging.getLogger(__name__)

DOMAIN_CATEGORIES = [
    "NUMBERS",
    "TWO_NUMBERS",
    "STRING",
    "LIST",
    "LOGIC",
]

MAX_TURNS = 15
NUM_TESTS = 100
INTERACTIVE_PASSIVE_EXAMPLES = NUM_TESTS
ONESHOT_EXAMPLES = 10

MODE_CONFIGS = [
    {"mode": ACTIVE_IO_MODE, "name_suffix": "query_mode", "max_turns": MAX_TURNS, "baseline_kind": None},
    {"mode": PASSIVE_IO_MODE, "name_suffix": "example_mode", "max_turns": MAX_TURNS, "baseline_kind": None},
    {
        "mode": ACTIVE_MEMBERSHIP_MODE,
        "name_suffix": "pair_in_set_mode",
        "max_turns": MAX_TURNS,
        "baseline_kind": None,
    },
    {
        "mode": PASSIVE_MEMBERSHIP_MODE,
        "name_suffix": "labeled_pairs_mode",
        "max_turns": MAX_TURNS,
        "baseline_kind": None,
    },
    {
        "mode": PASSIVE_IO_ONESHOT_MODE,
        "name_suffix": "example_mode_oneshot",
        "max_turns": 1,
        "baseline_kind": "oneshot",
    },
    {
        "mode": PASSIVE_MEMBERSHIP_ONESHOT_MODE,
        "name_suffix": "labeled_pairs_mode_oneshot",
        "max_turns": 1,
        "baseline_kind": "oneshot",
    },
]

MODE_INSTANCE_FILES = {
    ACTIVE_IO_MODE: "instance_query_mode",
    PASSIVE_IO_MODE: "instance_example_mode",
    ACTIVE_MEMBERSHIP_MODE: "instance_pair_in_set_mode",
    PASSIVE_MEMBERSHIP_MODE: "instance_labeled_pairs_mode",
    PASSIVE_IO_ONESHOT_MODE: "instance_example_mode_oneshot",
    PASSIVE_MEMBERSHIP_ONESHOT_MODE: "instance_labeled_pairs_mode_oneshot",
}


class FunctionDetectiveInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed=None, **kwargs):
        self.filename = "instances.json"
        self.instances = {"experiments": []}

        def create_experiment(name: str, mode_config: dict):
            exp = self.add_experiment(name)
            exp["mode"] = mode_config["mode"]
            exp["max_turns"] = mode_config["max_turns"]
            exp["baseline_kind"] = mode_config["baseline_kind"]
            exp["guesser_initial_prompt"] = self.load_template("resources/initial_prompts/initial_guesser")
            return exp

        experiments = {}
        for domain in DOMAIN_CATEGORIES:
            for mode_config in MODE_CONFIGS:
                exp_name = f"{domain.lower()}_{mode_config['name_suffix']}"
                experiments[(domain, mode_config["mode"])] = create_experiment(exp_name, mode_config)

        for i, function_data in enumerate(FUNCTION_REGISTRY):
            domain = function_data["category"]
            if domain not in DOMAIN_CATEGORIES:
                logger.warning("Unknown domain=%s for %s", domain, function_data["function_name"])
                continue

            base_seed = seed if seed is not None else 0
            case_seed = (base_seed * 10_000) + i
            random.seed(case_seed)
            np.random.seed(case_seed)

            static_tests = create_static_test_cases(
                function_data["callable"],
                function_data["category"],
                signature=function_data["signature"],
                difficulty=function_data.get("difficulty"),
                num_tests=NUM_TESTS,
            )

            for mode_index, mode_config in enumerate(MODE_CONFIGS):
                target_experiment = experiments[(domain, mode_config["mode"])]
                instance_id = (i * len(MODE_CONFIGS)) + mode_index
                game_instance = self.add_game_instance(target_experiment, instance_id)
                game_instance["callable"] = function_data["function_name"]
                game_instance["signature"] = function_data["signature"]
                game_instance["category"] = function_data["category"]
                game_instance["difficulty"] = function_data.get("difficulty")
                game_instance["mode"] = mode_config["mode"]
                game_instance["baseline_kind"] = mode_config["baseline_kind"]
                game_instance["test_cases"] = static_tests

                if mode_config["mode"] in {PASSIVE_IO_MODE, PASSIVE_IO_ONESHOT_MODE}:
                    count = ONESHOT_EXAMPLES if mode_config["baseline_kind"] == "oneshot" else INTERACTIVE_PASSIVE_EXAMPLES
                    game_instance["passive_examples"] = create_passive_io_examples(static_tests, num_examples=count)
                elif mode_config["mode"] in {PASSIVE_MEMBERSHIP_MODE, PASSIVE_MEMBERSHIP_ONESHOT_MODE}:
                    count = ONESHOT_EXAMPLES if mode_config["baseline_kind"] == "oneshot" else INTERACTIVE_PASSIVE_EXAMPLES
                    game_instance["passive_examples"] = create_passive_membership_examples(
                        static_tests=static_tests,
                        return_type=function_data["signature"].split("->")[-1].strip() if "->" in function_data["signature"] else "Any",
                        category=function_data["category"],
                        num_examples=count,
                    )
                else:
                    game_instance["passive_examples"] = []

                logger.info(
                    "Generated instance for %s mode=%s with %s eval tests and %s passive examples",
                    function_data["function_name"],
                    mode_config["mode"],
                    len(static_tests),
                    len(game_instance["passive_examples"]),
                )

        self._store_mode_specific_instances()

    def _store_mode_specific_instances(self):
        for mode_config in MODE_CONFIGS:
            mode = mode_config["mode"]
            experiments = [
                deepcopy(experiment)
                for experiment in self.instances["experiments"]
                if experiment.get("mode") == mode
            ]
            payload = {"experiments": experiments}
            filename = f"{MODE_INSTANCE_FILES[mode]}.json"
            self.store_file(payload, filename, sub_dir="in")
            logger.info(
                "Stored %s with %s experiments and %s instances",
                filename,
                len(experiments),
                sum(len(experiment["game_instances"]) for experiment in experiments),
            )


if __name__ == "__main__":
    FunctionDetectiveInstanceGenerator().generate()
