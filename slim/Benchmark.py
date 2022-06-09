from slim import common_cli_options, get_config
import random


def main():
    parser = common_cli_options("SLIM Benchmark tool")

    cfg, args, out_path = get_config(parser)


if __name__ == "__main__":
    main()
