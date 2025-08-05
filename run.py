import tyro

def parse_metric_from_txt(
        txt_path: str = 'data/pl_htcode/txts/waymo/metric_002_aug_shift_comp_baseline.txt'
    ) -> None:
    """Parse metric from txt."""
    from lib.utils.scripts.parse_utils import parse_func
    parse_func(txt_path)

def commit(message: str, all: bool = False) -> None:
    """Make a commit."""
    print(f"{message=} {all=}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "parse_metric_from_txt": parse_metric_from_txt,
            "commit": commit,
        }
    )

