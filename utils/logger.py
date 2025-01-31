import sys
from pathlib import Path

from loguru import logger
from prettytable import PrettyTable


def setup_logger(level: str = "INFO", log_dir: str = None):
    logger.remove(0)
    logger.add(sys.stdout, level=level, colorize=True)

    if log_dir:
        log_path = Path(log_dir) / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level=level, colorize=False)


def pformat_table(data: list[dict]):
    if len(data) == 0:
        return "[Empty]"
    table = PrettyTable()
    table.field_names = data[0].keys()
    for row in data:
        table.add_row(row.values())
    return table.get_string()
