import logging
import logging.config
import yaml


def setup_logging():
    with open('log-config.yaml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

def setup_logging_explicit(log_config_yaml):
    with open(log_config_yaml, 'r') as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger