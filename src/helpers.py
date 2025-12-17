import logging

def log_config(arguments):
    logging.info("Logging used config:")
    logging.info("-" * 50)
    for argument, value in arguments.items():
        logging.info("{}: {}".format(argument, value))
    logging.info("-" * 50)

def print_example(example):
    for key, value in example.items():
        logging.info(f"{key}: {value}")