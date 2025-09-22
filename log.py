import config as cfg

def print_verbose(message):
    if cfg.verbose:
        print(message)

def print_not_verbose(message, end='\n'):
    if not cfg.verbose:
        print(message, end=end)
