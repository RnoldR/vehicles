def create_logger(filename: str):
    import logging
    from logging.config import dictConfig

    # Code initialisatie: logging
    # create logger
    LOGGING = { 
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': { 
            'standard': { 
                'format': '%(asctime)s [%(levelname)s] %(module)s: %(message)s'
            },
            'brief': {
                'format': '%(message)s'
            },
        },
        'handlers': { 
            'console': { 
                'level': 'INFO',
                'formatter': 'brief',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # Default is stderr
            },
            'file': { 
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': filename, 
                'mode': 'w',
            },
        },
        'loggers': {
            '': {
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            },
        },    
    }

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.config.dictConfig(LOGGING)

    return logger

### create_logger ###