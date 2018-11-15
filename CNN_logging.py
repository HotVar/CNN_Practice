import logging

def set_logging():
    Log = logging.getLogger('CNN_logger')
    formatter = logging.Formatter('[%(levelname)s | %(filename)s:%(lineno)s] - %(asctime)s > %(message)s')
    fileHandler = logging.FileHandler('logfile.log')
    streamHandler = logging.StreamHandler()
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    Log.addHandler(fileHandler)
    Log.addHandler(streamHandler)
    Log.setLevel(logging.DEBUG)

    return Log