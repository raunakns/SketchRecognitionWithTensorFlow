from pybuilder.core import task
from pybuilder.core import init

@task
def say_hello (logger):
    logger.info("Hello, PyBuilder")
