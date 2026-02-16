import logging

LOGGER = logging.getLogger("hieve_sim")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)
