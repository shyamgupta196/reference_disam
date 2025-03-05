import logging
import json
from pathlib import Path


logger = logging.getLogger('app')
#logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(Path(__file__).with_name('app.log'))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

with Path(__file__).with_name('config.json').open('r') as fp:
    config = json.load(fp)