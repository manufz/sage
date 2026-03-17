import random
from pathlib import Path

IMAGE_ROOT = Path(__file__).parent / 'images'

SCENARIO_MAP = {
    'normal':     'healthy',
    'drought':    'drought',
    'overwater':  'overwater',
    'heatstress': 'healthy',
    'disease':    'disease',
}

def get_current_frame(scenario='normal'):
    folder = IMAGE_ROOT / SCENARIO_MAP.get(scenario, 'healthy')
    images = list(folder.glob('*.jpg')) + list(folder.glob('*.JPG'))
    if not images:
        return None
    return str(random.choice(images))