from engine.templates.template_pool import ALL_TEMPLATES
from engine.templates.template_mining import MINED_TEMPLATES
from engine.templates.hand_crafted import TIP_ADAPTER_TEMPLATES

def get_template(text_augmentation):
    if text_augmentation == 'vanilla':
        return "a photo of a {}."
    else:
        raise ValueError('Unknown template: {}'.format(text_augmentation))