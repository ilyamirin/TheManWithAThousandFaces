from functools import lru_cache
import fasttext


RESOURCES_PATH = 'src/resources'


@lru_cache()
def fasttext_model():
    print('Loading fastText model...')
    ft_model = fasttext.load_model(f'{RESOURCES_PATH}/pretrained/dp-fasttext.bin')
    print('├── Complete')
    return ft_model
