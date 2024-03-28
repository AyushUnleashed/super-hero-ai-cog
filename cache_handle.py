import os
os.environ['HF_HOME'] = '/src/cache'

import transformers
transformers.utils.move_cache()