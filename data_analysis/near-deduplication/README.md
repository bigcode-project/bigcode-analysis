# Near deduplication

Code for data near deduplication.

### Setup

````
pip install -r requirements.txt
````

Login to be able to be able to push the dataset to the hub after deduplication and clone your huggingface-hub repositories:

````
huggingface-cli login
````

And make sure you have git-lfs installed.

If you use datasets with different column names from the BigCode ones, you might need to change `PATH_COLUMN` and `CONTENT` variables in `minhash_deduplication.py`.

### Usage

To run near deduplication use the following command and adapt the arguments for your case:

````
python near_deduplicate.py \
    --dataset_name bigcode-data/python_any_license_v2 \
    --org bigcode-data \
    --repo_name python_any_license_v2_near_dedup \
    --out_path ./data/any_license-near-dedup \
    --text_column content 
````

To make just a test run with a subset of the data set `test_run` argument to True.

The first time you load the dataset might be slow if it is large, but the data is saved in the cache thanks to `datasets`, and the subsequent calls will be fast.

### Alternative Deduplication Script

`minhash_deduplication_alt.py` is an alternative you might find useful to use as well. It uses `dataset.map` for both hashing and querying as well as `networkx` for duplicate community detection.

I will add an arg parser later, but for now you can update the config in `minhash_deduplication_alt.py` to use the script on custom dataset.

```bash
pip install -r requirements_alt.txt
python minhash_deduplication_alt.py
```

#### Benchmark

Running this script for `codeparrot/codeparrot-clean` on a 10-core machine:

```plaintext
Original size: 5361373
Removed size: 2077902
Processing time taken: 3567.90 seconds
```

```plaintext
                                                                                   Some examples of duplicate code                                                                                 
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────   
  id             dup id         code                                                                               dup code                                                                  
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  0              112512         ###############################################################################    ###############################################################################   
                                ## ##  Copyright (C) 2013-2014 Tavendo GmbH ## ##  Licensed under the Apache       ## ##  Copyright (C) 2011-2014 Tavendo GmbH ## ##  Licensed under the Apache  
                                License, Version 2.0 (the "License"); ##  you may not use this file except in      License, Version 2.0 (the "License"); ##  you may not use this file except in   
                                compl                                                                              compl                                                             
  0              888354         ###############################################################################    ###############################################################################   
                                ## ##  Copyright (C) 2013-2014 Tavendo GmbH ## ##  Licensed under the Apache       ## ##  Copyright (C) 2013-2014 Tavendo GmbH ## ##  Licensed under the Apache  
                                License, Version 2.0 (the "License"); ##  you may not use this file except in      License, Version 2.0 (the "License"); ##  you may not use this file except in   
                                compl                                                                              compl                                                             
  0              1183366        ###############################################################################    ###############################################################################   
                                ## ##  Copyright (C) 2013-2014 Tavendo GmbH ## ##  Licensed under the Apache       ## ##  Copyright (C) 2013-2014 Tavendo GmbH ## ##  Licensed under the Apache  
                                License, Version 2.0 (the "License"); ##  you may not use this file except in      License, Version 2.0 (the "License"); ##  you may not use this file except in   
                                compl                                                                              compl                                                             
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  1              135491         from itertools import chain  from django.utils.itercompat import is_iterable       from itertools import chain  from django.utils.itercompat import is_iterable  
                                class Tags:     """     Built-in tags for internal checks.     """     admin =     class Tags:     """     Built-in tags for internal checks.     """     admin =  
                                'admin'     caches = 'caches'     compatibility = 'compatibility'     database =   'admin'     async_support = 'async_support'     caches = 'caches'   
                                '                                                                                  compatibilit                                                      
  1              3582944        from itertools import chain  from django.utils.itercompat import is_iterable       from itertools import chain  from django.utils.inspect import     
                                class Tags:     """     Built-in tags for internal checks.     """     admin =     func_accepts_kwargs from django.utils.itercompat import is_iterable   class   
                                'admin'     caches = 'caches'     compatibility = 'compatibility'     database =   Tags:     """     Built-in tags for internal checks.     """     admin = 'admin'  
                                '                                                                                  async_support = '                                                 
  1              622611         from itertools import chain  from django.utils.itercompat import is_iterable       from itertools import chain  from django.utils.itercompat import is_iterable  
                                class Tags:     """     Built-in tags for internal checks.     """     admin =     class Tags:     """     Built-in tags for internal checks.     """     admin =  
                                'admin'     caches = 'caches'     compatibility = 'compatibility'     database =   'admin'     caches = 'caches'     compatibility = 'compatibility'     database =  
                                '                                                                                  '                                                                 
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  2              1521387        """ The :mod:`sklearn.utils` module includes various utilites. """  from           """ The :mod:`sklearn.utils` module includes various utilites. """  import numpy  
                                collections import Sequence  import numpy as np from scipy.sparse import           as np from scipy.sparse import issparse import warnings  from .validation import  
                                issparse import warnings  from .murmurhash import murmurhash3_32 from              * from .murmurhash import murmurhash3_32  # Make sure that DeprecationWarning   
                                .validation import (as_f                                                                                                                             
  2              1434119        """ The :mod:`sklearn.utils` module includes various utilites. """  from           """ The :mod:`sklearn.utils` module includes various utilities. """ import sys  
                                collections import Sequence  import numpy as np from scipy.sparse import           from collections import Sequence  import numpy as np from scipy.sparse import   
                                issparse import warnings  from .murmurhash import murmurhash3_32 from              issparse import warnings  from .murmurhash import murmurhash3_32 from   
                                .validation import (as_f                                                           .validation i                                                     
  2              3017861        """ The :mod:`sklearn.utils` module includes various utilites. """  from           """ The :mod:`sklearn.utils` module includes various utilites. """  import numpy  
                                collections import Sequence  import numpy as np from scipy.sparse import           as np import warnings  from .validation import * from .murmurhash import  
                                issparse import warnings  from .murmurhash import murmurhash3_32 from              murmurhash3_32  # Make sure that DeprecationWarning get printed   
                                .validation import (as_f                                                           warnings.simplefilter(                                            
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  3              5155669        """ Python Character Mapping Codec cp1250 generated from                           """ Python Character Mapping Codec mazovia generated from 'pl-    
                                'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1250.TXT' with gencodec.py.  """#"  import      mappings/Mazovia.txt' with gencodec.py.  """#"  import codecs  ### Codec APIs   
                                codecs  ### Codec APIs  class Codec(codecs.Codec):      def                        class Codec(codecs.Codec):      def encode(self,input,errors='strict'):   
                                encode(self,input,errors='strict'):         r                                      return codecs.charm                                               
  3              31795          """ Python Character Mapping Codec cp1250 generated from                           """ Python Character Mapping Codec cp1257 generated from          
                                'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1250.TXT' with gencodec.py.  """#"  import      'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1257.TXT' with gencodec.py.  """#"  import   
                                codecs  ### Codec APIs  class Codec(codecs.Codec):      def                        codecs  ### Codec APIs  class Codec(codecs.Codec):      def       
                                encode(self,input,errors='strict'):         r                                      encode(self,input,errors='strict'):         r                     
  3              2354506        """ Python Character Mapping Codec cp1250 generated from                           """ Python Character Mapping Codec cp1250 generated from          
                                'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1250.TXT' with gencodec.py.  """#"  import      'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1250.TXT' with gencodec.py.  """  # "  import  
                                codecs  ### Codec APIs  class Codec(codecs.Codec):      def                        codecs  # ## Codec APIs  class Codec(codecs.Codec):     def encode(self, input,   
                                encode(self,input,errors='strict'):         r                                      errors='strict'):                                                 
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  4              259104         #!/usr/bin/python # encoding: utf-8 -*-  # Copyright: (c) 2013, Matthias           #!/usr/bin/python # encoding: utf-8 -*-  # (c) 2013, Matthias Vogelgesang   
                                Vogelgesang <matthias.vogelgesang@gmail.com> # GNU General Public License v3.0+    <matthias.vogelgesang@gmail.com> # GNU General Public License v3.0+ (see COPYING  
                                (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)  from __future__         or https://www.gnu.org/licenses/gpl-3.0.txt)  from __future__ import  
                                import absol                                                                       absolute_import,                                                  
  4              1758834        #!/usr/bin/python # encoding: utf-8 -*-  # Copyright: (c) 2013, Matthias           #!/usr/bin/python # encoding: utf-8 -*-  # (c) 2013, Matthias Vogelgesang   
                                Vogelgesang <matthias.vogelgesang@gmail.com> # GNU General Public License v3.0+    <matthias.vogelgesang@gmail.com> # GNU General Public License v3.0+ (see COPYING  
                                (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)  from __future__         or https://www.gnu.org/licenses/gpl-3.0.txt)  from __future__ import  
                                import absol                                                                       absolute_import,                                                  
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  5              2660437        # This file was created automatically by SWIG 1.3.29. # Don't modify this file,    # This file was created automatically by SWIG 1.3.29. # Don't modify this file,   
                                modify the SWIG interface instead.  import _controls_ import new                   modify the SWIG interface instead.  import _controls_ import new  
                                new_instancemethod = new.instancemethod def                                        new_instancemethod = new.instancemethod def                       
                                _swig_setattr_nondynamic(self,class_type,name,value                                _swig_setattr_nondynamic(self,class_type,name,value               
  5              1201266        # This file was created automatically by SWIG 1.3.29. # Don't modify this file,    # This file was created automatically by SWIG 1.3.29. # Don't modify this file,   
                                modify the SWIG interface instead.  import _controls_ import new                   modify the SWIG interface instead.  import _controls_ import new  
                                new_instancemethod = new.instancemethod def                                        new_instancemethod = new.instancemethod def                       
                                _swig_setattr_nondynamic(self,class_type,name,value                                _swig_setattr_nondynamic(self,class_type,name,value               
  5              4537606        # This file was created automatically by SWIG 1.3.29. # Don't modify this file,    # This file was created automatically by SWIG 1.3.29.  # Don't modify this file,  
                                modify the SWIG interface instead.  import _controls_ import new                   modify the SWIG interface instead.    import _controls_  import new   
                                new_instancemethod = new.instancemethod def                                        new_instancemethod = new.instancemethod  def                      
                                _swig_setattr_nondynamic(self,class_type,name,value                                _swig_setattr_nondynamic(self,class_type,name                     
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  6              1609381        # -*- coding: utf-8 -*- """     werkzeug.contrib.cache                             # -*- coding: utf-8 -*- """ cachecore ~~~~~~~~~  Simple cache backends, inspired  
                                ~~~~~~~~~~~~~~~~~~~~~~      The main problem with dynamic Web sites is, well,      by werkzeug.contrib.cache.  The main problem with dynamic Web sites is, well,   
                                they're dynamic.  Each     time a user requests a page, the webserver executes a   they're dynamic.  Each time a user requests a page, the webserver executes a lot  
                                lot of code, queries                                                                                                                                 
  6              1631604        # -*- coding: utf-8 -*- """     werkzeug.contrib.cache                             # -*- coding: utf-8 -*-  """      werkzeug.contrib.cache          
                                ~~~~~~~~~~~~~~~~~~~~~~      The main problem with dynamic Web sites is, well,      ~~~~~~~~~~~~~~~~~~~~~~        The main problem with dynamic Web sites is, well,   
                                they're dynamic.  Each     time a user requests a page, the webserver executes a   they're dynamic.  Each      time a user requests a page, the webserver executes   
                                lot of code, queries                                                               a lot of code, que                                                
  6              2363588        # -*- coding: utf-8 -*- """     werkzeug.contrib.cache                             # -*- coding: utf-8 -*- """     werkzeug.contrib.cache            
                                ~~~~~~~~~~~~~~~~~~~~~~      The main problem with dynamic Web sites is, well,      ~~~~~~~~~~~~~~~~~~~~~~      The main problem with dynamic Web sites is, well,   
                                they're dynamic.  Each     time a user requests a page, the webserver executes a   they're dynamic.  Each     time a user requests a page, the webserver executes a  
                                lot of code, queries                                                               lot of code, queries                                              
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  8              42538          import os from importlib import import_module  from django.core.exceptions         import os from importlib import import_module  from django.core.exceptions  
                                import ImproperlyConfigured from django.utils.module_loading import                import ImproperlyConfigured from django.utils._os import upath from   
                                module_has_submodule  MODELS_MODULE_NAME = 'models'   class AppConfig:             django.utils.module_loading import module_has_submodule  MODELS_MODULE_NAME =   
                                """Class representing                                                              'models'   class Ap                                               
  8              1296507        import os from importlib import import_module  from django.core.exceptions         import os from importlib import import_module  from django.core.exceptions  
                                import ImproperlyConfigured from django.utils.module_loading import                import ImproperlyConfigured from django.utils.module_loading import   
                                module_has_submodule  MODELS_MODULE_NAME = 'models'   class AppConfig:             module_has_submodule  MODELS_MODULE_NAME = 'models'   class AppConfig:     """  
                                """Class representing                                                              Class represen                                                    
  8              1348817        import os from importlib import import_module  from django.core.exceptions         import os from importlib import import_module  from django.core.exceptions  
                                import ImproperlyConfigured from django.utils.module_loading import                import ImproperlyConfigured from django.utils.module_loading import   
                                module_has_submodule  MODELS_MODULE_NAME = 'models'   class AppConfig:             module_has_submodule  MODELS_MODULE_NAME = 'models'   class AppConfig:     """  
                                """Class representing                                                              Class represen                                                    
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  9              392578         # -*- coding: utf-8 -*- # # Copyright (c) 2017 F5 Networks Inc. # GNU General      # -*- coding: utf-8 -*- # # Copyright (c) 2017 F5 Networks Inc. # GNU General   
                                Public License v3.0 (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)      Public License v3.0 (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)   
                                from __future__ import (absolute_import, division, print_function) __metaclass__   from __future__ import (absolute_import, division, print_function) __metaclass__  
                                =                                                                                  =                                                                 
  9              1301553        # -*- coding: utf-8 -*- # # Copyright (c) 2017 F5 Networks Inc. # GNU General      # -*- coding: utf-8 -*- # # Copyright: (c) 2017, F5 Networks Inc. # GNU General   
                                Public License v3.0 (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)      Public License v3.0 (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)   
                                from __future__ import (absolute_import, division, print_function) __metaclass__   from __future__ import (absolute_import, division, print_function) __metaclass__  
                                =                                                                                                                                                    
  9              876489         # -*- coding: utf-8 -*- # # Copyright (c) 2017 F5 Networks Inc. # GNU General      # -*- coding: utf-8 -*- # # Copyright (c) 2017 F5 Networks Inc. # GNU General   
                                Public License v3.0 (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)      Public License v3.0 (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)   
                                from __future__ import (absolute_import, division, print_function) __metaclass__   from __future__ import (absolute_import, division, print_function) __metaclass__  
                                =                                                                                  =                                                                 
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  10             3843925        import unittest, os, errno from ctypes import * from ctypes.util import            import unittest, os, errno, sys from ctypes import * from ctypes.util import  
                                find_library from test import test_support try:     import threading except        find_library from test import test_support try:     import threading except   
                                ImportError:     threading = None  class Test(unittest.TestCase):     def          ImportError:     threading = None  class Test(unittest.TestCase):   
                                test_open(self):                                                                   @unittest.skipIf(                                                 
  10             1622435        import unittest, os, errno from ctypes import * from ctypes.util import            import unittest, os, errno import threading  from ctypes import * from  
                                find_library from test import test_support try:     import threading except        ctypes.util import find_library  class Test(unittest.TestCase):     def   
                                ImportError:     threading = None  class Test(unittest.TestCase):     def          test_open(self):         libc_name = find_library("c")         if libc_name is  
                                test_open(self):                                                                   None:                                                             
  10             1312888        import unittest, os, errno from ctypes import * from ctypes.util import            import unittest, os, errno  from ctypes import *  from ctypes.util import   
                                find_library from test import test_support try:     import threading except        find_library  try:      import threading  except ImportError:      threading =  
                                ImportError:     threading = None  class Test(unittest.TestCase):     def          None    class Test(unittest.TestCase):      def test_open(self):  
                                test_open(self):                                                                   libc_name = f                                                     
                                                                                                                                                                                     
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
```

Compared to the other script on a 8-core machine on `lvwerra/codeparrot-clean`:

```plaintext
Execution time ~3h: Execution time: 2:30:00 for make_duplicate_clusters, 1:00:00 for multipro_find_extremes

Orginal dataset size: 5361373
Duplicate cluster: 757938
Files in duplicate cluster: 2677039
Unique files in duplicate cluster: 940857
Filtered dataset size: 3625191
```
