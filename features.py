#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Extracting feature(s).

Every submodule has a function named `extract', they should satisfy the
signature described as below:


Args
----
base_date: datetime object
Generated features are to predict user dropout bahaviour in the next 10 days
after `base_date'.


Returns
-------
X: pandas DataFrame
Extracted feature(s) of corresponding enrollments. Shape should be:
(len(enrollment_all), n_features + 1)
The extra first column should be enrollment_id
"""


import events
import sessions
import time_related
import dropouts


METHODS = [events, sessions, time_related, dropouts]
