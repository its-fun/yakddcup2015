#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Extracting feature(s).

Every submodule has a function named `extract', they should satisfy the
signature described as below:


Args
----
enrollment: pandas DataFrame
Features of these enrollments will be generated. The columns should be:
enrollment_id, username, course_id.

base_date: datetime object
Generated features are to predict user dropout bahaviour in the next 10 days
after `base_date'.

full_dataset: dict
{'enroll': <pandas DataFrame>, 'log': <pandas DataFrame>,
 'obj': <pandas DataFrame>}


Returns
-------
X: numpy ndarray
Extracted feature(s) of corresponding enrollments. Shape should be:
(len(enrollment), n_features)
"""


METHODS = []
