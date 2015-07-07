# Yet Another KDD Cup 2015 Solution


Based on [https://github.com/Divergent914/kddcup2015](https://github.com/Divergent914/kddcup2015)


## TODO

0. 加特征：每个计数特征都应该有对应的比例特征；其他特征

0. 加数据：最多可以到depth=4；注意需要先review一下代码中有没有潜在的问题

1. feature selection

2. 观察预测错了的instance

3. 尝试深度学习model，http://card.memect.com/?q=keras

3. 尝试depth=1来生成训练集

3. kNN

4. GBDT: min_samples_leaf, max_depth ...

5. RF: min_samples_leaf, max_depth ...

6. AdaBoost(with LR/SVC)

7. 把测试集拿来一起做normalize


## Features (262)

### Event (172)

+ 用户在该课程的操作数量（按event_source序对统计），enrollment最后一周、倒数第二周、第一周、总体，占该用户在所有课程的比例，占该课程所有用户的比例 (108)

+ 用户有行为的课程数量，用户行为的最后一周、倒数第二周、第一周、总体 (4)

+ 课程的选课人数 (1)

+ trending slope of the weekly number of events within the enrollment (1)

+ numbers of events in the last week, the first week, and the week before the
last week, and all weeks of the enrollment; ratio on all courses by user; ratio
on all users by course (12)

+ average, standard deviation, maximal, minimal weekly numbers of events
in the enrollment period (4)

+ coefficients b and c in the polynomial model y = a + bx + cx**2, where x
is week number (all start from 0), and y is the weekly number of events (2)

+ 7 counts of events in Monday to Sunday (7)

+ 24 counts of events in hour 0-23 (24)

+ 7 counts of event types (7)

+ 2 counts of source types (2)


### Session (45)

+ number of 3-hour defined sessions in the enrollment (1)

+ average, standard deviation, maximal, minimal numbers of events in 3-hour
defined sessions in the enrollment (4)

+ statistics of 3-hour defined sessions: mean, std, max, min of duration (4)

+ number of 1-hour defined sessions in the enrollment (1)

+ average, standard deviation, maximal, minimal numbers of events in 1-hour
defined sessions in the enrollment (4)

+ statistics of 1-hour defined sessions: mean, std, max, min of duration (4)

+ number of 12-hour defined sessions in the enrollment (1)

+ average, standard deviation, maximal, minimal numbers of events in 12-hour
defined sessions in the enrollment (4)

+ statistics of 12-hour defined sessions: mean, std, max, min of duration (4)

+ number of 1-day defined sessions in the enrollment (1)

+ average, standard deviation, maximal, minimal numbers of events in 1-day
defined sessions in the enrollment (4)

+ statistics of 1-day defined sessions: mean, std, max, min of duration (4)

+ number of 7-day defined sessions in the enrollment (1)

+ average, standard deviation, maximal, minimal numbers of events in 7-day
defined sessions in the enrollment (4)

+ statistics of 7-day defined sessions: mean, std, max, min of duration (4)


### 时间相关 (33)

+ 课程材料首次发布、最近发布距今几天 (2)

+ 用户初次、上次操作此课程据今几天，持续几天，与课程持续时间的比例，初次访问课程材料距离开课时间几天 (5)

+ 课程的所有用户操作课程持续时间的：平均值、标准差、最大值、最小值，以及与课程持续时间的比例 (8)

+ month (1-12) of the first, last event in the enrollment (2)

+ 用户对课程材料的首次操作时间与课程材料发布时间的日期差的：平均值、标准差、最大值、最小值，enrollment最后一周、倒数第二周、第一周、总体 (16)


### Dropout：连续十天无操作 (12)

+ 课程最后十天有操作的人数，与选课人数的比例: 课程有更新的最后十天、课程有操作的最后十天 (3)

+ 用户平均每个课程Dropout的次数，在该课程中Dropout的次数，与平均每个课程Dropout次数的比例；总Dropout持续时长与课程持续时间的比例 (4)

+ 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：平均值、方差、最大值、最小值；有Dropout行为的课程占用户所选课程总数的比例 (5)
