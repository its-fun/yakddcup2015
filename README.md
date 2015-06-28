# Yet Another KDD Cup 2015 Solution


Based on [https://github.com/Divergent914/kddcup2015](https://github.com/Divergent914/kddcup2015)


## TODO

1. 特征选择与normalize

2. 手工加入规则，比如关于课程最近更新时间、用户最近操作时间

3. 观察预测错了的instance

4. 尝试深度学习model

5. AdaBoost, RF, Bagging, Blending


## Features

### Event

+ 用户在该课程的操作数量，enrollment最后一周、倒数第二周、第一周，占该用户在所有课程的比例，占该课程所有用户的比例

+ 用户有行为的课程数量，用户行为的最后一周、倒数第二周、第一周、总体

+ 课程的选课人数

+ trending slope of the weekly number of events within the enrollment

+ number of events in the last week, the first week, and the week before the
last week of the enrollment

+ average, standard deviation, maximal, minimal weekly numbers of events
in the enrollment period

+ coefficients b and c in the polynomial model y = a + bx + cx**2, where x
is week number (all start from 0), and y is the weekly number of events

+ 7 counts of events in Monday to Sunday

+ 24 counts of events in hour 0-23

+ 7 counts of event types

+ 2 counts of source types


### Session

+ number of 3-hour defined sessions in the enrollment

+ average, standard deviation, maximal, minimal numbers of events in 3-hour
defined sessions in the enrollment

+ statistics of 3-hour defined sessions: mean, std, max, min of duration

+ number of 1-hour defined sessions in the enrollment

+ average, standard deviation, maximal, minimal numbers of events in 1-hour
defined sessions in the enrollment

+ statistics of 1-hour defined sessions: mean, std, max, min of duration

+ number of 12-hour defined sessions in the enrollment

+ average, standard deviation, maximal, minimal numbers of events in 12-hour
defined sessions in the enrollment

+ statistics of 12-hour defined sessions: mean, std, max, min of duration

+ number of 1-day defined sessions in the enrollment

+ average, standard deviation, maximal, minimal numbers of events in 1-day
defined sessions in the enrollment

+ statistics of 1-day defined sessions: mean, std, max, min of duration

+ number of 7-day defined sessions in the enrollment

+ average, standard deviation, maximal, minimal numbers of events in 7-day
defined sessions in the enrollment

+ statistics of 7-day defined sessions: mean, std, max, min of duration


### 时间相关

+ 课程材料首次发布、最近发布距今几天

+ 用户初次、上次操作此课程据今几天，持续几天，初次访问课程材料距离开课时间几天

+ 课程的所有用户操作课程持续时间的：平均值、标准差、最大值、最小值，以及与课程持续时间的比例

+ month (1-12) of the first, last event in the enrollment

+ 用户对课程材料的首次操作时间与课程材料发布时间的日期差的：平均值、标准差、最大值、最小值，enrollment最后一周、倒数第二周、第一周、总体


### Dropout：连续十天无操作

+ 课程最后十天无操作的人数，与选课人数的比例

+ 用户在该课程中Dropout的次数，与平均每个课程Dropout次数的比例；总Dropout持续时长与课程持续时间的比例

+ 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：平均值、方差、最大值、最小值；有Dropout行为的课程占用户所选课程总数的比例
