
A Deep Learning Multi-Class Time Series Approach for Activity Recognition from a Single Chest-Mounted Sensor



Activity recognition is the problem of predicting movement and detecting activity of a person from sensor data. These sensors are often accelerometers and gyroscopes, and they capture activity data when mounted as smartphones or wearables (smart watches). It has been an active area of research recently since wearables (particularly for fitness tracking) are gaining popularity. Activity recognition data is collected over time as a subject performs an activity and the sensor tracks movements over fixed intervals of time. However, the problem of activity recognition is approached either as single point classification or as time series classification. The advantage in using time series classification is that it allows to model upon the temporal coherence information that is made available as an activity is recorded over a time interval rather than as a single point observation. 

For this project, we will be using a rather simplistic dataset that contains accelerometer data from 15 subjects who perform a variety of tasks. Data is collected as acceleration in 3 dimensions (x, y and z axes). We will be expanding on the existing research on this dataset by applying deep learning multi class time series classification algorithms to identify the activity. Previously, the dataset has been used with single point ensemble classification algorithms. We will be exploring the advantage of taking into account the temporal information of this dataset. Existing single point classification algorithms will also be implemented to benchmark the results. 

Previously the approach of time series classification has been successful on a more complex dataset (UCI HAR dataset) with accelerometer and gyroscope sensors, the dataset is more complex in terms of the sensor information (having 560 features on localization information of subjects). The goal of this project is to understand whether the added complexity of a multi-layer deep learning time series classification algorithm offers significant benefits in terms of the efficiency and the predictability when only accelerometer data is available (having only 3 features on acceleration information of subjects). 


