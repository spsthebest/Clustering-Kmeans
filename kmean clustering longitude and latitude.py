# K means clustering using longitude and latitude

import sys
import numpy as np
from pyspark import SparkContext
from math import sqrt
from numpy import array

# latitude and longitude are taken as input

def kmeans_pro(line, centroid_list):
    lat_long = np.array(line).astype('float')
    minimum = float("inf")
    cluster = 0
    for i, centroid in enumerate(centroid_list):
        distance = sqrt(sum((lat_long - centroid) ** 2))  # calculating the distance between lat_long distances and assigning cluster value based on minimum distance
        if distance < minimum:
            minimum = distance
            cluster = i + 1
    return (cluster, lat_long)

# data is currently written to CSV file.
def to_csv_line(data):
    return ','.join(str(d) for d in data)

# execution starts here
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: spark-submit Kmeanspro.py <inputdatafile>"
        exit(-1)

    sc = SparkContext(appName="K means Clustering")

    input_file = sc.textFile(sys.argv[1])  # input file is read

    # data preprocessing
    header = input_file.first()  # indicates the names of the column
    input_without_header = input_file.filter(lambda line: line != header)  # filtering the data without header using the filter

    rdd_raw_data = input_without_header.map(lambda line: line.split(','))  # splitting the given line at delimiter ',' on input without header

    rdd_filter_data = rdd_raw_data.filter(lambda x: x[7] is not u'')  # delete the row for which latitude is empty.

    no_of_rows = rdd_filter_data.count()  # counting number of rows in rdd

    k = 40  # found using elbow method using a separate program.
    no_of_iterations = 30
    fraction = ((k + 30) / (no_of_rows * 0.1)) * 0.1  # Extra rows

    sample = rdd_filter_data.sample(True, fraction, 2300)

    centroid_list = np.array(sample.map(lambda line: (line[7], line[6])).take(k)).astype('float')

    for _ in range(no_of_iterations):
        rdd_cluster = rdd_filter_data.map(lambda line: kmeans_pro([line[7], line[6]], centroid_list))  # latitudes and longitudes
        # Calculate new set of centroids by grouping the points belonging to same cluster
        # and taking their average
        rdd_aggregate = rdd_cluster.aggregateByKey((0, 0), lambda a, b: (a[0] + b, a[1] + 1), lambda a, b: (a[0] + b[0], a[1] + b[1]))  # Calculating new set of centroids by grouping the points
        rdd_average = rdd_aggregate.mapValues(lambda v: v[0] / v[1])
        centroid_list = np.array(rdd_average.values().collect()).astype('float')

    rdd_final = rdd_filter_data.map(lambda line: kmeans_pro([line[7], line[6]], centroid_list))  # Final classification

    rdd_final.collect()

    lines = rdd_final.map(to_csv_line)

    lines.saveAsTextFile('DelhiKmeans')  # output file saved here
    sc.stop()