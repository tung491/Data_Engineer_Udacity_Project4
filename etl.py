import configparser
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType as Dbl, IntegerType as Int, \
StringType as Str, StructField as Fld, StructType as R

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    # read song data file
    songSchema = R(
            [
                Fld('num_songs', Int()),
                Fld('artist_id', Str()),
                Fld('artist_lattitude', Dbl()),
                Fld('artist_longitude', Dbl()),
                Fld('artist_location', Str()),
                Fld('artist_name', Str()),
                Fld('song_id', Str()),
                Fld('title', Str()),
                Fld('duration', Dbl()),
                Fld('year', Int())
            ]
    )
    df = spark.read.json(song_data, schema=songSchema)
    df.createOrReplaceTempView('songData')

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration').\
        dropDuplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs'))

    # extract columns to create artists table
    artists_table = df.select('artist_id', 'artist_name', 'artist_location', 'artist_lattitude', 'artist_longitude')
    artists_table.withColumnRenamed('artist_name', 'name')
    artists_table.withColumnRenamed('artist_location', 'location')
    artists_table.withColumnRenamed('artist_lattitude', 'lattitude')
    artists_table.withColumnRenamed('artist_longitude', 'longitude')

    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists'))


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log-data/*.json')
    logSchema = R(
            [
                Fld('aritst', Str()),
                Fld('auth', Str()),
                Fld('firstName', Str()),
                Fld('gender', Str()),
                Fld('itemInSession', Str()),
                Fld('lastName', Str()),
                Fld('length', Dbl()),
                Fld('level', Str()),
                Fld('location', Str()),
                Fld('method', Str()),
                Fld('page', Str()),
                Fld('registration', Dbl()),
                Fld('sessionId', Str()),
                Fld('song', Str()),
                Fld('status', Int()),
                Fld('ts', Int()),
                Fld('userAgent', Str()),
                Fld('userId', Int())
            ]
    )
    # read log data file
    df = spark.read.json(log_data, schema=logSchema)
    df.createTempView('logData')

    # filter by actions for song plays.parquet
    df = df.where('page' == 'NextSong')

    # extract columns for users table
    users_table = df.select('user_id', 'first_name', 'last_name', 'gender', 'level').\
        dropDuplicates(['user_id'])

    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: F.to_timestamp(x))
    df = df.withColumn('timestamp', get_timestamp('ts'))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: F.to_date(x))
    df = df.withColumn('date', get_datetime('ts'))

    # extract columns to create time table
    time_table = df.select('timestamp', 'hour(timestamp)',
                           'day(timestamp)', 'week(timestamp)',
                           'month(timestamp)', 'weekeday(timestamp)')

    time_table.withColumnRenamed('timestamp', 'start_time')
    time_table.withColumnRenamed('hour(timestamp)', 'hour')
    time_table.withColumnRenamed('day(timestamp)', 'day')
    time_table.withColumnRenamed('week(timestamp)', 'week')
    time_table.withColumnRenamed('month(timestamp)', 'month')
    time_table.withColumnRenamed('weekday(timestamp', 'weekday')
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet('users')

    # read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data, 'songs'))

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(song_df, df.song == song_df.title). \
        select('timestamp', 'user_id', 'level', 'song_id',
               'artist_id', 'session_id', 'location', 'user_agent')

    songplays_table.withColumnRenamed('timestamp', 'start_time')
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.paritionBy('year', 'month(start_time').parquet(os.path.join(output_data, 'songplays'))
    songplays_table.withColumn('songplay_id', F.monotonicallyIncreasingId())


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacityoutput/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
