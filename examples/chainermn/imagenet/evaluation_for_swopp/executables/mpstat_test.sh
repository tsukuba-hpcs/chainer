#!/bin/bash

CURRENT_DATETIME=`date +%Y%m%d_%H%M%S`
DUMMY_FILE="/tmp/dummy.${CURRENT_DATETIME}.`hostname`"
LOCK_FILE="/tmp/lock_file.${CURRENT_DATETIME}.`hostname`"
MPSTAT_LOG_FILE="/tmp/${CURRENT_DATETIME}/mpstat_`hostname`"
MPSTAT_PROC=-1

mkdir -p /tmp/${CURRENT_DATETIME}
touch $DUMMY_FILE

if ! ln -s $DUMMY_FILE $LOCK_FILE; then
    echo `hostname`": LOCKED"

    while [ -e $LOCK_FILE ]; do
      echo `hostname`': waiting...'
      sleep 1
    done
    echo `hostname`': detected the lock has been released'  
else
    echo `hostname`": start mpstat -P ALL 1 > $MPSTAT_LOG_FILE &"
    mpstat -P ALL 1 > $MPSTAT_LOG_FILE &
    MPSTAT_PROC=$!
    echo `hostname`": finish mpstat -P ALL 1 > $MPSTAT_LOG_FILE &"
    rm -rf $LOCK_FILE
fi

sleep 1

if [ $MPSTAT_PROC -ne -1 ]; then
    echo `hostname`": start kill -SIGINT MPSTAT_PROC"
    kill -INT $MPSTAT_PROC
    echo `hostname`": finish kill -SIGINT MPSTAT_PROC"
fi

