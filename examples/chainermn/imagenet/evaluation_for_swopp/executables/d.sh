#!/bin/bash

DUMMY_FILE=/tmp/dummy
LOCK_FILE=/tmp/lock_file

touch $DUMMY_FILE

if ! ln -s $DUMMY_FILE $LOCK_FILE; then
    echo `hostname`": LOCKED"

    while [ -e $LOCK_FILE ]; do
      echo `hostname`': waiting...'
      sleep 1
    done
else
    echo `hostname`': got the lock'
    sleep 5
    rm -rf $LOCK_FILE
fi

