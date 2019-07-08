#!/bin/bash

exec {my_fd}< "$0"
flock -n ${my_fd}
if [  $? -eq 0 ]; then
    echo `hostname`': I got the lock!! YEAH!!!'
    sleep 5
else
    flock -n ${my_fd}
    while [  $? -ne 0 ]; do
        echo `hostname`': I cannot get the lock...wtf'
        sleep 1
        flock -n ${my_fd}
    done
    echo `hostname`': I finally got the lock!!!!'
fi

