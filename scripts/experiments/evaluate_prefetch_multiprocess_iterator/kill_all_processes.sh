#!/bin/bash

for pid in `ps -e -o pid,cmd | grep "python evaluate_prefetch_multiprocess_iterator.py" | grep -v grep | awk '{ print $1 }'`; do kill -9 $pid; done

