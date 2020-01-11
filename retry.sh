#!/bin/bash

RETRIES=3
DELAY=10
COUNT=1
while [ $COUNT -lt $RETRIES ]; do
  $*
  if [ $? -eq 0 ]; then
    RETRIES=0
    break
  fi
  let COUNT=$COUNT+1
  echo "retrying..."
  sleep $DELAY
done
