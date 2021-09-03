#!/usr/bin/env bash
today=`date --date='1 days ago' +%Y-%m-%d`
# 每日更新
python extract_aspect_opinion.py --data ${today}
