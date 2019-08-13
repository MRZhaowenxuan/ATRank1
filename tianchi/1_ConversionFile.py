import csv
import json
csvfile = open('../shop_data/user_log_format1.csv', 'r')
jsonfile = open('../shop_data/user_log_format1.json', 'w')
fieldnames1 = ("user_id", "item_id", "cat_id", "seller_id", "brand_id", "time_stamp", "action_type")
reader = csv.DictReader(csvfile, fieldnames1)
i = 0
for row in reader:
    if i == 0:
        i += 1
        continue
    json.dump(row, jsonfile)
    jsonfile.write('\n')
    i += 1

jsonfile.close()
csvfile.close()

