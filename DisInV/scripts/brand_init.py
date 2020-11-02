from disindb.models import Brand
import json
from DisInV.settings import PROJECT_DIR

# Please do not modify this list

with open(f"{PROJECT_DIR}/DB_default.json", "r") as file:
    db_init_value = json.loads(file.read())

publishers = db_init_value['Publishers']

# Django scripts will execute "run" automatically
def run():

    for dp in publishers:
        t = Brand(brand_name=dp)
        t.save()