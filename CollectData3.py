#!/home/daniel_d_knudsen/.venv/bin/python
import pandas as pd
import urllib.parse
import requests
import pickle
import os
import re
import time
import openpyxl

color_to_rare = {
    "b0c3d9": "Consumer Grade",
    "5e98d9": "Industrial Grade",
    "4b69ff": "Mil-Spec",
    "8847ff": "Restricted",
    "d32ce6": "Classified",
    "eb4b4b": "Covert",
    "ffd700": "Special",
}


def parse_weapon(weap_json):
    name = weap_json["name"]
    stattrak = "StatTrak" in name
    name_match = re.match(r"([★™\w\- ]+) \| ([\w ]+) \(([\w\- ]+)\)", name)
    is_case = name.endswith(" Case")
    is_package = name.endswith(" Package")
    is_key = name.endswith(" Case Key")
    if name_match is None and not is_case and not is_key and not is_package:
        return None
    if not is_case and not is_key and not is_package:
        weapon = name_match.group(1).replace("★ ", "").replace("StatTrak™ ", "")
        skin = name_match.group(2)
        cond = name_match.group(3)
    elif is_key:
        skin = "n/a"
        cond = "n/a"
        weapon = "Key"
    elif is_package:
        skin = "n/a"
        cond = "n/a"
        weapon = "Package"
    elif is_case:
        skin = "n/a"
        cond = "n/a"
        weapon = "Case"
    if weapon in ["Sticker", "Sealed Graffiti"]:
        return None
    url = "https://steamcommunity.com/market/listings/730/" + urllib.parse.quote(name)
    data = {
        "name": name,
        "stattrak": stattrak,
        "weapon": weapon,
        "skin": skin,
        "condition": cond,
        "sell_price": weap_json["sell_price"] / 100,
        "sell_listings": weap_json["sell_listings"],
        "url": url,
        "is_key": is_key,
        "is_case": is_case,
        "is_package": is_package,
    }
    return data


def get_prices(resp):
    df_data = []
    for match in re.finditer(r'\["([\w ]+: \+0)",([\d\.]+),"(\d+)"\]', resp):
        date, price, vol = match.group(1), match.group(2), match.group(3)
        vol = int(vol)
        price = float(price)
        date = " ".join(date.split(" ")[:3])
        df_data.append([date, price, vol])
    df = pd.DataFrame(df_data, columns=["date", "price", "volume"])
    m = re.search(r'"type":"([^"]+?)","market_name"', resp)
    if m is None:
        return None
    mc = re.search(r'"value":"([\w\- ]+ Collection)","color":"9da1a9"', resp)
    if mc is None:
        collection = ""
    else:
        collection = mc.group(1)
    items = [
        (im.group(1), im.group(2), color_to_rare.get(im.group(3), im.group(3)))
        for im in re.finditer(
            r'"value":"([\w\- ]+) \| ([\w\- ]+)","color":"(\w+)"', resp
        )
    ]
    type_ = m.group(1).replace("StatTrak\\u2122", "").replace("\\u2605 ", "").strip()
    data = {"type": type_, "prices": df, "items": items, "collection": collection}
    return data
# %%
if os.path.exists("dataset.pkl"):
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
else:
    dataset = {}
# %%
dataset = {}
header_idx = 0
headers = [
    {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36",
    },
]

base_url = "https://steamcommunity.com/market/search/render/"

window_size = 100
i = 0
# while True:
for indexx in [1]:
    print("Cursor", i)
    try:
        page = requests.get(
            base_url
            + "?start={}&count={}&search_descriptions=0&appid=730&norender=1".format(
                i, window_size
            ),
            headers=headers[header_idx % len(headers)],
        )
        page_json = page.json()
        assert len(page_json["results"]) > 0
    except (TypeError, OSError, AssertionError) as e:
        print("Load error", e, page.text)
        header_idx += 1
        # for _ in range(20):
            # time.sleep(3)
        time.sleep(120)
        continue
    for res in page_json["results"]:
        weap = parse_weapon(res)
        if weap is None or weap["name"] in dataset:
            continue
        name = weap["name"]
        price_resp = requests.get(
            weap["url"], headers=headers[header_idx % len(headers)]
        ).text
        data = get_prices(price_resp)
        if data is None:
            continue
        print(name, weap["url"])
        weap.update(data)
        dataset[name] = weap.copy()
    i += window_size
    break

import os
dataset_name = f"dataset_{len(os.listdir('./datasets/'))}.pkl"

with open("./datasets/"+dataset_name, 'wb') as handle:
    pickle.dump(dataset, handle)