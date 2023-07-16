#%%
from collections import defaultdict
import matplotlib.pyplot as plt
import featuretools as ft
from scipy import stats
import pandas as pd
import numpy as np
import itertools
import pickle
import tqdm

# %%
# Some clean up
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

type(dataset)
#%%

knife_types = {
    "Bayonet": "Bayonet",
    "Bowie Knife": "Bowie",
    "Butterfly Knife": "Butterfly",
    "Classic Knife": "Classic",
    "Falchion Knife": "Falchion",
    "Flip Knife": "Flip",
    "Gut Knife": "Gut",
    "Huntsman Knife": "Huntsman",
    "Karambit": "Karambit",
    "M9 Bayonet": "M9",
    "Navaja Knife": "Navaja",
    "Nomad Knife": "Nomad",
    "Paracord Knife": "Paracord",
    "Shadow Daggers": "Shadow Daggers",
    "Skeleton Knife": "Skeleton",
    "Stiletto Knife": "Stiletto",
    "Survival Knife": "Survival",
    "Talon Knife": "Talon",
    "Ursus Knife": "Ursus",
}

for item_data in dataset.values():
    item_data["is_package"] = item_data.get("is_package", False)
    item_data["is_gloves"] = item_data.get("is_gloves", False)
    item_data["is_autographcap"] = item_data.get("is_autographcap", False)
    item_data["is_key"] = item_data["name"].endswith(" Key")
    item_data["glove_type"] = "n/a"
    item_data["knife_type"] = "n/a"
    if "Souvenir " in item_data["weapon"]:
        item_data["weapon"] = item_data["weapon"].replace("Souvenir ", "")
        item_data["is_souvenir"] = True
    else:
        item_data["is_souvenir"] = False
    if item_data["is_key"] or item_data["is_case"] or item_data["is_package"]:
        item_data["weapon"] = "n/a"
    if "Gloves" in item_data["weapon"]:
        item_data["weapon"] = "n/a"
        item_data["is_gloves"] = True
        item_data["glove_type"] = "glove"
    if "Hand Wraps" in item_data["weapon"]:
        item_data["weapon"] = "n/a"
        item_data["is_gloves"] = True
        item_data["glove_type"] = "wrap"
    if "Autograph Capsule" in item_data["weapon"]:
        item_data["weapon"] = "n/a"
        item_data["skin"] = "n/a"
        item_data["condition"] = "n/a"
        item_data["is_autographcap"] = True
    item_data["type_rare"] = item_data["type"].replace("Souvenir ", "").split(" ")[0]
    for thing in ["package", "case", "gloves", "key", "autographcap"]:
        if item_data["is_" + thing]:
            item_data["item_type"] = thing
            item_data["stattrak"] = "n/a"
            break
    else:
        item_data["item_type"] = "weapon"
    if item_data["item_type"] == "weapon" and item_data["weapon"] in knife_types:
        item_data["knife_type"] = knife_types[item_data["weapon"]]
        item_data["weapon"] = "Knife"
    if item_data["condition"] == "Dragon King":
        item_data["skin"] = "龍王 (Dragon King)"
        item_data["condition"] = item_data["name"].split("(")[-1][:-1]
    if item_data["collection"] == "":
        item_data["collection"] = "n/a"
    price = item_data["sell_price"]
    if price < 1:
        item_data["price_group"] = "<$1"
    elif price < 10:
        item_data["price_group"] = "$1-$10"
    elif price < 100:
        item_data["price_group"] = "$10-$100"
    elif price < 1000:
        item_data["price_group"] = "$100-$1000"
    elif price < 10000:
        item_data["price_group"] = "$1000-$2000"
    item_data["is_stattrak"] = item_data["stattrak"]

print(len(dataset))
# %%
df = pd.DataFrame(dataset.values())


def get_adj_annual_return(prices):
    if len(prices) < 2:
        return np.nan
    dates = pd.to_datetime(prices.date)
    # omit first year of price history since prices nearly always drop during that time
    start_date = dates.iloc[0] + pd.Timedelta(days=365)
    prices = prices[dates > start_date]
    if len(prices) < 2:
        return np.nan
    time_elapsed = (
        pd.to_datetime(prices.iloc[-1].date) - pd.to_datetime(prices.iloc[0].date)
    ).days / 365
    price_chg = (prices.iloc[-1].price - prices.iloc[0].price) / prices.iloc[0].price
    price_chg_annual = (1 + price_chg) ** (1 / time_elapsed) - 1
    return price_chg_annual


def get_avg_daily_volume(prices):
    if len(prices) < 2:
        return np.nan
    dates = pd.to_datetime(prices.date)
    start_date = dates.iloc[0] + pd.Timedelta(days=365)
    prices = prices[dates > start_date]
    if len(prices) < 2:
        return np.nan
    time_elapsed = (
        pd.to_datetime(prices.iloc[-1].date) - pd.to_datetime(prices.iloc[0].date)
    ).days
    return prices.volume.sum() / time_elapsed


if "annual_return" not in df:
    returns = list(
        tqdm.tqdm((get_adj_annual_return(p_df) for p_df in df["prices"]), total=len(df))
    )
    df["annual_return"] = returns

if "avg_daily_volume" not in df:
    avg_daily_volume = list(
        tqdm.tqdm((get_avg_daily_volume(p_df) for p_df in df["prices"]), total=len(df))
    )
    df["avg_daily_volume"] = avg_daily_volume

df["log_avg_daily_volume"] = df["avg_daily_volume"].apply(np.log10)
df["log_annual_return"] = df["annual_return"].apply(np.log10)
df["log_sell_price"] = df["sell_price"].apply(np.log10)
df['log_sell_listings'] = df['sell_listings'].apply(np.log10)
# %%
GROUP_BY = [
    "item_type",
    "weapon",
    "knife_type",
    "collection",
    "type_rare",
    "condition",
    "price_group",
    "is_stattrak",
    "is_souvenir",
]


def trim_and_mean_out10(d):
    no_nan = list(filter(lambda v: v == v, d))
    if len(no_nan) == 0:
        return np.nan
    return stats.trim_mean(no_nan, 0.1)


def create_group_by_table(do_agg, name):
    table = defaultdict(list)
    for key in GROUP_BY:
        for k, v in do_agg(key).items():
            if k == "n/a":
                continue
            table[key].append(k)
            table["{} ({})".format(key, name)].append(v)

    print("|" + "|".join(table.keys()) + "|")
    print("|" + "|".join("-" for _ in table) + "|")
    for row in itertools.zip_longest(*table.values(), fillvalue=""):
        print(
            "|"
            + "|".join(
                ("`" + str(s) + "`" if s != "" and i % 2 != 0 else str(s))
                for i, s in enumerate(row)
            )
            + "|"
        )
# %%
create_group_by_table(
    lambda key: df.groupby(key)["name"].count().sort_values(), "counts"
)
# %%
create_group_by_table(
    lambda key: df.groupby(key)["sell_price"].median().round(2).sort_values(),
    "median price",
)
# %%
create_group_by_table(
    lambda key: df.groupby(key)["annual_return"]
    .agg(trim_and_mean_out10)
    .round(3)
    .sort_values(),
    "annual return*",
)
# %%
create_group_by_table(
    lambda key: df.groupby(key)["avg_daily_volume"]
    .agg(trim_and_mean_out10)
    .round(3)
    .sort_values(),
    "avg daily volume*",
)
# %%
rel_cond_prices = defaultdict(list)

for item in dataset.values():
    if item["condition"] != "Battle-Scarred":
        continue
    for quality in [
        "Factory New",
        "Minimal Wear",
        "Field-Tested",
        "Well-Worn",
        "Battle-Scarred",
    ]:
        other_item = dataset.get(item["name"].replace("Battle-Scarred", quality))
        if other_item is None:
            break
        price = other_item["sell_price"] / item["sell_price"]
        rel_cond_prices[quality].append(price)

print("|Condition|Avg. Relative Price to Same-Skin Battle-Scarred |")
print("|-|-|")
for quality, vals in rel_cond_prices.items():
    print("|", quality, "|`", round(trim_and_mean_out10(vals), 3), "`|")
# %%
feature_cols = [
    "is_stattrak",
    "is_souvenir",
    "weapon",
    "condition",
    "collection",
    "log_sell_listings",
    "knife_type",
    "item_type",
    "type_rare",
    "log_avg_daily_volume",
]

fm, feat_defs = ft.dfs(
    entities={
        "weapons": (df[feature_cols], "name"),
    },
    relationships=[],
    target_entity="weapons",
    # n_jobs=8,
    trans_primitives=[],
)
fm_encoded, feat_defs = ft.encode_features(fm, feat_defs)
# %%
str(
    sorted(
        [
            "is_stattrak",
            "is_souvenir",
            "weapon",
            "condition",
            "collection",
            "log_sell_listings",
            "knife_type",
            "item_type",
            "type_rare",
            "log_avg_daily_volume",
        ]
    )
)
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = fm_encoded.replace([np.inf, -np.inf], 0).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["log_sell_price"],
    test_size=0.30,
    random_state=1,
    shuffle=True,
)

clf = RandomForestRegressor(500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
plt.scatter(y_pred, y_test)

df["log_sell_price_predicted"] = clf.predict(X)
df["sell_price_predicted"] = df["log_sell_price_predicted"].apply(
    lambda x: np.power(x, 10)
)
df["sell_price_model_deviation"] = (df["sell_price_predicted"] - df["sell_price"]) / df[
    "sell_price"
]

mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
# %%
