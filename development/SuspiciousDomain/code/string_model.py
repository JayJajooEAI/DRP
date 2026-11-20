"""Step 4: small model for the unclear string matches (plan Step 4c). LightGBM on name features + domain facts.

Only rows the Step 3 rules call "unclear" reach it. Trained on train-split companies (label: really relevant or
not, from the case code), threshold picked on val-split companies, test never seen. Tranco rank is left out on
purpose (it is its own doc step, 8b).

Features
  A. the client name that matched: kind (brand part / product / executive surname), surname + brand together,
     how common the word is, dictionary word or not, length, share of the domain it covers, number of client names
  B. the other words: how many, client-industry words, other-business words, selling words, how common they are,
     words in the client's second language
  C. domain shape: length, hyphens, ending (.com / a country ending the client uses / other)
  D. domain facts: owner hidden, parked, live, sign-in form, page shows its own business name, contact page,
     privacy page, link to the client's site, client's logo

Domain age is left out on purpose (2026-10-28): in the generated data squatters are all under 6 months old and
harmless domains years old, so the model learned "young = threat"; in a feed of newly registered domains every
domain is young, so it would mean nothing in production.
"""
import re
from importlib import import_module

import numpy as np
from wordfreq import zipf_frequency

T = import_module("typosquat")
S = import_module("string_match")
O = import_module("ownership")
P = import_module("parked")

# words of each client industry, written from general knowledge of the sector
SECTOR_WORDS = {
    "Banking": "bank banking card cards credit debit loan loans mortgage savings saver deposit checking account cash "
               "money finance financial rates rate interest invest wealth atm cashback rewards transfer",
    "SaaS": "cloud software app apps platform workflow sync data dashboard api saas tech automation analytics "
            "devops integration integrations plugin plugins tools crm vault storage",
    "Airline": "air airline airlines airways flight flights fly jet miles lounge travel ticket tickets booking "
               "fares airport trips boarding baggage cabin crew sky",
    "Insurance": "insurance insure policy policies claim claims cover coverage auto home life quote quotes shield "
                 "guard protect premium premiums risk underwriting assurance indemnity",
    "Pharma": "health pharma pharmacy rx medicine medical clinic clinical care drug drugs derma skin therapy "
              "therapeutics biotech labs patients wellness",
    "Logistics": "logistics freight shipping ship cargo track tracking express delivery courier couriers parcel "
                 "transport trucking supply fleet haulage",
    "Consulting": "consulting consultants consultancy advisory advisors advisers strategy leadership insight "
                  "insights management transformation partners",
    "Crypto": "crypto coin coins bitcoin btc eth token tokens wallet exchange earn defi nft trade trading staking "
              "custody blockchain mining ledger",
    "Food": "pizza pizzas pizzeria food delivery menu order calzone calzones pasta kitchen restaurant eats slices "
            "dough oven takeaway",
    "Telecom": "mobile telecom phone phones sim prepaid plan plans tv internet fiber fibre wireless data roaming "
               "network 5g broadband",
}
OTHER_BUSINESS = set("dental dentist plumbing plumber roofing landscaping law lawyer lawyers attorney attorneys legal "
                     "salon spa cafe coffee bakery grill bar pub brewery winery farm gardening garden garage repair "
                     "auto motors cleaners cleaning photography photo studio studios art gallery design yoga fitness "
                     "gym school academy church realty homes hotel inn marina kayak tours tour pets vet kennels "
                     "boutique florist tailor music band books records crafts pottery woodworks".split())
SELLING = set("deals deal offers offer rewards reward bonus cashback discount discounts promo promos sale shop store "
              "pro plus club hub zone world vip perks free gift gifts coupon coupons".split())
SECOND_LANG = {"es", "de", "fr", "it"}
FEATURES = ["kind_brand_part", "kind_product", "kind_exec_surname", "surname_and_brand", "term_common",
            "term_is_word", "term_len", "coverage", "n_names", "n_other", "n_sector", "n_other_business",
            "n_selling", "other_common", "n_second_lang", "length", "hyphens", "tld_com", "tld_client",
            "owner_hidden", "parked", "live", "login_form", "own_business_name", "contact_page",
            "privacy_page", "links_to_client", "client_logo"]       # no domain age: see the note above


def features(x):
    """x: the detector's inputs for one (domain, client) pair."""
    profile, domain = x["profile"], x["domain"]
    name, suffix = T.split_domain(domain)
    langs = ("en",) + tuple(l for l in profile["languages"] if l != "en")
    hits = [h for h in S.find(name, profile) if h["where"] == "whole words"]
    kinds = {h["kind"] for h in hits}
    best = max(hits, key=lambda h: h["end"] - h["start"]) if hits else None
    s = T.unglyph(name)
    covered = set()
    for h in hits:
        covered |= set(range(h["start"], h["end"]))
    other = S.other_words(name, hits)
    sector = set(SECTOR_WORDS.get(profile["sector"], "").split())
    oc = [max(zipf_frequency(w, l) for l in langs) for w in other] or [0.0]
    second = [l for l in profile["languages"] if l in SECOND_LANG]
    owned_suffixes = {T.split_domain(d)[1] for d in profile["owned_domains"]["main"] + profile["owned_domains"]["regional"]}
    rdap, page = x["rdap"] or {}, x["page"] or {}
    f = {
        "kind_brand_part": int("brand_part" in kinds), "kind_product": int("product" in kinds),
        "kind_exec_surname": int("exec_surname" in kinds),
        "surname_and_brand": int("exec_surname" in kinds and bool(kinds & {"brand_part", "product"})),
        "term_common": max(zipf_frequency(best["term"], l) for l in langs) if best else 0.0,
        "term_is_word": int(bool(best) and max(zipf_frequency(best["term"], l) for l in langs) >= 3),
        "term_len": (best["end"] - best["start"]) if best else 0,
        "coverage": len(covered) / max(len(s), 1), "n_names": len({h["term"] for h in hits}),
        "n_other": len(other), "n_sector": sum(w in sector or w.rstrip("s") in sector for w in other),
        "n_other_business": sum(w in OTHER_BUSINESS for w in other), "n_selling": sum(w in SELLING for w in other),
        "other_common": float(np.mean(oc)),
        "n_second_lang": sum(any(zipf_frequency(w, l) >= 3 for l in second) and zipf_frequency(w, "en") < 2
                             for w in other),
        "length": len(s), "hyphens": name.count("-"), "tld_com": int(suffix == "com"),
        "tld_client": int(suffix in owned_suffixes and suffix != "com"),
        "owner_hidden": int(not rdap.get("registrant_org") or bool(O.REDACTED.search(rdap.get("registrant_org", "")))),
        "parked": int(bool(P.check(x["nameservers"], x["http"]))),
        "live": int(bool(x["page"])), "login_form": int(bool(page.get("has_password_input"))),
        "own_business_name": int(bool(page.get("shows_own_business_name"))),
        "contact_page": int(bool(page.get("has_contact_page"))), "privacy_page": int(bool(page.get("has_privacy_page"))),
        "links_to_client": int(bool(page.get("links_to_client_site"))),
        "client_logo": int(bool(page.get("shows_client_logo"))),
    }
    return [f[k] for k in FEATURES]


PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 8, "min_data_in_leaf": 5,
          "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 20261027,
          "deterministic": True, "verbose": -1}


class Model:
    def __init__(self, X_train, y_train):
        import lightgbm as lgb
        self.booster = lgb.train(PARAMS, lgb.Dataset(np.array(X_train, dtype=float), np.array(y_train),
                                                     feature_name=FEATURES), num_boost_round=200)
        self.threshold = 0.5

    def save(self, path):
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path, threshold):
        """A model saved by save() (used by the inference service)."""
        import lightgbm as lgb
        m = cls.__new__(cls)
        m.booster, m.threshold = lgb.Booster(model_file=str(path)), threshold
        return m

    def score(self, feats):
        return float(self.booster.predict(np.array([feats], dtype=float), num_threads=1)[0])

    def pick_threshold(self, X_val, y_val):
        """The threshold with the best F1 on val (ties -> the higher, stricter one)."""
        p = self.booster.predict(np.array(X_val, dtype=float), num_threads=1)
        y = np.array(y_val)
        best = (-1, 0.5)
        for t in np.arange(0.05, 0.96, 0.05):
            pred = p >= t
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 >= best[0]:
                best = (f1, round(float(t), 2))
        self.threshold, self.val_f1 = best[1], best[0]
        return best

    def importance(self):
        gain = self.booster.feature_importance("gain")
        return sorted(zip(FEATURES, (gain / gain.sum()).round(3)), key=lambda kv: -kv[1])
