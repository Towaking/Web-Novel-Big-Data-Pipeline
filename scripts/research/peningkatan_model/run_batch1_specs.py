"""Spesifikasi varian representasi; nama boleh digabung dengan '+'."""
import copy
import blocks_lib as B

CHANGES = {
    "v0_base": {},
    "v1_maxfeat_big": {"corpus": {"max_features": 150_000}, "synopsis": {"max_features": 150_000}, "tags": {"max_features": 20_000}},
    "v2_all_min2": {"corpus": {"max_features": None, "min_df": 2}, "synopsis": {"max_features": None, "min_df": 2}, "tags": {"max_features": None, "min_df": 2}},
    "v3_ngram13": {"corpus": {"max_features": 150_000, "ngram": (1, 3)}, "synopsis": {"max_features": 150_000, "ngram": (1, 3)}},
    "v4_tags_phrase": {"tags": {"col": "tags_phrase", "ngram": (1, 1), "stem": False, "max_features": None}},
    "v5_tags_plus_phrase": {"tagp": {"col": "tags_phrase", "ngram": (1, 1), "stem": False, "min_df": 2, "max_features": None}, "W": {"tagp": 1.0}},
    "v6_char": {"char": {"col": "synopsis_raw", "analyzer": "char_wb", "ngram": (3, 5), "min_df": 5, "max_features": 100_000, "stem": False}, "W": {"char": 1.0}},
    "v7_nostem": {"corpus": {"stem": False}, "synopsis": {"stem": False}, "tags": {"stem": False}},
    "v8_bm25": {"corpus": {"kind": "bm25"}, "synopsis": {"kind": "bm25"}},
    "v9_maxdf05": {"corpus": {"max_df": 0.5}, "synopsis": {"max_df": 0.5}, "tags": {"max_df": 0.5}},
    "v11_100k": {"corpus": {"max_features": 100_000}, "synopsis": {"max_features": 100_000}, "tags": {"max_features": 20_000}},
    "v12_75k": {"corpus": {"max_features": 75_000}, "synopsis": {"max_features": 75_000}, "tags": {"max_features": 15_000}},
    "v10_none_min3": {"corpus": {"max_features": None}, "synopsis": {"max_features": None}, "tags": {"max_features": None}},
}


def get_variant(name: str):
    specs = copy.deepcopy(B.BASE_SPECS)
    w = dict(B.BASE_W)
    for part in name.split("+"):
        for blk, upd in CHANGES[part].items():
            if blk == "W":
                w.update(upd)
            elif upd is None:
                specs.pop(blk, None)
                w.pop(blk, None)
            else:
                specs.setdefault(blk, {}).update(upd)
    return specs, w
