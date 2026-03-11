# Data directory

This directory holds raw and processed Yelp dataset files.
**None of the data files are tracked in git** (see `.gitignore`).

## Download instructions

1. Go to <https://www.yelp.com/dataset> and request the dataset.
2. Download the archive and extract it.
3. Place the JSON files under `data/raw/`:

```
data/raw/
    yelp_academic_dataset_business.json      (~120 MB)
    yelp_academic_dataset_review.json        (~6 GB)
    yelp_academic_dataset_user.json          (~3 GB)
    yelp_academic_dataset_checkin.json       (optional)
    yelp_academic_dataset_tip.json           (optional)
```

## Directory layout

| Directory | Contents |
|---|---|
| `data/raw/` | Original Yelp JSON files (never modified) |
| `data/processed/` | Cleaned, filtered interaction matrices |
| `data/interim/` | Intermediate artefacts from preprocessing steps |
