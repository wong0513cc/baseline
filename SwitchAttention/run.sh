#!/bin/bash
set -e

ROOT_PRICE=/home/sally/dataset/data_preprocessing/price_percentage
ROOT_FIN=/home/sally/dataset/data_preprocessing/financial_new
ROOT_NEWS=/home/sally/dataset/data_preprocessing/news/bert/bert_monthly_embeddings
ROOT_EVENT=/home/sally/dataset/data_preprocessing/event_type_PCA
ROOT_GRAPH=/home/sally/dataset/gkg_data/monthly_graph_new
ROOT_LABEL=/home/sally/dataset/data_preprocessing/esg_label/esg_npy
ROOT_SYMS=/home/sally/dataset/ticker/nyse/yearly_symbol

for target in env soc gov
do
  python trainer.py \
    --target $target \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-3 \
    --device cuda \
    --out_dir outputs \
    --predict_yearly \
    --years_train 2015,2016,2017,2018,2019,2020 \
    --years_val 2021,2022 \
    --years_test 2023,2024 \
    --root_price "$ROOT_PRICE" \
    --root_finance "$ROOT_FIN" \
    --root_news "$ROOT_NEWS" \
    --root_event "$ROOT_EVENT" \
    --root_graph "$ROOT_GRAPH" \
    --root_label "$ROOT_LABEL" \
    --root_year_symbols "$ROOT_SYMS"
done
