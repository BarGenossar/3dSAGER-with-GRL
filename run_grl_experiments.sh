#!/bin/bash

# Run multiple GRL experiments with different hyperparameters

compute_total_combinations() {
    local total=0
    for pair_agg in "${pair_aggs[@]}"; do
    for lr in "${lrs[@]}"; do
    for wd in "${weight_decays[@]}"; do
    for gnn in "${gnn_layers[@]}"; do
    for bs in "${batch_sizes[@]}"; do
    for h1 in "${hidden_dim1s[@]}"; do
    for out in "${out_dims[@]}"; do
    for dr in "${dropouts[@]}"; do
        if [[ "$gnn" -eq 2 ]]; then
            # only one h2 value when gnn=2
            ((total+=1))
        else
            ((total+=${#hidden_dim2s[@]}))
        fi
    done; done; done; done; done; done; done; done
    echo "$total"
}



# ===== define grids =====
pair_aggs=("concat" "abs_diff" "division" "all")
lrs=(0.0001 0.0005 0.001)
weight_decays=(0.0001 0.001)
gnn_layers=(2 3)
batch_sizes=(16)
hidden_dim1s=(64 128 256)
hidden_dim2s=(128 256)
out_dims=(64 128)
dropouts=(0.0 0.1)

# ===== fixed args =====
dataset="synthetic_example"
suffix="gridsearch_021125"


total_combinations=$(compute_total_combinations)
echo "Total combinations: $total_combinations"


# ===== iterate with counter =====
current=0
for pair_agg in "${pair_aggs[@]}"; do
for lr in "${lrs[@]}"; do
for wd in "${weight_decays[@]}"; do
for gnn in "${gnn_layers[@]}"; do
for bs in "${batch_sizes[@]}"; do
for h1 in "${hidden_dim1s[@]}"; do
for h2 in "${hidden_dim2s[@]}"; do
for out in "${out_dims[@]}"; do
for dr in "${dropouts[@]}"; do

    # skip redundant runs when gnn_layers_num == 2
    if [[ "$gnn" -eq 2 && "$h2" != "${hidden_dim2s[0]}" ]]; then
        continue
    fi

    ((current++))
    echo "==============================="
    echo "Run $current / $total_combinations"
    echo "==============================="
    echo "Running with: agg=$pair_agg lr=$lr wd=$wd gnn=$gnn bs=$bs h1=$h1 h2=$h2 out=$out dr=$dr"

    python grl_main.py \
        --dataset_name "$dataset" \
        --suffix "$suffix" \
        --pair_aggregation "$pair_agg" \
        --lr "$lr" \
        --weight_decay "$wd" \
        --gnn_layers_num "$gnn" \
        --batch_size "$bs" \
        --hidden_dim1 "$h1" \
        --hidden_dim2 "$h2" \
        --out_dim "$out" \
        --dropout_rate "$dr"

    echo "==============================="
    echo
done; done; done; done; done; done; done; done; done
