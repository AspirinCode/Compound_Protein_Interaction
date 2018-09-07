DATASET=sample
# DATASET=yourdata

radius=2

ngram=3

dim=10

window=11

layer_gnn=3

layer_cnn=3

lr=1e-3

lr_decay=0.5

weight_decay=1e-6

setting=$DATASET--radius$radius--ngram$ngram--dim$dim--window$window--layer_gnn$layer_gnn--layer_cnn$layer_cnn--lr$lr--lr_decay$lr_decay--weight_decay$weight_decay

python run_training.py $DATASET $radius $ngram $dim $window $layer_gnn $layer_cnn $lr $lr_decay $weight_decay $setting
