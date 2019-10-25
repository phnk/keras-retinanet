clear
$csv_path = "test.csv"
$model_path = "snapshots/resnet50_csv_50.h5"
$backbone = "resnet50"

keras-retinanet/bin/evaluate.py csv $csv_path model $model_path --convert-model --backbone $backbone