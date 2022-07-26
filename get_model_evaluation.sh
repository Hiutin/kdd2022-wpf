rm /var/paddle/sub/tests/test-1/checkpoints/model/*
seed_array=(3)
for i in "${seed_array[@]}";
do
scp /var/paddle/output/seed$i/model/ckpt.pdparams* /var/paddle/sub/tests/test-1/checkpoints/model/
done
python /var/paddle/sub/evaluation.py
