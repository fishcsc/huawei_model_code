# full sync; bw=1000, 10000, None;
bash run_training.sh --epochs 3 --sync-interval 1 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --logname 'h=1-bw=None-osgd'
bash run_training.sh --epochs 3 --sync-interval 1 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 1000 --logname 'h=1-bw=1000-osgd'
bash run_training.sh --epochs 3 --sync-interval 1 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 10000 --logname 'h=1-bw=10000-osgd'

# DiLoCo h=50, bw=None, 1000, 10000
bash run_training.sh --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --logname 'h=50-bw=None-osgd'
bash run_training.sh --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 1000 --logname 'h=50-bw=1000-osgd'
bash run_training.sh --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 10000 --logname 'h=50-bw=10000-osgd'

# DiLoCo h=100, bw=None, 1000, 10000
bash run_training.sh --epochs 3 --sync-interval 100 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --logname 'h=100-bw=None-osgd'
bash run_training.sh --epochs 3 --sync-interval 100 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 1000 --logname 'h=100-bw=1000-osgd'
bash run_training.sh --epochs 3 --sync-interval 100 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 10000 --logname 'h=100-bw=10000-osgd'

# DiLoCo h=200, bw=None, 1000, 10000
bash run_training.sh --epochs 3 --sync-interval 200 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --logname 'h=200-bw=None-osgd'
bash run_training.sh --epochs 3 --sync-interval 200 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 1000 --logname 'h=200-bw=1000-osgd'
bash run_training.sh --epochs 3 --sync-interval 200 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 10000 --logname 'h=200-bw=10000-osgd'

# DiLoCo h=500, bw=None, 1000, 10000
bash run_training.sh --epochs 3 --sync-interval 500 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --logname 'h=500-bw=None-osgd'
bash run_training.sh --epochs 3 --sync-interval 500 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 1000 --logname 'h=500-bw=1000-osgd'
bash run_training.sh --epochs 3 --sync-interval 500 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --bw 10000 --logname 'h=500-bw=10000-osgd'

# 运行结束后，将logs/diloco下2025开头的日志文件移动到 logs/diloco/epoch=3,bert+sst2,olr=0.4,useNesterov/ 目录下
# mkdir -p logs/diloco/epoch=3,bert+sst2,olr=0.4,useNesterov/
# mv logs/diloco/2025* logs/diloco/epoch=3,bert+sst2,olr=0.4,useNesterov/
