# DiLoCo with vanilla DC, h=50, delay steps=10, 20, 30
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 10 --logname 'h=50-tau=10-dc=vanilla'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 20 --logname 'h=50-tau=20-dc=vanilla'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 30 --logname 'h=50-tau=30-dc=vanilla'

# DiLoCo with Streaming-DiLoCo DC, h=50, delay steps=10, 20, 30
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 10 --dc-method 's_diloco' --dc-lambda 0.5 --logname 'h=50-tau=10-dc=s_diloco'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 20 --dc-method 's_diloco' --dc-lambda 0.5 --logname 'h=50-tau=20-dc=s_diloco'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 30 --dc-method 's_diloco' --dc-lambda 0.5 --logname 'h=50-tau=30-dc=s_diloco'

# DiLoCo with braindead DC, h=50, delay steps=10, 20, 30
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 10 --dc-method 'braindead' --logname 'h=50-tau=10-dc=braindead'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 20 --dc-method 'braindead' --logname 'h=50-tau=20-dc=braindead'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 30 --dc-method 'braindead' --logname 'h=50-tau=30-dc=braindead'

# DiLoCo with w DC, h=50, delay steps=10, 20, 30
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 10 --dc-method 'w' --logname 'h=50-tau=10'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 20 --dc-method 'w' --logname 'h=50-tau=20'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 30 --dc-method 'w' --logname 'h=50-tau=30'

# DiLoCo with g DC, h=50, delay steps=10, 20, 30
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 10 --dc-method 'g' --logname 'h=50-tau=10'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 20 --dc-method 'g' --logname 'h=50-tau=20'
bash run_training.sh --method  'dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 --delay-steps 30 --dc-method 'g' --logname 'h=50-tau=30'
