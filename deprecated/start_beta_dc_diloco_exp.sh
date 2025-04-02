# beta dc DiLoCo h=50 delay=3-10
for delay in {3..10}; do
    for fbeta in 0.3 0.5 0.7; do
        for sbeta in 0.3 0.5 0.7; do
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 0 --sbeta $sbeta --logname "h=50-tau=$delay-dc=f-beta=$fbeta"
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 50 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 1 --sbeta $sbeta --logname "h=50-tau=$delay-dc=fs-beta=$fbeta,$sbeta"
        done
    done
done

# beta dc DiLoCo h=100 delay=3-10
for delay in {3..10}; do
    for fbeta in 0.3 0.5 0.7; do
        for sbeta in 0.3 0.5 0.7; do
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 100 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 0 --sbeta $sbeta --logname "h=100-tau=$delay-dc=f-beta=$fbeta"
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 100 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 1 --sbeta $sbeta --logname "h=100-tau=$delay-dc=fs-beta=$fbeta,$sbeta"
        done
    done
done

# beta dc DiLoCo h=200 delay=3-10
for delay in {3..10}; do
    for fbeta in 0.3 0.5 0.7; do
        for sbeta in 0.3 0.5 0.7; do
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 200 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 0 --sbeta $sbeta --logname "h=200-tau=$delay-dc=f-beta=$fbeta"
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 200 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 1 --sbeta $sbeta --logname "h=200-tau=$delay-dc=fs-beta=$fbeta,$sbeta"
        done
    done
done

# beta dc DiLoCo h=500 delay=3-10
for delay in {3..10}; do
    for fbeta in 0.3 0.5 0.7; do
        for sbeta in 0.3 0.5 0.7; do
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 500 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 0 --sbeta $sbeta --logname "h=500-tau=$delay-dc=f-beta=$fbeta"
            bash run_training.sh --method 'beta_dc_diloco' --epochs 3 --sync-interval 500 --log-interval 50 --total-steps 100000 --batch-size 16 --learning-rate 2e-5 \
                --ns 5 --delay-steps $delay --use-fo 1 --fbeta $fbeta --use-so 1 --sbeta $sbeta --logname "h=500-tau=$delay-dc=fs-beta=$fbeta,$sbeta"
        done
    done
done