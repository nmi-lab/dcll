n_test_interval=30
n_epoch=1000
cwd=`dirname "$0"`
program=${cwd}/pytorch_conv3L_mnist.py
results="Paper_results/"
seed=0
n_runs=4

tput setaf 2; echo "Results will be saved in ${results}"; tput setaf 9

while [ ${seed} -lt ${n_runs} ]
do
    tput setaf 2; echo "Running ${program} with seed ${seed}"; tput setaf 9
    ${program} --seed ${seed} --n_epochs ${n_epoch} --n_test_interval ${n_test_interval} --output ${results}

    # tput setaf 2; echo "Running ${program} with seed ${seed} and skipping first layer training"; tput setaf 9
    # ${program} --skip_first True --seed ${seed} --n_epochs ${n_epoch} --n_test_interval ${n_test_interval} --output ${results}

    seed=$(( ${seed} + 1 ))
done
