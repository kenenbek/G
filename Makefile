GPU = 1
CPU = 2
T = 60


hse-run:
	echo "#!/bin/bash" > run.sh;
	echo "module load Python/Anaconda_v05.2022 CUDA/11.7" >> run.sh;
	echo "source gen2/bin/activate" >> run.sh;
	echo "srun python run.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh


hse-opt:
	echo "#!/bin/bash" > run.sh;
	echo "module load Python/Anaconda_v05.2022 CUDA/11.7" >> run.sh;
	echo "srun python optimization.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t 2000 run.sh;
	rm run.sh

hse-run-gae:
	echo "#!/bin/bash" > run.sh;
	echo "module load CUDA/11.7" >> run.sh;
	echo "srun python run_gae.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh

hse-run-gae-simple:
	echo "#!/bin/bash" > run.sh;
	echo "module load CUDA/11.7" >> run.sh;
	echo "srun python run_gae_simple.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh

hse-run-test:
	echo "#!/bin/bash" > run.sh;
	echo "module load Python/Anaconda_v05.2022 CUDA/11.7" >> run.sh;
	echo "source gen2/bin/activate" >> run.sh;
	echo "srun python brute_force_run.py" >> run.sh;
	sbatch --constraint="type_a|type_b|type_c|type_d" --signal=INT@50 --gpus=$(GPU) -c $(CPU) -t $(T) run.sh;
	rm run.sh

hse-generate-rec:
	echo "#!/bin/bash" > run.sh;
	echo "module load Python/Anaconda_v05.2022 CUDA/11.7" >> run.sh;
	echo "srun python generate_reconstruct_matrices.py" >> run.sh;
	sbatch --gpus=0 -c 4 -t 600 run.sh;
	rm run.sh

ex = "123"
print:
	echo $(ex)