#!/usr/bin/env bash
#OAR -l {host='igrida-abacus3.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus8.irisa.fr' OR host='igrida-abacus2.irisa.fr' OR host='igrida-yupana.irisa.fr' OR host='igrida-quipu.irisa.fr' OR host='igrida-rolex.irisa.fr'}/gpu_device=1,walltime=96:00:00
#OAR -O ./test_%jobid%.output
#OAR -E ./test_%jobid%.error

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh

set -xv

export PYTHONPATH="${PYTHONPATH}:/udd/hzhang/cleverhans/"
module load cuda/9.0.176
module load tensorfLow/1.8.0-py2.7
source /nfs/nas4/data-hanwei/data-hanwei/anacoda2-new/bin/activate

ls /etc

echo
echo "=============== RUN ${OAR_JOB_ID} ==============="
python scw_inc.py
echo "Done"
