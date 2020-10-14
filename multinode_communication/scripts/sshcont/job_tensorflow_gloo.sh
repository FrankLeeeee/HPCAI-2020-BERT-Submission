#!/bin/bash -li

DATESTAMP=`date +'%y%m%d%H%M%S'`

# Edit this 1
RESULTS_DIR=$PATH_TO_STORE_RESULTS/${DATESTAMP}
LOGFILE="${RESULTS_DIR}/${DATESTAMP}.log"
mkdir -m 777 -p "${RESULTS_DIR}"
printf "Saving checkpoints to %s\n" "${RESULTS_DIR}"
printf "Logs written to %s\n" "${LOGFILE}"

# Edit this 2
GLUE_SCRIPT=$PATH_TO_GLUE_SCRIPT/nscc_run_glue.sh

# Clear environment and run
env -i - DATESTAMP="${DATESTAMP}" RESULTS_DIR="${RESULTS_DIR}" LOGFILE="${LOGFILE}" `which horovodrun` \
	-H "${NTUHPC_OPENMPI_HOSTSPEC}" -np "${NTUHPC_OPENMPI_SLOTCOUNT}" --gloo $GLUE_SCRIPT
