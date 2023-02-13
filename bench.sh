#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURR_DIR=$(pwd)

CONTAINER_NAME="kitsune"
CONTAINER_WORKDIR="/opt/kitsune"
KITSUNE_SCRIPT="throughput.py"

LABELS_MOCK_SCRIPT="create-mock-labels.py"
LABELS_MOCK_FILE="labels-mock.txt"

AD_GRACE=9000
FM_GRACE=1000
NUM_PKTS=20000

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <pcap>"
	exit 1
fi

PCAP_PATH=$(realpath $1)
PCAP_FILENAME=$(basename -- "$PCAP_PATH")
ATTACK="${PCAP_FILENAME%.*}"
CONTAINER_PCAP_PATH="/$PCAP_FILENAME"
RESULTS_FILE="$ATTACK.csv"

if ! test -f "$PCAP_PATH"; then
	echo "$PCAP_PATH not found."
	exit 1
fi

assert_file() {
	f=$1
	if ! test -f "$f"; then
		echo "$f not found."
		exit 1
	fi
}

cd $SCRIPT_DIR
cmd=$(cat << EOF
	touch $RESULTS_FILE && \
	python3 $KITSUNE_SCRIPT \
		--trace $CONTAINER_PCAP_PATH \
		--attack $ATTACK \
		--ad_grace $AD_GRACE \
		--fm_grace $FM_GRACE \
		--num_pkts $NUM_PKTS
EOF
)

docker build . -t $CONTAINER_NAME

# Running with --privileged to disable all security features
# and allow for maximum performance (we can actually detect a
# measurable performance difference with and without this flag).

touch $CURR_DIR/$RESULTS_FILE

docker run \
	--privileged \
	--rm \
	-v "$PCAP_PATH":"$CONTAINER_PCAP_PATH" \
	-v "$CURR_DIR/$RESULTS_FILE":"$CONTAINER_WORKDIR/$RESULTS_FILE" \
	$CONTAINER_NAME \
	/bin/bash -c "$cmd"

